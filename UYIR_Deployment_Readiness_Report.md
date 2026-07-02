# UYIR — Deployment Readiness & Improvement Report
**Prepared for:** Coimbatore Traffic Police Control Room Pilot
**Scope:** Full codebase review (detection pipeline, fusion logic, web app, stream pipeline, UI) + literature grounding
**Date:** July 2026

---

## 1. Executive Summary

UYIR is architecturally sound: a DL hard-gate + three-phase kinematic vote + weighted fusion is a defensible, explainable design that mirrors patterns used in published CCTV accident-detection work (trajectory-intersection-then-stop logic, optical-flow anomaly confirmation, multi-signal fusion). The code has clearly been through real debugging cycles — the padding-direction fix, the `lstm_peak` warmup guard, and the "recently-moving" Phase B guard are all correct, non-obvious fixes that address real failure modes.

**However, the system is not yet ready for unsupervised real-time deployment in a police control room.** The three gaps already identified in prior sessions (no camera calibration, no benchmark evaluation, no production-grade live-camera hardening) are real and are the highest-priority blockers. This review found several additional, concrete issues during a full pass of every module — some are correctness bugs, several are deployment-blocking (compute, security, data quality), and others are UX gaps that matter specifically because this is going into a *police* control room, not a research demo.

**Bottom line:** Do not point this at live Gandhipuram/Coimbatore RTSP feeds unsupervised yet. Run it as an **operator-supervised, alert-suggestion tool** during a pilot, fix the items in §4 (Critical) first, and use the KPIs in §7 as the go/no-go gate for autonomous alerting.

---

## 2. How the Design Maps to the Literature

### 2.1 Phase A — Proximity + TTC
Distance-gated proximity followed by time-to-collision is standard (NJIT-style conflict detection, closing-speed TTC). The physics is correct. The problem is not the formula — it's that **TTC and proximity are computed in raw pixels**, not real-world units. Published TTC/speed-estimation work is unanimous that this is the single most calibration-sensitive step in any vision-based conflict system: *"speed estimation is highly sensitive to the calibration quality"* and virtually every serious deployment uses a homography (planar road assumption) to convert pixel tracks to metric coordinates before computing speed, TTC, or proximity thresholds (Revaud et al., ICCV 2021; Sochor et al.). Two cameras at different heights/angles/zoom will produce wildly different `PROXIMITY_THRESHOLD`/`TTC_MAX_FRAMES` behavior for the same physical distance — this is exactly the gap already flagged in memory, and the literature confirms it's not optional for multi-camera deployment.

### 2.2 Phase B — Trajectory Conflict (IITH-style)
The "intersection alone ≠ collision, intersection + stop = collision" rule, plus treating emergency-stop as an independent signal, matches the CCTV accident-detection literature closely (e.g. *"Computer Vision-based Accident Detection in Traffic Surveillance"*, arXiv:1911.10037 — trajectory anomaly after vehicle overlap, tuned for a high detection rate / low false-alarm rate on general CCTV footage across weather conditions). This is a good match for a fixed-camera junction system and is one of the stronger parts of the design.

### 2.3 Phase C — Optical Flow Anomaly
Farneback flow magnitude spike + angular dispersion + multi-frame consistency is consistent with classical flow-based incident detection (HFG-style magnitude spikes; fuzzy/dispersion-based chaos detection). Reasonable, but see §4.6 on compute cost.

### 2.4 CNN-BiLSTM as a hard gate, not a fusion weight
This is a defensible and correct architectural decision *given the current training data*: with `FUSION_WEIGHTS["cnn_lstm"] = 0.0`, an under-trained or domain-mismatched DL model can only gate detections closed, not manufacture false positives via a nonzero fusion weight. Good design discipline. But it means **all the weight of "is this really an accident" sits on the three kinematic phases**, so the calibration problem in §2.1 has outsized impact on real-world accuracy.

### 2.5 Benchmark datasets — a correction to the roadmap
The prior research report lists CADP, DoTA, DAD, and the ACCIDENT dataset as benchmark targets. On closer reading of the current literature, **these are not equivalent**:

| Dataset | Camera type | Fit for UYIR |
|---|---|---|
| **DAD**, **DoTA**, **CCD**, **A3D**, **DADA-2000** | Dashcam / ego-centric, moving camera | **Poor fit.** These are built for *anticipation* (predicting a crash before it happens from a moving driver's-eye view) and the accident is deliberately placed near the end of a short 5s clip. UYIR is a *fixed, top-down/oblique junction camera* system — the motion statistics, occlusion patterns, and camera geometry are fundamentally different. |
| **CADP** | Fixed CCTV, real-world | **Good fit** — closest existing public analogue to Coimbatore junction cameras. |
| **ACCIDENT** (2026 benchmark) | Fixed CCTV-style viewpoints (highways, signalized intersections, roundabouts), mixed synthetic + real | **Good fit**, purpose-built for this exact camera geometry, and current. |
| **UVH-26**, **IITM-HeTra**, **DriveIndia** | Indian CCTV / mixed traffic | **Best fit for domain adaptation** — these specifically capture Indian heterogeneous traffic (autorickshaws, overloaded two-wheelers, lack of lane discipline), which COCO-trained YOLOv8n and a non-Indian-trained CNN-BiLSTM will not have seen. |

**Recommendation:** Re-prioritize evaluation toward CADP + ACCIDENT + Indian CCTV sets (UVH-26, IITM-HeTra) over DAD/DoTA. Dashcam benchmarks will not tell you much about false-positive behavior on a Gandhipuram junction feed.

### 2.6 Indian traffic domain gap — now with numbers
This gap was already flagged qualitatively; the literature makes it concrete and higher-priority than it may have seemed:
- Two-wheelers account for roughly **44.5% of Indian road traffic fatalities**, more than any other vehicle category (per the two-wheeler violation detection literature).
- COCO's classes (`car`, `bike`(motorcycle), `bus`, `truck`, `person`) have **no class for autorickshaws, cycle-rickshaws, or overloaded/triple-riding two-wheelers**, all common in Coimbatore traffic. YOLOv8n will silently misclassify or miss these, which directly degrades Phase A/B (wrong `pair_threshold`, wrong track continuity) for the vehicle type most involved in fatal accidents.
- Public Indian CCTV datasets (UVH-26 especially) exist now and are directly usable for fine-tuning detection and validating false-positive rates on Indian-density traffic before pilot go-live.

---

## 3. Codebase Deep-Dive Findings

### 3.1 Critical (fix before any live camera pilot)

1. **Hardcoded CPU inference (`model.py`: `DEVICE = torch.device("cpu")`).**
   Every frame runs YOLOv8n detection + ByteTrack, EfficientNet-B0 feature extraction, a BiLSTM forward pass, *and* full-frame Farneback dense optical flow — all on CPU, for every camera process. This combination is heavy even on a workstation; for a 24/7 multi-camera control-room deployment on typical server/edge hardware it is very unlikely to sustain real-time FPS at `FRAME_WIDTH=1280 / FRAME_HEIGHT=720`. If throughput can't keep up, `FRAME_SKIP`/frame drops silently increase and the effective detection latency (and the risk of missing a fast event between processed frames) grows without any alarm being raised. **Action:** make `DEVICE` configurable, benchmark actual sustained FPS per camera on real target hardware, and size GPU/CPU budget for however many concurrent Coimbatore feeds are planned before committing to a hardware order.

2. **No authentication/authorization on `app.py`.**
   `/api/incidents` (DELETE), `/api/incidents/{id}` (DELETE), `/train-model`, `/log-feature`, `/start-stream` have zero auth. Anyone on the network segment can wipe the entire incident history (`clearAllIncidents`) or retrain the refinement model with garbage data. For a system whose output may inform police response and potentially legal/evidentiary use, this is a hard blocker, not a nice-to-have.

3. **`accident_features.csv` (XGBoost training data) is degenerate.**
   Rows 26–50 are a duplicated, near-identical feature vector (`0.0587…, 0.0, 0.21, 0.0454…`) with **contradictory labels** — the exact same feature vector appears labeled both `1` and `0` multiple times. Training XGBoost on this will not learn a meaningful decision boundary for that region of feature space and will likely just memorize noise/pick a default class. Any deployment currently relying on `model_output/accident_xgboost.json` should treat its refinement scores as unreliable until this file is cleaned and rebuilt from diverse, correctly-labeled Coimbatore footage.

4. **No pixel-to-metric camera calibration (already known, reconfirmed above in §2.1).**
   Every threshold that matters for cross-camera consistency (`PROXIMITY_THRESHOLD`, `TTC_MAX_FRAMES`, `RECENTLY_MOVING_MIN_SPEED`, congestion-suppression's `avg_scene_speed < 5.0`) is expressed in raw px/frame. These will need **per-camera recalibration** for every Coimbatore junction with a different mounting height/angle/zoom, or the system needs a homography step so the same physical thresholds work everywhere. Without this, expect wildly inconsistent false-positive/false-negative rates camera-to-camera.

5. **Naming collision risk: two different "accident_model" files.**
   `config.ACCIDENT_MODEL_PATH = "accident_model.pt"` (optional Stage-1 YOLO gate) vs. `model_output/accident_model.pth` (the CNN-BiLSTM checkpoint). These are unrelated models with near-identical names in the same project. In an operational handover to police IT staff, this is a very easy mix-up (e.g. accidentally deleting/replacing the wrong one during an update). **Action:** rename one, e.g. `stage1_yolo_gate.pt`.

### 3.2 High priority (fix before unsupervised alerting; fine for supervised pilot)

6. **Global per-camera cooldown can hide a second real accident.** `COOLDOWN_SECONDS = 20` suppresses *all* new confirmed alerts on a camera for 20s after one fires. A secondary collision (chain-reaction pileup, a second vehicle hitting the first incident) within that window will not generate a new incident record or clip. Recommend making cooldown spatial/pair-scoped (e.g. keyed by rough incident location or involved track IDs) rather than global to the whole camera.

7. **No supervising process / crash recovery documented.** `stream_processor.py`'s reconnect loop gives up and exits the whole pipeline (`break`) after 5 failed reconnect attempts (~10s). For 24/7 operation this needs an external supervisor (systemd unit with `Restart=always`, or a process manager) — currently undocumented as an operational requirement.

8. **No per-camera config injection.** `CAMERA_ID`, `CAMERA_LOCATION`, `RTSP_URL` are hardcoded in `config.py`. Running N Coimbatore junction cameras means either N code forks or a wrapper that overrides these via CLI/env — `stream_processor.py` only accepts `--source`, not camera ID/location. Needs a small refactor (env vars or a per-camera YAML/JSON config) before multi-camera rollout.

9. **Unbounded thread spawning under alert bursts.** `FirebaseUploader` and the SSE incident pipeline spawn a raw `threading.Thread` per event/clip/LLM-call with no pool or cap. A burst of near-simultaneous incidents (e.g. a multi-vehicle pileup, or a mis-tuned camera producing rapid false triggers) could spawn many concurrent threads doing ffmpeg transcodes and network I/O. A bounded thread pool / queue would be safer for unattended operation.

10. **Weather/lighting robustness is untested.** Farneback optical flow and YOLO detection both degrade under rain, glare, fog, and low light/night — all realistic for Coimbatore (monsoon season, dusk/dawn glare at junctions). No day/night/rain evaluation exists yet. This should be part of the stratified evaluation, not assumed to generalize from daytime test clips.

11. **DL checkpoint provenance/generalization unknown.** `training_history.json` shows only 10 epochs with a non-monotonic validation loss (dips, rises, dips again) — consistent with a fairly small/narrow training set. Combined with the domain gap noted in §2.6, this checkpoint should be validated (not assumed) against Indian CCTV footage, and ideally fine-tuned on some Coimbatore-representative clips before being trusted as the hard gate.

### 3.3 Medium priority (quality/maintainability, not correctness-critical)

12. **`VEHICLE_CLASSES`/`TARGET_CLASSES` inconsistency in `config.py`.** `VEHICLE_CLASSES` includes `"motorcycle"` and `"auto"`, but `TARGET_CLASSES` (the actual YOLO class-id→label mapping) never produces those strings — only `"bike"`. The `"motorcycle"`/`"auto"` entries are dead code and could mislead a future maintainer into thinking autorickshaws are already handled (they are not — see §2.6).

13. **`import math` placed at the bottom of `accident_detector.py`**, after the class body that uses it. Works today because Python resolves the name at call time, not at class-definition time, but it's a fragile pattern — move it to the top with the other imports.

14. **Inconsistent logging.** `stream_processor.py` and `health_monitor.py` use the `logging` module properly; `accident_detector.py`, `data_logger.py`, `threshold_analyzer.py`, and large parts of `app.py` use bare `print()`. For a 24/7 system, standardize on structured logging (rotating file handler, consistent levels) so control-room IT can actually monitor system health without grepping stdout.

15. **`threshold_analyzer.py` references `trajectory_deviation_px` with a hardcoded `40.0` threshold** rather than reading a `config.py` constant, inconsistent with the project's own stated convention ("All tunable thresholds live in `config.py`. Do not hardcode values in other files" — from `tech_Des.md`).

16. **`test_xgboost.json` appears to be a stale/leftover artifact** (a near-trivial 3-node-per-tree, near-zero-weight model) sitting alongside the real `accident_xgboost.json`. Not loaded by `app.py`, but its presence in `model_output/` is confusing housekeeping — recommend removing or clearly labeling it as a test fixture.

### 3.4 Data governance / compliance (worth flagging explicitly for a police deployment)

17. Incident snapshots and ±5s clips (potentially showing injured people) are stored locally under `static/uploads/incidents/` and optionally embedded as base64 in Firestore, with **no documented retention policy, encryption-at-rest, or access control**. For a police-operated system handling imagery of real accidents/victims, this needs an explicit data-handling policy (retention period, who can view/export, audit logging of access) — this is as much a legal/procedural gap as a technical one, and worth raising with whoever owns compliance for the pilot (relevant under India's DPDP Act 2023 given personal/imagery data of identifiable individuals).

---

## 4. UI / Operator Experience

The existing "pending / confirmed / suspicious / suppressed" tiering (already noted in memory as a good foundation) is genuinely a reasonable start toward a verification-tier UI, and the toast + SSE live-update mechanism is solid engineering. For an actual control-room rollout, though, the interface as it exists is still a **single-video upload/playback dashboard**, not a live multi-camera watch-floor tool. Concrete gaps:

- **No multi-camera grid view.** A control room monitoring multiple Coimbatore junctions needs one screen showing live thumbnails/status for all cameras simultaneously, with the ability to "pop out" any camera on an alert. The current UI is built around uploading one video/image at a time.
- **No operator acknowledgment workflow.** Incidents can only be viewed or deleted — there's no "Acknowledge → Dispatch → Resolve (false alarm / confirmed, with reason code)" state machine. This matters for two reasons: (a) it gives police a proper audit trail of response times (important for a pilot evaluation and for any after-the-fact review), and (b) the reason codes an operator enters on dismissal are exactly the labeled data you need to keep improving Phase A/B/C thresholds and the XGBoost refinement model over time — right now that feedback loop doesn't exist in the live pipeline (only the manual "Log for Model Improvement" buttons on the upload-testing UI).
- **No camera health/status indication.** If an RTSP feed drops (see §3.2 item 7), there's currently no UI surface showing "Camera 3: offline since 14:02" — an operator would only notice by the feed going stale.
- **No escalation/audible alarm differentiation.** For a control room, a confirmed accident should be very hard to miss (persistent audible alert until acknowledged), distinct from "suspicious"/"pending" which can stay visual-only. The current toast auto-dismisses after 5 seconds regardless of severity.
- **Threshold slider is per-session, not per-camera.** `FUSION_THRESHOLD` tuning via the UI slider only affects the currently-loaded video/job; there's no persisted, per-camera sensitivity setting an admin could set once for, say, a congested market-area junction vs. a fast arterial road.

---

## 5. Prioritized Roadmap

### Pre-pilot (must-do before touching a live Coimbatore RTSP feed)
- [ ] Move inference off hardcoded CPU; benchmark real sustained FPS per camera on target hardware; size compute budget for the planned number of concurrent feeds.
- [ ] Add authentication/authorization to all `app.py` mutating endpoints (delete, train, log-feature, start-stream at minimum).
- [ ] Clean/rebuild `accident_features.csv` and retrain (or explicitly disable) the XGBoost refinement layer until the training data is trustworthy.
- [ ] Implement camera calibration (homography per camera) so proximity/TTC/speed thresholds are physically meaningful and portable across junctions.
- [ ] Rename `accident_model.pt` vs `accident_model.pth` to remove the naming collision.
- [ ] Stratified evaluation against CADP + ACCIDENT + at least one Indian CCTV set (UVH-26/IITM-HeTra) rather than dashcam-only benchmarks; explicitly measure day/night/rain performance.
- [ ] Fine-tune/validate YOLO detection on Indian vehicle mix (autorickshaws, triple-riding two-wheelers) — this is the single highest-leverage fix given two-wheelers' outsized share of fatalities.
- [ ] Decide and document a data retention/access policy for incident snapshots and clips.

### During pilot (supervised, alert-suggestion mode — not autonomous dispatch)
- [ ] Build the operator acknowledge/dispatch/resolve workflow with reason codes; feed dismissal reasons back into a labeled dataset.
- [ ] Track false-positive rate per camera per shift against an explicit target (see §7) before considering unsupervised alerting.
- [ ] Add multi-camera grid view + camera health/offline indicators.
- [ ] Make cooldown spatial/incident-scoped rather than whole-camera.
- [ ] Add a process supervisor (systemd/Restart=always or equivalent) and alerting if a camera process dies.
- [ ] Load-test concurrent incident bursts against the thread-spawning paths (Firebase upload, clip extraction, LLM analysis).

### Post-pilot hardening
- [ ] Fine-tune the CNN-BiLSTM specifically on accumulated Coimbatore footage (via the pilot's labeled acknowledge/dismiss data).
- [ ] Bounded thread pools / task queues to replace raw `threading.Thread` spawning throughout.
- [ ] Structured logging across all modules (not just `stream_processor.py`/`health_monitor.py`).
- [ ] Per-camera persisted sensitivity configuration in the UI, editable by an admin role.
- [ ] Formal edge-case regression suite (occlusion-heavy junctions, night/rain clips, high two-wheeler density) run on every threshold/model change.

---

## 6. Suggested Go/No-Go Criteria for Autonomous Alerting

Based on general video-analytics operational benchmarks (industry guidance puts a well-tuned system's false-positive rate below ~5% of raised alerts), a reasonable pilot exit bar before letting UYIR alert without a human in the loop:

- **False positive rate:** < 10% of "confirmed" incidents dismissed by operators as non-accidents, measured across ≥ 2 weeks and ≥ 3 distinct camera locations (target < 5% before full rollout).
- **Recall on a held-out labeled set** of real/staged Coimbatore accidents or near-misses: system should not be silently missing a majority-share event type (e.g. two-wheeler-involved collisions must be caught, given their fatality share).
- **Sustained real-time FPS** on target hardware ≥ the rate needed so that consecutive-frame confirmation + clip extraction don't lag more than a few seconds behind wall-clock time.
- **Zero unauthenticated write access** to incident records or the training pipeline.
- **Documented camera calibration** for every deployed junction, not just the test camera.

---

## 7. References (research pulled for this review)

- *Computer Vision-based Accident Detection in Traffic Surveillance*, arXiv:1911.10037 — closest published analogue to Phase B's trajectory-overlap-then-anomaly design.
- *ACCIDENT: A Benchmark Dataset for Vehicle Accident Detection from Traffic Surveillance Videos* (2026) — fixed-CCTV-viewpoint benchmark; better fit than dashcam sets.
- Revaud et al., *Robust Automatic Monocular Vehicle Speed Estimation for Traffic Surveillance*, ICCV 2021 — camera calibration's outsized effect on speed/TTC accuracy.
- Dashcam-benchmark literature (DAD, DoTA, CCD, A3D, DADA-2000 survey papers) — confirms these are ego-motion/anticipation datasets, a geometry mismatch with fixed-junction CCTV.
- *DriveIndia: An Object Detection Dataset for Diverse Indian Traffic Scenes* (arXiv:2507.19912) and *UVH-26* (arXiv:2511.02563) — Indian-traffic domain gap and available fine-tuning data.
- Two-wheeler violation/fatality statistics from Indian road-safety literature (PMC12321817) — two-wheelers ≈ 44.5% of Indian road fatalities, motivating class-coverage fixes in detection.
- Video-analytics false-alarm/operator-fatigue industry guidance (SafetyScope, Scylla AI) — informs the go/no-go false-positive-rate bar in §7.

---

*This report reflects a full read of every module currently in the project (detection, tracking, phases A–C, fusion, model, app, stream pipeline, UI, and configuration) cross-referenced against the technical description and README already in the repository, plus current literature on CCTV accident detection, camera calibration, benchmark datasets, and Indian traffic conditions.*
