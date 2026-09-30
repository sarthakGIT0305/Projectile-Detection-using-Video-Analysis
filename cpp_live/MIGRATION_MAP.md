# Python to C++ Migration Map

| Python component | C++ component | Responsibility | Strategy / behavioral notes |
| --- | --- | --- | --- |
| `config.py` | `config.hpp/.cpp`, `pipeline_default.ini` | Tunable values and flags | Values are externalized; defaults preserve documented thresholds. |
| `video_reader.py` | `camera_capture.hpp/.cpp` | Frame acquisition | Redesigned as a camera thread; the pipeline is source-independent at the buffer boundary. |
| `preprocessing/roi.py` | `Preprocessor::roi` | Optional crop and offset | Same centered, clamped ROI convention. |
| `preprocessing/clahe_gray.py` | `Preprocessor::gray` | LAB L-channel CLAHE and grayscale | Same OpenCV color conversion sequence. |
| `motion/tophat.py` | `MotionDetector` | White/black top-hat mask | Same elliptical kernel and mode behavior. |
| `motion/frame_diff.py` | `MotionDetector` | Two interval temporal AND | Same gap and threshold; rolling buffer is bounded. |
| `motion/bg_sub.py` | `MotionDetector` | MOG2 background mask | Uses BGR input and thresholds shadows out at 200. |
| `motion/mask_combine.py`, `morph_clean.py` | `MotionDetector` | Fusion and morphology | Active `tophat_and_any` path and 2x2 open/close preserved. |
| `detection/contour_detect.py`, `blob_filter.py` | `BlobDetector` | Contours and geometric filtering | Explicit `Detection` replaces Python dicts; filters and order preserved. |
| `detection/isolation_filter.py` | `BlobDetector` | Cluster rejection | O(n^2) implementation matches the normal Python path; no SciPy dependency. |
| `tracking/kalman_tracker.py` | `KalmanTracker`, `hungarian.cpp` | Prediction, assignment, correction | Constant-velocity 4-state filter and gated Hungarian matching. |
| `tracking/trail_store.py` | `TrailStore` | Bounded observed/predicted history | Fixes the Python list-dispatch syntax bug while preserving intended behavior. |
| `main.py` arc logic | `TrajectoryAnalyzer` | Parabola residual, arc and landing extrapolation | Uses least-squares normal equations and the same positive-a/residual/span tests. |
| `main.py` drawing | `Visualizer` | Boxes, trails, arcs, HUD | GUI is optional through configuration. |

## Intentional redesigns and differences

- Live capture and processing are separate threads with a one-frame latest buffer. The Python implementation is synchronous and file-only.
- The capture layer now supports both prerecorded files (`video_path`) and camera devices; the default configuration selects `assets/50m-1.mp4` so the migrated pipeline can be exercised without camera hardware.
- Frame skipping defaults to one in the C++ configuration because stale-frame latency is handled by dropping at capture; the Python file config documented a value of two.
- Disabled Python trajectory validator and classifier remain unimplemented as active pipeline stages because both are disabled in the reference runtime. The active arc detector is migrated.
- The C++ build has no NumPy/SciPy dependency. OpenCV matrices and a native Hungarian implementation replace those dependencies.
- Cross-language output comparison is not automated yet. Tests currently validate deterministic C++ contracts; recorded-frame parity remains an unresolved validation task.

## Ownership and unresolved issues

`cv::Mat` packets are reference-counted and moved through the bounded buffer. Pipeline state is single-owner on the processing thread. Camera verification and measured FPS/latency require hardware access. File and RTSP sources can be added as alternative capture implementations without changing detector APIs.
