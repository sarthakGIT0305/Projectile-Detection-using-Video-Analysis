# Projectile Live C++

This directory is an independent C++17/OpenCV live-camera implementation. The Python tree remains untouched and is the behavioral reference.

## Prerequisites

- CMake 3.16 or newer
- A C++17 compiler
- OpenCV 4.x development files with `core`, `imgproc`, `highgui`, `video`, and `videoio`

## Build and test

From `cpp_live`:

```text
cmake -S . -B build
cmake --build build --config Release
ctest --test-dir build --output-on-failure -C Release
```

## Run

```text
build\Release\projectile_live.exe config\pipeline_default.ini
```

From the repository root, the simplest Windows startup method is to double-click `run_projectile_live.bat`. It sets the MinGW64 runtime path, finds the executable, and starts the configured source. Edit `config/pipeline_default.ini` before running to choose a stored `video_path` or set `video_path=` and select a `camera_index`.

The default configuration runs the prerecorded sample at `assets/50m-1.mp4`. Run the command from the repository root because `video_path` is relative to the current directory. Change `video_path` to `assets/50m-2.mp4`, `assets/50m-3.mp4`, or `assets/50m-4.mp4` to select another sample. To use a camera instead, set `video_path=` and choose the `camera_index`.

Set `camera_index`, `width`, `height`, and `fps` in the INI file. Set `display=false` and `headless=true` for a machine without a GUI. Press `q` or Escape to stop the displayed application.

The application reports captured, processed, dropped, and processing-FPS counters on shutdown. Live camera input uses a bounded latest-frame buffer with capacity one: when processing falls behind, old frames are discarded and the newest frame is retained. Prerecorded video uses the same bounded buffer with back-pressure, so playback waits for processing and does not skip frames.

## Behavior and limitations

The detector preserves the Python reference thresholds and mask stages. The current C++ entry point accepts a camera device; file/RTSP inputs can be added behind the capture abstraction. A camera cannot be verified in this environment, so live FPS and end-to-end latency require measurement on the target machine. Cross-language frame-by-frame parity tooling is not yet automated; deterministic module tests are included.

OpenCV errors, invalid configuration, camera-open failure, empty frames, disconnects, and shutdown are handled without raw-pointer ownership.
