# Spatial CAD Explorer

A standalone, hands-first exhibition for exploring a 3D assembly. It runs from this directory on its own loopback server. STEP files are read locally by OpenCascade in a browser worker.

## Run the exhibit

With Node.js 22.12+ and a WebGL 2 browser, run these commands from this directory.

```sh
npm ci
npm run dev
```

Open [the exhibit](http://localhost:4180/). It starts with Team Arion’s Ultraviolet car. Drag to orbit, scroll or pinch on a touchscreen to zoom, and click or tap a part to inspect it. With the model focused, use the arrow keys to orbit, **+ / −** to zoom, **Enter** to select the center part, or **Home** to fit the view.

Open **Setup → Import STEP** to load a `.step` or `.stp` export. Setup keeps the live preview alongside the assembly tree, view presets, colors, undo, and **Reset exhibit**. Cancel an import to keep the current assembly. Operator edits and files opened through the picker live in the current browser session.

The startup mesh is generated from `src/models/ultraviolet-full-car.step` and preserves its assembly and face colors. Run `npm run model:generate` after editing the STEP. `npm run model:check` verifies the source and generated mesh, and the production build runs that check automatically.

## Set up a RealSense hand camera

Install the camera service in a Python environment that supports the pinned RealSense and MediaPipe packages. Run these commands from the project directory after activating that environment.

```sh
python -m pip install -r camera/requirements.txt
mkdir -p .runtime/camera
curl -fL -o .runtime/camera/hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
python -B camera/server.py --model .runtime/camera/hand_landmarker.task
```

Open the [operator view](http://localhost:4180/?operator=1), then use **Setup → Hand camera → Connect**. It starts in **Fuzzy · no calibration** mode with a live mirrored camera preview. Use **Show live camera preview** in Setup to hide or show it while navigating. Close Setup and bring one relaxed open hand into view. Move it across the central camera area to aim, pinch to move a part, or make a fist to orbit. With that fist held, pull your hand toward yourself to zoom in and push it toward the camera to zoom out. Add a second pinch to resize and rotate a part. The cursor is mirrored and smoothed, with a small dead zone to reduce jitter. Depth adapts while your hands are open and stays fixed during a grip.

A short tracking interruption pauses the held part or view. Returning the same grip resumes from that position. If you move out of frame, come back and hold a pinch over a part to grab it again. When one hand drops out during a two-hand grip, the hand still in view can keep moving the part.

Hold a thumbs-up with either hand for 1.5 seconds to restore the assembly and its starting view. A compact countdown appears during the hold. Lower the gesture before resetting again.

For a fitted reach area, use **Optional reach calibration**. Put the browser on the final exhibit display first, ideally full-screen. Select **Capture** at each target. Use the three-second countdown to move your whole hand toward the ring, then hold your fingertip steady with your palm facing the camera. The mirrored preview marks your fingertip in green when depth is available and amber when only the hand is detected. The corners define the reach area. Once they pass, the center helps refine the resting depth. Check that both hands can reach the car without stretching before opening the exhibit to visitors.

Optional calibration is saved per camera and display. If the camera or display moves, choose fuzzy navigation or recalibrate in the operator view. Open the normal exhibit URL for visitors. The browser remembers the navigation mode and whether the camera is enabled, and offers direct touch, mouse, and keyboard controls while camera input is unavailable. The local service sends hand positions and gestures to the browser. The video preview is available in the operator view and during calibration. Frames stay in memory.

### Compare RGB hand models

**Operator setup → Hand camera → Camera testing** selects a hand model and 30 or 60 fps capture. Select **Apply and restart** to activate a choice. The camera uses RGB with the depth projector enabled. Settings persist on the local camera service. Model changes use the same RGB reach profile and start a new gesture session.

- **MediaPipe Full** uses the current MediaPipe Tasks hand detector and full landmark model.
- **RTMPose** uses the independent RTMDet-nano hand detector and RTMPose-m 21-point landmark network through ONNX Runtime. Its 2D finger geometry feeds the shared gestures, and RealSense supplies physical depth. This model has higher CPU cost on the current laptop. [Official hand model demo](https://github.com/open-mmlab/mmpose/blob/main/demo/docs/en/2d_hand_demo.md)

To install both models in a project-local environment, run the following commands from this directory.

```sh
python3 -m venv .runtime/camera/venv
.runtime/camera/venv/bin/python -m pip install -r camera/requirements-models.txt
.runtime/camera/venv/bin/python camera/download_rgb_models.py
.runtime/camera/venv/bin/python -B camera/server.py --model .runtime/camera/hand_landmarker.task
```

The downloader verifies the model files and archives against pinned SHA-256 values. Model assets stay in ignored `.runtime/camera/` storage. Older IR or MediaPipe Lite settings open with RGB and MediaPipe Full.

Start at 30 fps. The capture setting is the sensor rate; **Pose rate** shows how many updates the tracker actually processes. When comparing 60 fps, check whether two-hand gestures feel steadier and whether **Frame age** goes down.

## Simulate the hands

On the development server, open [the simulator](http://localhost:4180/?simulate=1) to enable these controls. The production build presents the exhibit without the simulator.

- Aim at a component with the pointer or **W A S D**. Click and hold or hold **Space** to pinch. Move sideways or use **Q / E** to move the grip toward or away from you.
- While holding the first pinch, hold **Enter** to bring in and pinch with the second hand. Use the **arrow keys** to move it, or **[ / ]** for depth. Spread or turn the two grips to scale or rotate. Release one grip to keep moving with the other.
- Hold **V** and move to orbit from anywhere. Use **Q / E** while holding the fist to zoom with depth. **B** holds the right fist instead, with the **arrow keys** for orbit and **[ / ]** for depth. **Shift-drag** also orbits with the pointer.
- On a touchscreen, start a drag on a component to move it, or start on empty space to orbit. Add a second finger to rotate and resize the part or zoom the view.
- **Esc** simulates tracking loss. The part stays at its last position and requires a fresh grip. Open Setup to pause the simulator while preparing the exhibit.

`npm test` checks STEP import, the startup mesh, and pose handling. `python -B -m unittest discover -s camera` checks the camera calibration and tracking logic. `npm run build` produces a static `dist/` bundle with the startup mesh and on-demand STEP importer. `npm run preview` serves that bundle on the same port when the dev server is stopped.

The importer preserves STEP face colors when it can match them to the geometry, with a part color as the fallback.
