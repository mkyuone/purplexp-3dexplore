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
python camera/download_rgb_models.py
python -B camera/server.py --model .runtime/camera/hand_landmarker.task
```

Open the [operator view](http://localhost:4180/?operator=1), then use **Setup → Hand camera → Connect**. It starts in **Fuzzy · no calibration** mode with a live mirrored camera preview. Use **Show live camera preview** in Setup to hide or show it while navigating. Close Setup and bring one relaxed open hand into view. Move it across the central camera area to aim, pinch to move a part, or make a fist to orbit. With one fist held, pull your hand toward yourself to zoom in and push it toward the camera to zoom out. With two fists, move them together to pan, tilt the pair to orbit left or right, or change the distance between them to zoom. Add a second pinch to resize and rotate a part. The cursor is mirrored and smoothed, with a small dead zone to reduce jitter. Depth adapts while your hands are open and stays fixed during a grip.

A short tracking interruption pauses the held part or view. Returning the same grip resumes from that position. If you move out of frame, come back and hold a pinch over a part to grab it again. When one hand drops out during a two-hand grip, the hand still in view can keep moving the part.

In fuzzy navigation, a visible hand can still orbit the view with a fist or drag a part across the screen with a pinch when palm depth drops out. Two fists can still pan, orbit, and zoom from their positions on screen. Push/pull zoom and depth movement pause until depth returns, then resume from the current position. Calibrated navigation uses measured depth to map your hand to the display.

Hold a thumbs-up with either hand for 1.5 seconds to restore the assembly and its starting view. Curl your fingers and extend your thumb above your knuckles. A compact countdown appears during the hold. Lower the gesture before resetting again.

For a fitted reach area, use **Optional reach calibration**. Put the browser on the final exhibit display first, ideally full-screen. Select **Capture** at each target. Use the three-second countdown to move your whole hand toward the ring, then hold your fingertip steady with your palm facing the camera. The mirrored preview marks your fingertip in green when depth is available and amber when only the hand is detected. The corners define the reach area. Once they pass, the center helps refine the resting depth. Check that both hands can reach the car without stretching before opening the exhibit to visitors.

Optional calibration is saved per camera and display. If the camera or display moves, choose fuzzy navigation or recalibrate in the operator view. Open the normal exhibit URL for visitors. The browser remembers the navigation mode and whether the camera is enabled, and offers direct touch, mouse, and keyboard controls while camera input is unavailable. The local service sends hand positions and gestures to the browser. The video preview is available in the operator view and during calibration. Frames stay in memory.

### Camera settings

**Operator setup → Hand camera → Camera settings** offers 30 or 60 fps capture. Select **Apply and restart** to activate a choice. The camera uses RGB with the depth projector enabled and MediaPipe Full for hand tracking. Capture settings persist on the local camera service.

Start at 30 fps. The capture setting is the sensor rate. **Pose rate** shows how many updates the tracker actually processes. When comparing 60 fps, check whether two-hand gestures feel steadier and whether **Frame age** goes down.

## Try head-coupled perspective

In [Operator Setup](http://localhost:4180/?operator=1), open **Setup → Head-coupled perspective → Start** and allow camera access. Use the integrated RGB webcam or another webcam separate from the D435i. Face it until the view centers, then close Setup and move your head sideways or up and down. **Recenter** sets a new resting position. The preview and controls are in Setup, and the experiment starts off on reload.

The webcam tracks one viewer locally at a fixed virtual viewing distance. The RealSense hand controls work alongside it. While you pinch and hold a part, the head perspective stays steady and catches up after release. The webcam pauses when you switch away from the tab and centers again when you return.

The head diagnostics show webcam and pose rates, inference time, and the interval from a browser video frame callback to render submission. Move your head to collect render timings. For physical latency, film your head movement and the display in the same high-frame-rate recording.

## Simulate the hands

On the development server, open [the simulator](http://localhost:4180/?simulate=1) to enable these controls. The production build presents the exhibit without the simulator.

- Aim at a component with the pointer or **W A S D**. Click and hold or hold **Space** to pinch. Move sideways or use **Q / E** to move the grip toward or away from you.
- While holding the first pinch, hold **Enter** to bring in and pinch with the second hand. Use the **arrow keys** to move it, or **[ / ]** for depth. Spread or turn the two grips to scale or rotate. Release one grip to keep moving with the other.
- Hold **V** to orbit with the left hand and use **Q / E** to zoom. Hold **B** to orbit with the right hand and use **[ / ]** to zoom. Hold **V** and **B** together, then move the hands together to pan, tilt the pair to orbit, or change their spacing to zoom. **Shift-drag** also orbits with the pointer.
- On a touchscreen, start a drag on a component to move it, or start on empty space to orbit. Add a second finger to rotate and resize the part or pan, orbit, and zoom the view.
- **Esc** simulates tracking loss. The part stays at its last position and requires a fresh grip. Open Setup to pause the simulator while preparing the exhibit.

`npm test` checks STEP import, the startup mesh, and pose handling. `python -B -m unittest discover -s camera` checks the camera calibration and tracking logic. `npm run build` produces a static `dist/` bundle with the startup mesh and on-demand STEP importer. `npm run preview` serves that bundle on the same port when the dev server is stopped.

The importer preserves STEP face colors when it can match them to the geometry, with a part color as the fallback.
