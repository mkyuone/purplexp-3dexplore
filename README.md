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

Open the [operator view](http://localhost:4180/?operator=1), then use **Setup → Hand camera → Connect → Calibrate reach**. Put the browser on the final exhibit display first, ideally full-screen. Aim one index finger at each of the four corner rings and the center ring at a comfortable distance, hold it steady, and select **Capture**. Check that both hands can reach the car and the join ring without stretching before opening the exhibit to visitors.

Calibration is saved per camera and display. If the camera or display moves, recalibrate in the operator view. Open the normal exhibit URL for visitors. The browser remembers whether the camera is enabled and offers direct touch, mouse, and keyboard controls while camera input is unavailable. The local service sends hand positions and gestures to the browser.

## Simulate the hands

On the development server, open [the simulator](http://localhost:4180/?simulate=1) to enable these controls. The production build presents the exhibit without the simulator.

- Aim at a component with the pointer or **W A S D**. Click and hold or hold **Space** to pinch. Move sideways or use **Q / E** to move the grip toward or away from you.
- While holding the first pinch, hold **Enter** to bring in and pinch with the second hand at the ring. Use the **arrow keys** to move it, or **[ / ]** for depth. Spread or turn the two grips to scale or rotate. Release one grip to keep moving with the other.
- Hold **V** and move to orbit from anywhere, or **Shift-drag** with the pointer. Hold **B** to bring in the second closed hand at the ring, then use the **arrow keys** to change hand separation and zoom. View movement leaves the parts in place.
- On a touchscreen, start a drag on a component to move it, or start on empty space to orbit. Touch the second finger at the join ring to rotate and resize the part or zoom the view.
- **Esc** simulates tracking loss. The part stays at its last position and requires a fresh grip. Open Setup to pause the simulator while preparing the exhibit.

`npm test` checks STEP import, the startup mesh, and pose handling. `python -B -m unittest discover -s camera` checks the camera calibration and tracking logic. `npm run build` produces a static `dist/` bundle with the startup mesh and on-demand STEP importer. `npm run preview` serves that bundle on the same port when the dev server is stopped.

The importer preserves STEP face colors when it can match them to the geometry, with a part color as the fallback.
