# Spatial CAD Explorer

A standalone, hands-first exhibition for exploring a 3D assembly. It runs from this directory on its own loopback server. STEP files are read locally by OpenCascade in a browser worker.

## Run the exhibit

With Node.js 22.12+ and a WebGL 2 browser, run these commands from this directory.

```sh
npm ci
npm run dev
```

Open [the exhibit](http://localhost:4180/). It starts with Team Arion’s Ultraviolet car. Drag to orbit, scroll or pinch on a touchscreen to zoom, and click or tap a part to inspect it. With the model focused, use the arrow keys to orbit, **+ / −** to zoom, **Enter** to select the center part, or **Home** to fit the view.

Open [Operator settings](http://localhost:4180/operator) and choose **Import STEP** to load a `.step` or `.stp` export. The operator page puts the assembly tree, view presets, colors, undo, and **Reset exhibit** alongside a preview. Cancel an import to keep the current assembly. Choose **Open exhibit** to take the current model and edits into the visitor view in the same tab.

Select a part or assembly to edit its **Display name**. To edit several at once, choose **Export names CSV** in the assembly panel, fill in the **Display Name** column, and save the file as CSV. Use **Import names CSV** with the same model loaded. Blank display names restore the original STEP names. The browser remembers display names for each model.

The startup mesh is generated from `src/models/ultraviolet-full-car.step` and preserves its assembly and face colors. Its default display names come from `src/models/ultraviolet-part-display-names.csv`. Run `npm run model:generate` after editing the STEP. The production build checks that the generated mesh and display names match the assembly.

## Set up the hand camera

Install the camera service in a Python environment that supports the pinned RealSense and MediaPipe packages. Run these commands from the project directory after activating that environment.

```sh
python -m pip install -r camera/requirements.txt
python camera/download_rgb_models.py
python -B camera/server.py --model .runtime/camera/hand_landmarker.task
```

This command uses the RealSense D435i by default.

### Orbbec Astra Pro on Linux

The original Astra Pro exposes its RGB camera (USB `2bc5:0501`) through V4L2 and its depth camera (`2bc5:0403`) through OpenNI2. The bridge pairs the devices through the Astra Pro's internal USB hub and requests 640×480 at 30 fps.

Extract the Linux [Orbbec OpenNI2 2.3.0.86 SDK](https://www.orbbec.com/developers/openni-sdk/) for your CPU architecture. Pass the extracted `sdk/libs` directory, including its `OpenNI2/Drivers` subdirectory, to the bridge. If the SDK is under `.runtime/camera/openni2/sdk`, the flag is optional.

```sh
python -B camera/server.py --camera astra --model .runtime/camera/hand_landmarker.task \
  --openni2-dir /path/to/OpenNI/sdk/libs
```

The bridge runs as your regular user. If Linux denies access to depth, grant that user read/write access to the `2bc5:0403` USB node or configure a udev rule for that ID. Operator settings reports the device path when access is denied. RGB tracking works in fuzzy navigation while depth is unavailable. The Astra Pro uses its own saved camera and calibration settings. Use `--astra-usb-port` to select a hub if multiple Astra Pros are connected.

### Connect and calibrate

Open [Operator settings](http://localhost:4180/operator) and use **Hand camera → Connect**. It starts in **Fuzzy · no calibration** mode with a live mirrored preview on the operator page. The visitor preview is off by default. Enable **Show hand camera preview in the exhibit** to show it for visitors. The browser remembers that choice. Choose **Open exhibit** and bring one relaxed open hand into view. Move it across the central camera area to aim, pinch to move a part, or make a fist to orbit. With one fist held, pull your hand toward yourself to zoom in and push it toward the camera to zoom out. With two fists, move them together to pan or change the distance between them to zoom. Add a second pinch to resize and rotate a part. The cursor is mirrored and smoothed, with a small dead zone to reduce jitter. Depth adapts while your hands are open and stays fixed during a grip.

A short tracking interruption pauses the held part or view. Returning the same grip resumes from that position. If you move out of frame, come back and hold a pinch over a part to grab it again. When one hand drops out during a two-hand grip, the hand still in view can keep moving the part.

In fuzzy navigation, a visible hand can still orbit the view with a fist or drag a part across the screen with a pinch when palm depth drops out. Two fists can still pan and zoom from their positions on screen. Push/pull zoom and depth movement pause until depth returns. An established part grip waits for two consistent depth samples before resuming depth movement from its current position. Calibrated navigation can also keep an established pinch moving across the screen using the last measured reach and the live RGB image.

Hold a thumbs-up with either hand for 1.5 seconds to restore the assembly and its starting view. Curl your fingers and extend your thumb above your knuckles. A compact countdown appears during the hold. Lower the gesture before resetting again.

To set the push/pull range, choose **Hand camera → Set depth range** in Operator settings. At your closest comfortable reach, show one hand with your thumb extended and fingers open. When the thumb reading is steady, curl your fingers into a thumbs-up and hold it without moving the tip of your thumb. Open your hand and repeat at the farthest reach, at least 20 cm farther away. The second confirmation saves the range for this camera. Choose **Use automatic range** to restore the default. This sets where depth movement stops; it does not change what distances the sensor can measure.

For a fitted reach area, use **Optional reach calibration**. Put the browser on the final exhibit display first, ideally full-screen. Select **Capture** at each target. Use the three-second countdown to move your whole hand toward the ring, then hold your fingertip steady with your palm facing the camera. The mirrored preview marks your fingertip in green when depth is available and amber when only the hand is detected. The corners define the reach area. Once they pass, the center helps refine the resting depth. Check that both hands can reach the car without stretching before opening the exhibit to visitors.

Optional calibration is saved per camera and display. If the camera or display moves, choose fuzzy navigation or recalibrate on the operator page. The browser remembers the navigation mode and whether the camera is enabled, and offers direct touch, mouse, and keyboard controls while camera input is unavailable. The local service sends hand positions and gestures to the browser. The video preview is also available during calibration. Frames stay in memory.

### Camera settings

**Operator settings → Hand camera → Camera settings** shows the active camera and configured capture rate. The RealSense D435i offers 30 or 60 fps. Select **Apply and restart** to change it. The Astra Pro uses the 30 fps capture mode. Both use RGB and MediaPipe Full for hand tracking. Capture settings persist on the local camera service.

**Operator settings → Hand camera → Diagnostics** opens [a separate page](http://localhost:4180/diagnostics) with live MediaPipe landmarks on RGB and per-hand grip and depth readings. The Astra Pro supplies one infrared preview and the D435i supplies two. Infrared previews start on demand when Diagnostics is visible. Camera diagnostics stay local and do not record frames.

Start at 30 fps. The capture setting is the requested sensor rate, **Pose rate** counts tracker output, and **Browser rate** counts delivered updates. When comparing 60 fps, check pose age and skipped frames as well as how the controls feel. Move a part or the view in the exhibit, then use the browser’s Back button to return to the operator page and see render-submission timings. Those timings end when the browser submits the draw, before it reaches the display.

## Head-coupled perspective experiment

The experiment is disabled. The webcam tracker, projection, and operator panel remain in the source for future development.

## Simulate the hands

On the development server, open [the simulator](http://localhost:4180/?simulate=1) to enable these controls. The production build presents the exhibit without the simulator.

- Aim at a component with the pointer or **W A S D**. Click and hold or hold **Space** to pinch. Move sideways or use **Q / E** to move the grip toward or away from you.
- While holding the first pinch, hold **Enter** to bring in and pinch with the second hand. Use the **arrow keys** to move it, or **[ / ]** for depth. Spread or turn the two grips to scale or rotate. Release one grip to keep moving with the other.
- Hold **V** to orbit with the left hand and use **Q / E** to zoom. Hold **B** to orbit with the right hand and use **[ / ]** to zoom. Hold **V** and **B** together, then move the hands together to pan or change their spacing to zoom. **Shift-drag** also orbits with the pointer.
- On a touchscreen, start a drag on a component to move it, or start on empty space to orbit. Add a second finger to rotate and resize the part or pan and zoom the view.
- **Esc** simulates tracking loss. The part stays at its last position and requires a fresh grip. Open Setup to pause the simulator while preparing the exhibit.

`npm test` checks STEP import, the startup mesh, and pose handling. `python -B -m unittest discover -s camera` checks the camera calibration and tracking logic. `npm run build` produces a static `dist/` bundle with the startup mesh and on-demand STEP importer. `npm run preview` serves that bundle on the same port when the dev server is stopped.

The importer preserves STEP face colors when it can match them to the geometry, with a part color as the fallback.
