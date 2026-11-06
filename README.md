# Spatial CAD Explorer

A standalone, hands-first exhibition for exploring a 3D assembly. It runs from this directory on its own loopback server. STEP files are read locally by OpenCascade in a browser worker.

## Run the exhibit

With Node.js 22.12+ and a WebGL 2 browser, run these commands from this directory.

```sh
npm ci
npm run dev
```

Open [the exhibit](http://localhost:4180/). It starts with Team Arion’s Ultraviolet car from `src/models/ultraviolet-full-car.step`. The STEP includes the car and helmet colors from the engineering renderings.

Open **Setup → Import STEP** to load a `.step` or `.stp` export. Setup also contains the assembly tree, view presets, colors, undo, and **Reset exhibit**. Operator edits and files opened through the picker live in the current browser session.

## Simulate the hands

On the development server, open [the simulator](http://localhost:4180/?simulate=1) to enable these controls. The production build presents the exhibit without the simulator.

- Aim at a component with the pointer or **W A S D**. Click and hold or hold **Space** to pinch. Move sideways or use **Q / E** to move the grip toward or away from you.
- While holding the first pinch, hold **Enter** to bring in and pinch with the second hand at the ring. Use the **arrow keys** to move it, or **[ / ]** for depth. Spread or turn the two grips to scale or rotate. Release one grip to keep moving with the other.
- Hold **V** and move to orbit from anywhere, or **Shift-drag** with the pointer. Hold **B** to bring in the second closed hand at the ring, then use the **arrow keys** to change hand separation and zoom. View movement leaves the parts in place.
- On a touchscreen, start a drag on a component to move it, or start on empty space to orbit. Touch the second finger at the join ring to rotate and resize the part or zoom the view.
- **Esc** simulates tracking loss. The part stays at its last position and requires a fresh grip. Open Setup to pause the simulator while preparing the exhibit.

`npm test` checks STEP import and pose handling. `npm run build` produces a static `dist/` bundle with the car STEP, importer, and WebAssembly runtime. `npm run preview` serves that bundle on the same port when the dev server is stopped.

The current input source simulates both hands and their depth. The pose contract in `src/gestures.ts` is the integration point for a calibrated depth camera. The importer preserves STEP face colors when it can match them to the geometry, with a part color as the fallback.
