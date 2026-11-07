import { Box, Crosshair, Expand, Hand, LoaderCircle, Orbit, Settings, TriangleAlert, Upload, X } from 'lucide-react';
import { useEffect, useMemo, useRef, useState, type ChangeEvent, type DragEvent } from 'react';
import { importStep, readStepFile, type CadAssembly } from './cad';
import { loadBundledAssembly } from './bundled-assembly';
import { CadScene } from './scene';
import { GestureController, type GestureState } from './gestures';
import { HandSimulator } from './simulator';
import { AssemblyPanel, indexOccurrences, PartPanel } from './panels';
import { calibrateHands, connectLiveHands, handDisplayKey, type LiveHandStatus } from './live-hands';
import { attachVisitorControls } from './visitor-controls';
import { CalibrationPanel } from './calibration-panel';

const startingModelName = 'Ultraviolet Full Car.step';
const query = new URLSearchParams(window.location.search);
const simulationMode = import.meta.env.DEV && query.get('simulate') === '1';
const operatorMode = !simulationMode && query.get('operator') === '1';
const cameraPreference = 'spatial-cad-explorer.hand-camera';
const emptyRoots: CadAssembly['roots'] = [];
type Runtime = {
  scene: CadScene;
  gestures: GestureController;
  simulator?: HandSimulator;
  setupOpen: boolean;
  calibrating: boolean;
  contextLost: boolean;
  hasModel: boolean;
  nativeUntil: number;
};
type Exhibit = { name: string; roots: CadAssembly['roots']; revision: number };

function visitorCue(state: GestureState | null) {
  if (state?.status === 'lost') return 'Move your hands back into view and start with an open hand.';
  if (state?.status === 'both') return state.target === 'part' ? 'Move together, spread to resize, or turn to rotate.' : 'Spread your hands to zoom in. Bring them together to zoom out.';
  if (state?.status === 'holding') return state.target === 'part' ? 'Move the part, or pinch at the ring with your other hand.' : 'Move to look around. Make a second fist at the ring to zoom.';
  return 'Pinch a part to move it. Make a fist to look around.';
}

export function App() {
  const viewport = useRef<HTMLElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const setupTrigger = useRef<HTMLButtonElement>(null);
  const setupClose = useRef<HTMLButtonElement>(null);
  const picker = useRef<HTMLInputElement>(null);
  const runtime = useRef<Runtime | null>(null);
  const importAbort = useRef<AbortController | null>(null);
  const calibrationDisplay = useRef<string | null>(null);
  const modelRevision = useRef(0);
  const [exhibit, setExhibit] = useState<Exhibit | null>(null);
  const [loading, setLoading] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [rendererError, setRendererError] = useState<string | null>(null);
  const [contextLost, setContextLost] = useState(false);
  const [setupOpen, setSetupOpen] = useState(false);
  const [enabled, setEnabled] = useState(simulationMode);
  const [liveEnabled, setLiveEnabled] = useState(() => {
    try { return !simulationMode && localStorage.getItem(cameraPreference) === 'on'; }
    catch { return false; }
  });
  const [liveStatus, setLiveStatus] = useState<LiveHandStatus>({ phase: 'connecting' });
  const [calibrationProgress, setCalibrationProgress] = useState<LiveHandStatus | null>(null);
  const [calibrationBusy, setCalibrationBusy] = useState(false);
  const [calibrationError, setCalibrationError] = useState<string | null>(null);
  const [hasExplored, setHasExplored] = useState(false);
  const [selected, setSelected] = useState<string | null>(null);
  const [pose, setPose] = useState<GestureState | null>(null);
  const [, setRevision] = useState(0);
  const calibrating = calibrationProgress !== null;
  const roots = exhibit?.roots ?? emptyRoots;
  const index = useMemo(() => indexOccurrences(roots), [roots]);
  const part = selected === null ? undefined : index.byId.get(selected);
  const focusId = pose?.held ?? pose?.hovered ?? selected;
  const focusPart = focusId ? index.byId.get(focusId) : undefined;
  const scene = runtime.current?.scene;
  const details = setupOpen && part ? scene?.describe(part.id) : null;

  async function load(name: string, read: (signal: AbortSignal) => Promise<CadAssembly>) {
    importAbort.current?.abort();
    const controller = new AbortController();
    importAbort.current = controller;
    setLoading(name);
    setError(null);
    try {
      const assembly = await read(controller.signal);
      const active = runtime.current;
      if (controller.signal.aborted || !active) return;
      active.gestures.cancel();
      active.scene.load(assembly);
      active.hasModel = true;
      setExhibit({ name, roots: assembly.roots, revision: ++modelRevision.current });
      setSelected(null);
    } catch (cause) {
      if (!controller.signal.aborted) {
        setError(`${name} · ${cause instanceof Error ? cause.message : 'The file could not be imported.'}`);
      }
    } finally {
      if (importAbort.current === controller) {
        importAbort.current = null;
        setLoading(null);
      }
    }
  }

  function loadStartingModel() {
    return load(startingModelName, loadBundledAssembly);
  }

  useEffect(() => {
    const stage = viewport.current;
    const surface = canvas.current;
    if (!stage || !surface) return;
    let scene: CadScene;
    try {
      scene = new CadScene(surface, (status) => {
        const active = runtime.current;
        if (active) {
          active.contextLost = status === 'lost';
          active.gestures.resetStream();
        }
        setContextLost(status === 'lost');
      });
    } catch {
      setRendererError('A WebGL 2 context could not be created. Open the explorer in a browser with hardware acceleration enabled.');
      return;
    }
    const gestures = new GestureController(scene, state => {
      runtime.current?.simulator?.setJoinTarget(state.join);
      setSelected(scene.getSelected());
      setPose(state);
      if (state.target) setHasExplored(true);
    });
    const active: Runtime = { scene, gestures, setupOpen: false, calibrating: false, contextLost: false, hasModel: false, nativeUntil: 0 };
    runtime.current = active;
    const sizeObserver = new ResizeObserver(([entry]) => {
      const { width, height } = entry.contentRect;
      gestures.setViewport(width, height);
    });
    sizeObserver.observe(surface);
    gestures.setViewport(surface.clientWidth, surface.clientHeight);
    let detachControls: (() => void) | undefined;
    if (simulationMode) {
      active.simulator = new HandSimulator(stage, frame => {
        if (!active.setupOpen && !active.contextLost && active.hasModel) gestures.receive(frame);
      }, (x, y) => scene.pick(x, y) ? 'part' : 'view');
      active.simulator.setEnabled(true);
    } else {
      detachControls = attachVisitorControls(surface, scene, {
        available: () => active.hasModel && !active.contextLost && !active.calibrating,
        pauseHands: (pointerHeld) => {
          const now = performance.now();
          if (now >= active.nativeUntil) gestures.resetStream();
          active.nativeUntil = pointerHeld ? Infinity : now + 450;
        },
        select: (id) => { scene.select(id); setSelected(id); },
        used: () => setHasExplored(true),
      });
    }
    void loadStartingModel();
    return () => {
      importAbort.current?.abort();
      importAbort.current = null;
      sizeObserver.disconnect();
      detachControls?.();
      active.simulator?.dispose();
      gestures.dispose();
      scene.dispose();
      runtime.current = null;
    };
  }, []);

  useEffect(() => {
    const active = runtime.current;
    if (simulationMode || !liveEnabled || !active) return;
    setLiveStatus({ phase: 'connecting' });
    const disconnect = connectLiveHands({
      receive: frame => {
        if (!active.setupOpen && !active.calibrating && !active.contextLost && active.hasModel && performance.now() >= active.nativeUntil) active.gestures.receive(frame);
      },
      reset: () => active.gestures.resetStream(),
      status: setLiveStatus,
    });
    return () => {
      disconnect();
      if (runtime.current === active) active.gestures.cancel();
    };
  }, [liveEnabled]);

  function receiveFile(file?: File) {
    if (!file) return;
    if (!/\.st(?:e)?p$/i.test(file.name)) {
      setError('Choose a STEP assembly or part with a .step or .stp extension.');
      return;
    }
    void load(file.name, async signal => importStep(await readStepFile(file, signal), signal));
  }

  function onFile(event: ChangeEvent<HTMLInputElement>) {
    receiveFile(event.currentTarget.files?.[0]);
    event.currentTarget.value = '';
  }

  function onDrop(event: DragEvent<HTMLElement>) {
    if (!event.dataTransfer.types.includes('Files')) return;
    event.preventDefault();
    if (!rendererError && !contextLost) receiveFile(event.dataTransfer.files[0]);
  }

  function edit(change: (scene: CadScene) => void) {
    const active = runtime.current;
    if (!active || active.contextLost) return;
    active.gestures.cancel();
    change(active.scene);
    setRevision(value => value + 1);
  }

  function openSetup() {
    const active = runtime.current;
    if (active) {
      active.setupOpen = true;
      active.simulator?.setEnabled(false);
      active.gestures.resetStream();
    }
    setSetupOpen(true);
  }

  function closeSetup() {
    setSetupOpen(false);
    const active = runtime.current;
    if (active) {
      active.setupOpen = false;
      active.gestures.resetStream();
      active.simulator?.setEnabled(enabled);
    }
    requestAnimationFrame(() => (simulationMode ? viewport.current : setupTrigger.current)?.focus({ preventScroll: true }));
  }

  useEffect(() => {
    if (setupOpen) setupClose.current?.focus({ preventScroll: true });
  }, [setupOpen]);

  function toggleLive() {
    const next = !liveEnabled;
    try { localStorage.setItem(cameraPreference, next ? 'on' : 'off'); }
    catch { /* The camera can still be used for this session. */ }
    setLiveEnabled(next);
    setCalibrationError(null);
  }

  function finishCalibration() {
    calibrationDisplay.current = null;
    const active = runtime.current;
    if (active) {
      active.calibrating = false;
      active.gestures.resetStream();
    }
    setCalibrationProgress(null);
    requestAnimationFrame(() => setupTrigger.current?.focus({ preventScroll: true }));
  }

  async function changeCalibration(action: 'begin' | 'capture' | 'cancel') {
    if (!operatorMode || calibrationBusy) return;
    const display = action === 'begin' ? handDisplayKey() : calibrationDisplay.current;
    if (!display) return;
    setCalibrationBusy(true);
    setCalibrationError(null);
    try {
      const next = await calibrateHands(action, display);
      if (display === handDisplayKey()) setLiveStatus(next);
      if (action === 'begin') {
        if (next.phase !== 'calibration') throw new Error(next.message ?? 'The hand camera is unavailable.');
        calibrationDisplay.current = display;
        const active = runtime.current;
        if (active) active.calibrating = true;
        setCalibrationProgress(next);
        closeSetup();
      } else if (action === 'cancel' || next.phase === 'ready') {
        finishCalibration();
      } else if (next.phase === 'calibration') {
        setCalibrationProgress(next);
      } else {
        setCalibrationError(next.message ?? 'The hand camera is unavailable.');
      }
    } catch (cause) {
      if (action === 'cancel') finishCalibration();
      else setCalibrationError(cause instanceof Error ? cause.message : 'The camera did not respond.');
    } finally {
      setCalibrationBusy(false);
    }
  }

  const graphicsIssue = rendererError ?? (contextLost ? 'The graphics connection was interrupted. Waiting for it to recover.' : null);
  const stageMessage = graphicsIssue ? <div className="message error" role="alert"><TriangleAlert size={18} /><span>{graphicsIssue}</span>
    <button className="button" type="button" onClick={() => window.location.reload()}>Reload</button>
  </div> : loading ? <div className="message loading" role="status"><LoaderCircle size={18} /> <span>Loading {loading}</span>
    <button className="button" type="button" onClick={() => importAbort.current?.abort()}>Cancel</button>
  </div> : error ? <div className="message error" role="alert"><TriangleAlert size={18} /> <span>{error}</span>
    <button type="button" className="icon-button close-message" aria-label="Dismiss import error" onClick={() => setError(null)}><X size={16} /></button>
  </div> : null;
  const handsEnabled = simulationMode ? enabled : liveEnabled && liveStatus.phase === 'ready';
  const showPose = handsEnabled && !setupOpen && !calibrating && !contextLost && !!exhibit;
  const visibleHint = !setupOpen && !calibrating && exhibit && !contextLost && !hasExplored && (!simulationMode || enabled);
  const hint = handsEnabled ? visitorCue(pose) : 'Drag to look around. Tap a part to inspect it.';

  return <main className={`explorer${setupOpen ? ' is-setup-open' : ''}`} onKeyDown={event => {
    if (event.key !== 'Escape' || event.defaultPrevented) return;
    if (calibrating) void changeCalibration('cancel');
    else if (setupOpen) closeSetup();
  }}>
    <h1 className="sr-only">Spatial CAD Explorer</h1>
    <section ref={viewport} className="exhibit" tabIndex={simulationMode ? 0 : undefined} aria-label="Interactive CAD exhibit" aria-describedby={simulationMode ? 'visitor-cue' : undefined}>
      <div className="stage-light" />
      <canvas ref={canvas} className="model" tabIndex={simulationMode ? undefined : 0} aria-label="3D CAD assembly" aria-describedby={simulationMode ? undefined : 'canvas-help'} />
      {!simulationMode && <p id="canvas-help" className="sr-only">Use the arrow keys to orbit, plus and minus to zoom, Enter to select the part in the center, Escape to clear selection, or Home to reset the view.</p>}
      {!setupOpen && !calibrating && <button ref={setupTrigger} className="setup-trigger" type="button" onClick={openSetup}><Settings size={22} /> Setup</button>}
      <p id="visitor-cue" className={visibleHint ? `visitor-hint${simulationMode ? ' simulator-hint' : ''}` : 'sr-only'} aria-live="polite">{hint}</p>
      {!setupOpen && !calibrating && (stageMessage || !exhibit) && <div className="stage-message">
        {stageMessage}
        {!exhibit && !loading && !graphicsIssue && <div className="empty-stage">
          <Box size={35} /><p>Open Setup to load an assembly.</p>
          <button type="button" className="button" onClick={() => void loadStartingModel()}>Load Ultraviolet</button>
        </div>}
      </div>}
      {showPose && <div className="hands-layer" aria-hidden="true">
        {pose?.hands.map(hand => <div key={hand.id}
          className={`hand-cursor${hand.id === 'right' ? ' second' : ''}${hand.pinched ? ' is-pinched' : ''}${hand.fist ? ' is-view-grip' : ''}`}
          style={{ left: `${hand.x * 100}%`, top: `${hand.y * 100}%`, transform: `translate(-50%, -50%) scale(${Math.max(0.65, Math.min(1.45, 1 + hand.depth))})` }}>
          {hand.fist ? <Orbit size={25} /> : <Hand size={25} />}
        </div>)}
        {pose?.join && <div className="join-target" style={{ left: `${pose.join.x * 100}%`, top: `${pose.join.y * 100}%` }}>
          <span className="join-ring" style={{ width: 2 * pose.join.radiusPx, height: 2 * pose.join.radiusPx }}>{pose.target === 'view' ? <Orbit size={24} /> : <Crosshair size={24} />}</span>
          <span className="join-label">{pose.target === 'view' ? 'Make a fist here' : 'Pinch here'}</span>
        </div>}
      </div>}
      {!setupOpen && !calibrating && (simulationMode || focusPart || pose?.target === 'view') && <footer className="exhibit-footer">
        {pose?.target === 'view'
          ? <div className="part-caption"><Orbit size={16} /><span>Exhibit view</span></div>
          : focusPart && <div className="part-caption" aria-live="polite"><Box size={16} /><span>{focusPart.name}</span></div>}
        {simulationMode && <span className="input-status" data-active={enabled && pose?.status !== 'lost' && !rendererError}><i />{enabled ? 'Simulated hands' : 'Simulation paused'}</span>}
      </footer>}
      {calibrationProgress && <CalibrationPanel step={calibrationProgress.step ?? 0} message={calibrationError ?? calibrationProgress.message ?? null} busy={calibrationBusy}
        onCapture={() => void changeCalibration('capture')} onCancel={() => void changeCalibration('cancel')} />}
    </section>

    {setupOpen && <aside id="exhibit-setup" className="setup-panel" aria-labelledby="setup-title">
      <div className="setup-head">
        <div><h2 id="setup-title">Exhibit setup</h2><p>Choose an assembly and prepare the exhibit.</p></div>
        <button ref={setupClose} className="button icon-button" type="button" aria-label="Close setup" onClick={closeSetup}><X size={20} /></button>
      </div>
      <div className="setup-scroll" onDragOver={event => { if (event.dataTransfer.types.includes('Files')) event.preventDefault(); }} onDrop={onDrop}>
        <section className="panel file-panel" aria-label="STEP import">
          <h2>Model</h2>
          <p className="file-name" title={exhibit?.name}>{exhibit?.name ?? 'No model loaded'} <span>mm · Z up</span></p>
          <div className="file-actions">
            <button className="button button-primary" type="button" onClick={() => picker.current?.click()} disabled={!!graphicsIssue}><Upload size={16} /> Import STEP</button>
            <button className="button" type="button" onClick={() => void loadStartingModel()} disabled={!!graphicsIssue || !!loading}>Load starting model</button>
          </div>
          <input ref={picker} type="file" aria-label="STEP file" accept=".step,.stp" onChange={onFile} hidden />
          {stageMessage}
        </section>
        <AssemblyPanel key={exhibit?.revision ?? 0} roots={roots} index={index} selected={selected} onSelect={id => edit(scene => {
          scene.select(id);
          setSelected(id);
        })} />
        <PartPanel part={part} details={details} canUndo={scene?.canUndo() ?? false} onColor={color => {
          if (part) edit(scene => scene.color(part.id, color));
        }} onRestore={() => { if (part) edit(scene => scene.reset(part.id)); }} onUndo={() => edit(scene => scene.undo())} />
        <section className="panel camera-panel" aria-label="Camera controls">
          <h2>Exhibit view</h2>
          <div className="view-presets">
            <button className="button" type="button" onClick={() => edit(scene => scene.fit())} disabled={!exhibit}><Expand size={15} /> Fit</button>
            {(['front', 'side', 'isometric'] as const).map(view => <button key={view} className="button" type="button" onClick={() => edit(scene => scene.fit(view))} disabled={!exhibit}>{view}</button>)}
            <button className="button" type="button" onClick={() => edit(scene => {
              scene.reset();
              scene.select(null);
              scene.fit('isometric');
              setSelected(null);
            })} disabled={!exhibit}>Reset exhibit</button>
          </div>
        </section>
        {operatorMode && <section className="panel" aria-label="Live hand camera">
          <div className="panel-heading"><h2>Hand camera</h2>
            <button className="button" type="button" onClick={toggleLive}>{liveEnabled ? 'Disconnect' : 'Connect'}</button>
          </div>
          <p className="setup-hint" role="status">{!liveEnabled ? 'Touch, mouse, and keyboard controls are ready.'
            : liveStatus.phase === 'ready' ? 'Connected. Close Setup to use hand gestures.'
            : liveStatus.phase === 'calibration' ? 'Calibrate the camera at this display to enable hand gestures.'
            : liveStatus.phase === 'connecting' ? 'Connecting to the local hand camera.' : liveStatus.message ?? 'Camera unavailable. Start the local camera service and check the USB connection. Retrying automatically.'}</p>
          {liveEnabled && (liveStatus.phase === 'ready' || liveStatus.phase === 'calibration') && <button className="button camera-calibrate" type="button" disabled={calibrationBusy || !!graphicsIssue} onClick={() => void changeCalibration('begin')}>Calibrate reach</button>}
          {calibrationError && <p className="setup-hint" role="alert">{calibrationError}</p>}
        </section>}
        {simulationMode && <section className="panel sim-panel" aria-label="Simulator controls">
          <div className="panel-heading"><h2>Simulated hands</h2>
            <button className="sim-toggle" role="switch" aria-label="Simulated hands" aria-checked={enabled} type="button" onClick={() => setEnabled(value => !value)} disabled={!!rendererError}><span /></button>
          </div>
          <dl className="shortcuts">
            <div><dt><kbd>W A S D</kbd></dt><dd>Move left hand or use the pointer</dd></div>
            <div><dt><kbd>Space</kbd></dt><dd>Hold left pinch or click and hold</dd></div>
            <div><dt><kbd>↑ ↓ ← →</kbd></dt><dd>Move right hand</dd></div>
            <div><dt><kbd>Enter</kbd></dt><dd>Hold right pinch. Aims at the join ring.</dd></div>
            <div><dt><kbd>V</kbd> <kbd>B</kbd></dt><dd>Hold left / right fist to move the view</dd></div>
            <div><dt><kbd>Q</kbd> <kbd>E</kbd> <span>/</span> <kbd>[</kbd> <kbd>]</kbd></dt><dd>Adjust left / right depth</dd></div>
            <div><dt><kbd>Esc</kbd></dt><dd>Drop tracking</dd></div>
          </dl>
          <p className="setup-hint">Shift-drag to orbit with the pointer. On touchscreens, drag a part to move it or drag empty space to orbit. Add a second finger at the ring to resize or zoom.</p>
        </section>}
      </div>
    </aside>}
  </main>;
}
