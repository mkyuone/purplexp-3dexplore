import { Box, Expand, Hand, LoaderCircle, Move, Orbit, Settings, ThumbsUp, TriangleAlert, Upload, X } from 'lucide-react';
import { useEffect, useMemo, useRef, useState, type ChangeEvent, type DragEvent } from 'react';
import { importStep, readStepFile, type CadAssembly } from './cad';
import { loadBundledAssembly, ultravioletFinishes } from './bundled-assembly';
import { CadScene, type SurfaceFinishes } from './scene';
import { GestureController, RESET_HOLD_MS, type GestureState } from './gestures';
import { HandSimulator } from './simulator';
import { AssemblyPanel, indexOccurrences, PartPanel } from './panels';
import { applyHandCameraSettings, calibrateHands, connectLiveHands, fetchHandCameraSettings, handDisplayKey, type HandCameraSettings, type HandNavigation, type LiveHandStatus } from './live-hands';
import { attachVisitorControls } from './visitor-controls';
import { CalibrationPanel } from './calibration-panel';
import { CameraPreview } from './camera-preview';
import { HeadPerspectivePanel, useHeadPerspective } from './head-perspective-panel';

const startingModelName = 'Ultraviolet Full Car.step';
const query = new URLSearchParams(window.location.search);
const simulationMode = import.meta.env.DEV && query.get('simulate') === '1';
const operatorMode = !simulationMode && query.get('operator') === '1';
const cameraPreference = 'spatial-cad-explorer.hand-camera';
const navigationPreference = 'spatial-cad-explorer.hand-navigation';
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

function visitorCue(state: GestureState | null, simulation: boolean) {
  if (state?.status === 'lost') return simulation
    ? 'Move your hands back into view and start with an open hand.'
    : 'Bring your hands back into view.';
  if (state?.status === 'both') return state.target === 'view' ? 'Move together to pan, tilt to orbit, or spread and close to zoom.' : 'Move together, spread to resize, or turn to rotate.';
  if (state?.status === 'holding') return state.target === 'part' ? 'Move the part, or add a second pinch to resize and rotate.' : 'Move your fist to orbit. Pull back to zoom in, push forward to zoom out.';
  return 'Pinch a part to move it. Make a fist to look around.';
}

function CameraSettings({ navigation }: { navigation: HandNavigation }) {
  const session = useRef<AbortController | null>(null);
  const requestVersion = useRef(0);
  const posting = useRef(false);
  const dirty = useRef(false);
  const [settings, setSettings] = useState<HandCameraSettings | null>(null);
  const [activeFps, setActiveFps] = useState<number | null>(null);
  const [draftFps, setDraftFps] = useState<number | null>(null);
  const [sending, setSending] = useState(false);
  const [pollError, setPollError] = useState<string | null>(null);
  const [applyError, setApplyError] = useState<string | null>(null);

  function receive(next: HandCameraSettings) {
    setSettings(next);
    if (!next.applying) {
      if (!next.error) setActiveFps(next.fps);
      if (!dirty.current) setDraftFps(next.fps);
    } else setDraftFps(current => current ?? next.fps);
  }

  useEffect(() => {
    const controller = new AbortController();
    session.current = controller;
    let timer: ReturnType<typeof setTimeout>;
    async function poll() {
      const version = requestVersion.current;
      try {
        if (posting.current) return;
        const next = await fetchHandCameraSettings(controller.signal);
        if (controller.signal.aborted || version !== requestVersion.current) return;
        receive(next);
        setPollError(null);
      } catch (cause) {
        if (!controller.signal.aborted && version === requestVersion.current) {
          setPollError(cause instanceof Error ? cause.message : 'Camera settings are unavailable.');
        }
      } finally {
        if (!controller.signal.aborted) timer = setTimeout(poll, 1000);
      }
    }
    void poll();
    return () => {
      controller.abort();
      clearTimeout(timer);
      session.current = null;
    };
  }, []);

  const busy = sending || settings?.applying === true;
  const validDraft = draftFps !== null && settings?.frameRates.includes(draftFps);
  const changed = draftFps !== activeFps;

  function editDraft(fps: number) {
    dirty.current = true;
    setDraftFps(fps);
    setApplyError(null);
  }

  async function apply() {
    const controller = session.current;
    if (!controller || draftFps === null || !validDraft || busy || posting.current) return;
    posting.current = true;
    requestVersion.current++;
    setSending(true);
    setApplyError(null);
    try {
      const next = await applyHandCameraSettings(draftFps, controller.signal);
      if (controller.signal.aborted) return;
      dirty.current = false;
      setDraftFps(next.fps);
      receive(next);
      setPollError(null);
    } catch (cause) {
      if (!controller.signal.aborted) setApplyError(cause instanceof Error ? cause.message : 'The camera settings could not be applied.');
    } finally {
      posting.current = false;
      if (!controller.signal.aborted) setSending(false);
    }
  }

  const metrics = settings?.diagnostics;
  const issue = applyError ?? settings?.error ?? pollError;
  const rgbOnly = navigation === 'fuzzy' && metrics?.depthHands !== undefined
    && (metrics.trackedHands ?? 0) > metrics.depthHands;
  const waitingForDepth = (metrics?.estimatedHands ?? 0) > 0 && metrics?.depthHands === 0;
  return <div className="camera-testing">
    <fieldset disabled={busy || !settings}>
      <legend>Camera settings</legend>
      {settings && draftFps !== null ? <>
        <label className="navigation-choice">Capture rate
          <select value={draftFps} onChange={event => editDraft(Number(event.target.value))}>
            {settings.frameRates.map(fps => <option key={fps} value={fps}>{fps} fps</option>)}
          </select>
        </label>
        <p className="camera-option-note">Changing the capture rate restarts the camera.</p>
        <button className="button" type="button" disabled={!changed || !validDraft} onClick={() => void apply()}>
          {busy && <LoaderCircle size={15} className="camera-settings-spinner" />}{busy ? 'Applying…' : 'Apply and restart'}
        </button>
      </> : <p className="camera-option-note">Loading camera settings…</p>}
    </fieldset>
    {busy && <p className="camera-option-note" role="status">Restarting the camera with the requested settings…</p>}
    {issue && <p className="camera-settings-error" role="alert">{issue}</p>}
    {activeFps !== null && <dl className="camera-diagnostics" aria-label="Active camera diagnostics">
      <div><dt>Active capture</dt><dd>{activeFps} fps</dd></div>
      {!busy && metrics && [metrics.estimatedHands, metrics.depthHands, metrics.trackedHands].some(value => value !== undefined) && <div><dt>Hands</dt><dd>{metrics.estimatedHands ?? '—'} seen · {metrics.depthHands ?? '—'} depth · {metrics.trackedHands ?? '—'} tracked</dd></div>}
      {!busy && metrics?.poseFps !== undefined && <div><dt>Pose rate</dt><dd>{metrics.poseFps.toFixed(1)} fps</dd></div>}
      {!busy && metrics?.processingMs !== undefined && <div><dt>Processing</dt><dd>{metrics.processingMs.toFixed(1)} ms</dd></div>}
      {!busy && metrics?.frameAgeMs !== undefined && <div><dt>Frame age</dt><dd>{Math.round(metrics.frameAgeMs)} ms</dd></div>}
    </dl>}
    {!busy && (rgbOnly || waitingForDepth) && <p className="camera-option-note">
      {rgbOnly ? 'RGB tracking active. Depth movement is paused.' : 'Hand seen, waiting for palm depth.'}
    </p>}
  </div>;
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
  const [navigation, setNavigation] = useState<HandNavigation>(() => {
    try { return localStorage.getItem(navigationPreference) === 'calibrated' ? 'calibrated' : 'fuzzy'; }
    catch { return 'fuzzy'; }
  });
  const [previewVisible, setPreviewVisible] = useState(true);
  const [liveStatus, setLiveStatus] = useState<LiveHandStatus>({ phase: 'connecting' });
  const [calibrationProgress, setCalibrationProgress] = useState<LiveHandStatus | null>(null);
  const [calibrationBusy, setCalibrationBusy] = useState(false);
  const [calibrationError, setCalibrationError] = useState<string | null>(null);
  const [calibrationSaved, setCalibrationSaved] = useState(false);
  const [hasExplored, setHasExplored] = useState(false);
  const [selected, setSelected] = useState<string | null>(null);
  const [pose, setPose] = useState<GestureState | null>(null);
  const [, setRevision] = useState(0);
  const headPerspective = useHeadPerspective(runtime, setupOpen);
  const calibrating = calibrationProgress !== null;
  const calibrationDisplayChanged = calibrating && liveStatus.display !== undefined && liveStatus.display !== calibrationProgress.display;
  const matchingCalibrationPoll = liveStatus.phase === 'calibration' && liveStatus.display === calibrationProgress?.display
    && liveStatus.step === calibrationProgress?.step ? liveStatus : null;
  const calibrationOffline = liveStatus.phase === 'offline';
  const captureReady = !calibrationDisplayChanged && !calibrationOffline
    && (matchingCalibrationPoll?.captureReady ?? calibrationProgress?.captureReady);
  const captureHint = calibrationDisplayChanged ? 'The display changed. Cancel and restart calibration in Setup.'
    : calibrationOffline ? liveStatus.message ?? 'Waiting for the hand camera.'
      : matchingCalibrationPoll?.captureHint ?? calibrationProgress?.captureHint;
  const roots = exhibit?.roots ?? emptyRoots;
  const index = useMemo(() => indexOccurrences(roots), [roots]);
  const part = selected === null ? undefined : index.byId.get(selected);
  const focusId = pose?.held ?? pose?.hovered ?? selected;
  const focusPart = focusId ? index.byId.get(focusId) : undefined;
  const scene = runtime.current?.scene;
  const details = setupOpen && part ? scene?.describe(part.id) : null;

  async function load(name: string, read: (signal: AbortSignal) => Promise<CadAssembly>, finishes?: SurfaceFinishes) {
    importAbort.current?.abort();
    const controller = new AbortController();
    importAbort.current = controller;
    runtime.current?.gestures.resetStream();
    setLoading(name);
    setError(null);
    try {
      const assembly = await read(controller.signal);
      const active = runtime.current;
      if (controller.signal.aborted || !active) return;
      active.gestures.cancel();
      active.scene.load(assembly, finishes);
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
    return load(startingModelName, loadBundledAssembly, ultravioletFinishes);
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
      scene.setHeadGripHeld(state.held !== null);
      setSelected(scene.getSelected());
      setPose(state);
      if (state.target) setHasExplored(true);
    }, {
      joinAnywhere: true,
      liveTracking: !simulationMode,
      onReset: () => {
        const active = runtime.current;
        if (!active || active.setupOpen || active.calibrating || active.contextLost || !active.hasModel || importAbort.current) return;
        resetExhibit(false);
      },
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
        if (!active.setupOpen && !active.contextLost && active.hasModel && !importAbort.current) gestures.receive(frame);
      }, (x, y) => scene.pick(x, y) ? 'part' : 'view');
      active.simulator.setEnabled(true);
    } else {
      detachControls = attachVisitorControls(surface, scene, {
        available: () => active.hasModel && !active.contextLost && !active.calibrating,
        pauseHands: (pointerHeld) => {
          const now = performance.now();
          if (now >= active.nativeUntil) gestures.resetStream(false);
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
      navigation,
      receive: frame => {
        if (!active.setupOpen && !active.calibrating && !active.contextLost && active.hasModel && !importAbort.current && performance.now() >= active.nativeUntil) active.gestures.receive(frame);
      },
      reset: () => active.gestures.resetStream(),
      status: setLiveStatus,
    });
    return () => {
      disconnect();
      if (runtime.current === active) active.gestures.cancel();
    };
  }, [liveEnabled, navigation]);

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

  function edit(change: (scene: CadScene) => void, cancelGestures = true) {
    const active = runtime.current;
    if (!active || active.contextLost) return;
    if (cancelGestures) active.gestures.cancel();
    change(active.scene);
    setRevision(value => value + 1);
  }

  function resetExhibit(cancelGestures = true) {
    edit(scene => {
      scene.reset();
      scene.select(null);
      scene.fit('isometric');
      setSelected(null);
    }, cancelGestures);
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

  useEffect(() => {
    if (!calibrationSaved) return;
    const timer = window.setTimeout(() => setCalibrationSaved(false), 6500);
    return () => window.clearTimeout(timer);
  }, [calibrationSaved]);

  function toggleLive() {
    const next = !liveEnabled;
    try { localStorage.setItem(cameraPreference, next ? 'on' : 'off'); }
    catch { /* The camera can still be used for this session. */ }
    setLiveEnabled(next);
    setCalibrationError(null);
  }

  function changeNavigation(next: HandNavigation) {
    try { localStorage.setItem(navigationPreference, next); }
    catch { /* This mode remains available for the current session. */ }
    setNavigation(next);
    runtime.current?.gestures.resetStream();
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
      if (action === 'begin') {
        if (next.phase !== 'calibration') throw new Error(next.message ?? 'The hand camera is unavailable.');
        setCalibrationSaved(false);
        setNavigation('calibrated');
        calibrationDisplay.current = display;
        const active = runtime.current;
        if (active) active.calibrating = true;
        setCalibrationProgress(next);
        closeSetup();
      } else if (action === 'cancel' || next.phase === 'ready') {
        if (action === 'capture') {
          setCalibrationSaved(true);
          changeNavigation('calibrated');
        } else changeNavigation('fuzzy');
        finishCalibration();
      } else if (next.phase === 'calibration') {
        setCalibrationProgress(current => ({ ...next, previewToken: current?.previewToken }));
      } else {
        setCalibrationError(next.message ?? 'The hand camera is unavailable.');
      }
    } catch (cause) {
      if (action === 'cancel') {
        changeNavigation('fuzzy');
        finishCalibration();
      } else setCalibrationError(cause instanceof Error ? cause.message : 'The camera did not respond.');
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
  const dualViewGrip = pose?.target === 'view' && pose.status === 'both';
  const resetProgress = showPose && !loading && !graphicsIssue ? Math.max(0, Math.min(1, pose?.resetProgress ?? 0)) : 0;
  const resetSeconds = Math.max(0.1, (1 - resetProgress) * RESET_HOLD_MS / 1000).toFixed(1);
  const immersiveHeadView = headPerspective.enabled && !setupOpen;
  const visibleHint = !immersiveHeadView && !resetProgress && !setupOpen && !calibrating && exhibit && !contextLost && (calibrationSaved || !hasExplored) && (!simulationMode || enabled);
  const hint = calibrationSaved ? 'Calibration saved. Point at the model to try it.'
    : handsEnabled ? visitorCue(pose, simulationMode) : 'Drag to look around. Tap a part to inspect it.';

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
      {resetProgress > 0 && <div className="reset-countdown">
        <ThumbsUp size={21} aria-hidden="true" />
        <div>
          <p role="status">
            <span className="sr-only">Hold thumbs up until the exhibit resets.</span>
            <span aria-hidden="true">Hold thumbs up · Reset in {resetSeconds} s</span>
          </p>
          <progress value={resetProgress} max={1} aria-hidden="true" />
        </div>
      </div>}
      {operatorMode && liveEnabled && previewVisible && !calibrating && !immersiveHeadView && <CameraPreview className="navigation-preview" display={liveStatus.display ?? handDisplayKey()} />}
      {!setupOpen && !calibrating && (stageMessage || !exhibit) && <div className="stage-message">
        {stageMessage}
        {!exhibit && !loading && !graphicsIssue && <div className="empty-stage">
          <Box size={35} /><p>Open Setup to load an assembly.</p>
          <button type="button" className="button" onClick={() => void loadStartingModel()}>Load Ultraviolet</button>
        </div>}
      </div>}
      {showPose && <div className="hands-layer" aria-hidden="true">
        {pose?.hands.map(hand => {
          const active = pose.activeHands.includes(hand.id);
          const movingView = active && hand.fist;
          return <div key={hand.id}
            className={`hand-cursor${hand.id === 'right' ? ' second' : ''}${active && hand.pinched ? ' is-pinched' : ''}${movingView ? ' is-view-grip' : ''}`}
            style={{ left: `${hand.x * 100}%`, top: `${hand.y * 100}%`, transform: `translate(-50%, -50%) scale(${Math.max(0.65, Math.min(1.45, 1 + hand.depth))})` }}>
            {movingView ? (dualViewGrip ? <Move size={25} /> : <Orbit size={25} />) : <Hand size={25} />}
          </div>;
        })}
      </div>}
      {!setupOpen && !calibrating && (simulationMode || focusPart || pose?.target === 'view') && <footer className="exhibit-footer">
        {pose?.target === 'view'
          ? <div className="part-caption">{dualViewGrip ? <Move size={16} /> : <Orbit size={16} />}<span>{dualViewGrip ? 'Pan · Orbit · Zoom' : 'Exhibit view'}</span></div>
          : focusPart && <div className="part-caption" aria-live="polite"><Box size={16} /><span>{focusPart.name}</span></div>}
        {simulationMode && <span className="input-status" data-active={enabled && pose?.status !== 'lost' && !rendererError}><i />{enabled ? 'Simulated hands' : 'Simulation paused'}</span>}
      </footer>}
      {calibrationProgress && <CalibrationPanel step={calibrationProgress.step ?? 0} message={calibrationError ?? calibrationProgress.message ?? null} busy={calibrationBusy}
        captureReady={captureReady} captureHint={captureHint} previewDisplay={calibrationProgress.display} previewToken={calibrationProgress.previewToken}
        onCapture={() => void changeCalibration('capture')} onCancel={() => void changeCalibration('cancel')} />}
    </section>

    {setupOpen && <aside id="exhibit-setup" className="setup-panel" aria-labelledby="setup-title">
      <div className="setup-head">
        <div><h2 id="setup-title">{operatorMode ? 'Operator setup' : 'Exhibit setup'}</h2><p>Choose an assembly and prepare the exhibit.</p></div>
        <button ref={setupClose} className="button icon-button" type="button" aria-label="Close setup" onClick={closeSetup}><X size={20} /></button>
      </div>
      <div className="setup-scroll" onDragOver={event => { if (event.dataTransfer.types.includes('Files')) event.preventDefault(); }} onDrop={onDrop}>
        {operatorMode && <HeadPerspectivePanel controls={headPerspective} unavailable={!!graphicsIssue || !exhibit} />}
        {operatorMode && <section className="panel" aria-label="Live hand camera">
          <div className="panel-heading"><h2>Hand camera</h2>
            <button className="button" type="button" onClick={toggleLive}>{liveEnabled ? 'Disconnect' : 'Connect'}</button>
          </div>
          <p className="setup-hint" role="status">{!liveEnabled ? 'Touch, mouse, and keyboard controls are ready.'
            : liveStatus.phase === 'ready' ? navigation === 'fuzzy' ? 'Fuzzy navigation is ready. Close Setup and move an open hand into view.' : 'Connected. Close Setup to use hand gestures.'
            : liveStatus.phase === 'calibration' ? 'This display needs a reach profile. Choose fuzzy navigation or calibrate below.'
            : liveStatus.phase === 'connecting' ? 'Connecting to the local hand camera.' : liveStatus.message ?? 'Camera unavailable. Start the local camera service and check the USB connection. Retrying automatically.'}</p>
          {liveEnabled && <label className="navigation-choice">Hand navigation
            <select value={navigation} onChange={event => changeNavigation(event.target.value as HandNavigation)}>
              <option value="fuzzy">Fuzzy · no calibration</option>
              <option value="calibrated">Calibrated reach</option>
            </select>
          </label>}
          {liveEnabled && <label className="preview-choice">
            <input type="checkbox" checked={previewVisible} onChange={event => setPreviewVisible(event.target.checked)} />
            Show live camera preview
          </label>}
          {liveEnabled && (liveStatus.phase === 'ready' || liveStatus.phase === 'calibration') && <button className="button camera-calibrate" type="button" disabled={calibrationBusy || !!graphicsIssue} onClick={() => void changeCalibration('begin')}>{navigation === 'fuzzy' ? 'Optional reach calibration' : 'Calibrate reach'}</button>}
          {calibrationError && <p className="setup-hint" role="alert">{calibrationError}</p>}
          <CameraSettings navigation={navigation} />
        </section>}
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
            <button className="button" type="button" onClick={() => resetExhibit()} disabled={!exhibit}>Reset exhibit</button>
          </div>
        </section>
        {simulationMode && <section className="panel sim-panel" aria-label="Simulator controls">
          <div className="panel-heading"><h2>Simulated hands</h2>
            <button className="sim-toggle" role="switch" aria-label="Simulated hands" aria-checked={enabled} type="button" onClick={() => setEnabled(value => !value)} disabled={!!rendererError}><span /></button>
          </div>
          <dl className="shortcuts">
            <div><dt><kbd>W A S D</kbd></dt><dd>Move left hand or use the pointer</dd></div>
            <div><dt><kbd>Space</kbd></dt><dd>Hold left pinch or click and hold</dd></div>
            <div><dt><kbd>↑ ↓ ← →</kbd></dt><dd>Move right hand</dd></div>
            <div><dt><kbd>Enter</kbd></dt><dd>Hold right pinch to resize and rotate</dd></div>
            <div><dt><kbd>V</kbd> <kbd>B</kbd></dt><dd>Hold one fist to orbit and zoom with depth. With both, move together to pan, tilt to orbit, or spread and close to zoom.</dd></div>
            <div><dt><kbd>Q</kbd> <kbd>E</kbd> <span>/</span> <kbd>[</kbd> <kbd>]</kbd></dt><dd>Adjust left / right depth</dd></div>
            <div><dt><kbd>Esc</kbd></dt><dd>Drop tracking</dd></div>
          </dl>
          <p className="setup-hint">Shift-drag to orbit with the pointer. On touchscreens, drag a part to move it or drag empty space to orbit. Add a second finger to resize a part or pan, orbit, and zoom the view.</p>
        </section>}
      </div>
    </aside>}
  </main>;
}
