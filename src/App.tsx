import { Box, Expand, Hand, LoaderCircle, Move, Orbit, Settings, ThumbsUp, TriangleAlert, Upload, X } from 'lucide-react';
import { memo, useEffect, useImperativeHandle, useMemo, useRef, useState, type ChangeEvent, type DragEvent, type MouseEvent, type Ref } from 'react';
import { importStep, readStepFile, type CadAssembly, type CadOccurrence } from './cad';
import { sha256 } from './assembly-binary';
import { loadBundledAssembly, ultravioletFinishes } from './bundled-assembly';
import { CadScene, type SurfaceFinishes } from './scene';
import { GestureController, RESET_HOLD_MS, type GestureState } from './gestures';
import { HandSimulator } from './simulator';
import { AssemblyPanel, indexOccurrences, PartPanel } from './panels';
import { applyHandCameraSettings, calibrateHands, connectLiveHands, fetchHandCameraSettings, handDisplayKey, type HandCameraSettings, type HandNavigation, type LiveHandStatus } from './live-hands';
import { attachVisitorControls } from './visitor-controls';
import { CalibrationPanel } from './calibration-panel';
import { CameraPreview } from './camera-preview';
import { DepthRangePanel } from './depth-range-panel';
import { HeadPerspectivePanel, useHeadPerspective } from './head-perspective-panel';
import { HandPerformance } from './hand-performance';
import { collectPartNameRows, exportPartNames, importPartNames } from './part-names';

const startingModelName = 'Ultraviolet Full Car.step';
const query = new URLSearchParams(window.location.search);
const simulationMode = import.meta.env.DEV && query.get('simulate') === '1';
const operatorPath = '/operator';
const cameraPreference = 'spatial-cad-explorer.hand-camera';
const navigationPreference = 'spatial-cad-explorer.hand-navigation';
const previewPreference = 'spatial-cad-explorer.visitor-camera-preview';
const partNamesPrefix = 'spatial-cad-explorer.part-names.';
const emptyRoots: CadAssembly['roots'] = [];

function legacyOperatorUrl(url: URL) {
  return url.pathname === '/' && url.searchParams.get('operator') === '1';
}

function isOperatorPage() {
  const url = new URL(window.location.href);
  return !simulationMode && (url.pathname.replace(/\/$/, '') === operatorPath || legacyOperatorUrl(url));
}

function readPreviewPreference() {
  try { return localStorage.getItem(previewPreference) === 'on'; }
  catch { return false; }
}

function partNamesKey(sourceHash: Uint8Array | null) {
  if (!sourceHash) return null;
  const hex = Array.from(sourceHash, byte => byte.toString(16).padStart(2, '0')).join('');
  return partNamesPrefix + hex;
}

function readPartNames(key: string | null): Record<string, string> | undefined {
  if (!key) return undefined;
  try {
    const value = JSON.parse(localStorage.getItem(key) ?? '{}') as unknown;
    if (!value || typeof value !== 'object' || Array.isArray(value)) return {};
    return Object.fromEntries(Object.entries(value).filter(([, name]) => typeof name === 'string' && (name === '' || name.trim())));
  } catch {
    return undefined;
  }
}

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
type CalibrationAttempt = { display: string; previousNavigation: HandNavigation; pending: boolean; abandoned: boolean };
type Exhibit = {
  name: string;
  roots: CadAssembly['roots'];
  revision: number;
  namesKey: string | null;
  defaultDisplayNames: Readonly<Record<string, string>>;
  displayNames: Readonly<Record<string, string>>;
};
type LoadedAssembly = {
  assembly: CadAssembly;
  sourceSha256: Uint8Array | null;
  defaultDisplayNames?: Readonly<Record<string, string>>;
};
type GestureFeedback = Pick<GestureState, 'status' | 'target' | 'held' | 'hovered' | 'resetProgress'>;
type CursorControls = { update: (state: GestureState) => void };

const HandCursors = memo(function HandCursors({ visible, controls }: { visible: boolean; controls: Ref<CursorControls> }) {
  const [state, setState] = useState<GestureState | null>(null);
  useImperativeHandle(controls, () => ({ update: setState }), []);
  if (!visible) return null;
  return <div className="hands-layer" aria-hidden="true">
    {state?.hands.map(hand => {
      const active = state.activeHands.includes(hand.id);
      const movingView = active && hand.fist;
      const scale = Math.max(0.65, Math.min(1.45, 1 + hand.depth));
      return <div key={hand.id}
        className={`hand-cursor${hand.id === 'right' ? ' second' : ''}${active && hand.pinched ? ' is-pinched' : ''}${movingView ? ' is-view-grip' : ''}`}
        style={{ transform: `translate3d(${hand.x * 100}cqw, ${hand.y * 100}cqh, 0) translate(-50%, -50%) scale(${scale})` }}>
        {movingView ? (state.status === 'both' ? <Move size={25} /> : <Orbit size={25} />) : <Hand size={25} />}
      </div>;
    })}
  </div>;
});

function visitorCue(state: GestureFeedback | null, simulation: boolean) {
  if (state?.status === 'lost') return simulation
    ? 'Move your hands back into view and start with an open hand.'
    : 'Bring your hands back into view.';
  if (state?.status === 'both') return state.target === 'view' ? 'Move together to pan, or spread and close to zoom.' : 'Move together, spread to resize, or turn to rotate.';
  if (state?.status === 'holding') return state.target === 'part' ? 'Move the part, or add a second pinch to resize and rotate.' : 'Move your fist to orbit. Pull back to zoom in, push forward to zoom out.';
  return 'Pinch a part to move it. Make a fist to look around.';
}

function CameraSettings({ navigation, performance }: { navigation: HandNavigation; performance: HandPerformance }) {
  const session = useRef<AbortController | null>(null);
  const requestVersion = useRef(0);
  const posting = useRef(false);
  const dirty = useRef(false);
  const [settings, setSettings] = useState<HandCameraSettings | null>(null);
  const [browser, setBrowser] = useState(() => performance.read());
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
      setBrowser(performance.read());
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
  }, [performance]);

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
  const depthUnavailable = metrics?.depthFrameAvailable === false;
  const rgbOnly = navigation === 'fuzzy' && (depthUnavailable
    || (metrics?.depthHands !== undefined && (metrics.trackedHands ?? 0) > metrics.depthHands));
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
      {!busy && metrics?.processingP95Ms !== undefined && <div><dt>Processing p95</dt><dd>{metrics.processingP95Ms.toFixed(1)} ms</dd></div>}
      {!busy && metrics?.inferenceMs !== undefined && <div><dt>Bridge stages</dt><dd>{metrics.inferenceMs.toFixed(1)} ms inference · {metrics.alignmentMs?.toFixed(1) ?? '—'} ms align · {metrics.fusionMs?.toFixed(1) ?? '—'} ms fusion</dd></div>}
      {!busy && metrics?.queueMs !== undefined && <div><dt>Capture queue</dt><dd>{metrics.queueMs.toFixed(1)} ms</dd></div>}
      {!busy && metrics?.frameAgeMs !== undefined && <div><dt>Pose age</dt><dd>{Math.round(metrics.frameAgeMs)} ms at bridge{metrics.frameAgeSource ? ` from ${metrics.frameAgeSource === 'hostArrival' ? 'host arrival' : 'SDK dequeue'}` : ''}</dd></div>}
      {!busy && metrics?.captureDroppedFrames !== undefined && <div><dt>Capture skipped</dt><dd>{metrics.captureDroppedFrames} this session</dd></div>}
      {!busy && browser.deliveryFps !== undefined && <div><dt>Browser rate</dt><dd>{browser.deliveryFps.toFixed(1)} fps · {browser.skipped ?? 0} skipped in recent samples</dd></div>}
      {!busy && browser.receiptToSubmitMs !== undefined && <div><dt>Last interaction</dt><dd>{browser.receiptToSubmitMs.toFixed(1)} ms from receipt · {browser.poseToSubmitMs?.toFixed(1)} ms from pose to render submission (p95)</dd></div>}
    </dl>}
    {!busy && (rgbOnly || waitingForDepth || depthUnavailable) && <p className="camera-option-note">
      {rgbOnly ? 'RGB tracking active. Depth movement is paused.'
        : depthUnavailable ? 'Calibrated navigation is waiting for depth.' : 'Hand seen, waiting for palm depth.'}
    </p>}
  </div>;
}

export function App() {
  const viewport = useRef<HTMLElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const setupClose = useRef<HTMLButtonElement>(null);
  const operatorTitle = useRef<HTMLHeadingElement>(null);
  const picker = useRef<HTMLInputElement>(null);
  const runtime = useRef<Runtime | null>(null);
  const cursorControls = useRef<CursorControls>(null);
  const [handPerformance] = useState(() => new HandPerformance());
  const importAbort = useRef<AbortController | null>(null);
  const calibrationAttempt = useRef<CalibrationAttempt | null>(null);
  const modelRevision = useRef(0);
  const currentExhibit = useRef<Exhibit | null>(null);
  const csvRequest = useRef(0);
  const [exhibit, setExhibit] = useState<Exhibit | null>(null);
  const [namesMessage, setNamesMessage] = useState<{ text: string; error?: boolean } | null>(null);
  const [loading, setLoading] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [rendererError, setRendererError] = useState<string | null>(null);
  const [contextLost, setContextLost] = useState(false);
  const [operatorMode, setOperatorMode] = useState(isOperatorPage);
  const operatorSession = useRef(operatorMode);
  const [simulatorSetupOpen, setSimulatorSetupOpen] = useState(false);
  const [enabled, setEnabled] = useState(simulationMode);
  const [liveEnabled, setLiveEnabled] = useState(() => {
    try { return !simulationMode && localStorage.getItem(cameraPreference) === 'on'; }
    catch { return false; }
  });
  const [navigation, setNavigation] = useState<HandNavigation>(() => {
    try { return localStorage.getItem(navigationPreference) === 'calibrated' ? 'calibrated' : 'fuzzy'; }
    catch { return 'fuzzy'; }
  });
  const [visitorPreviewVisible, setVisitorPreviewVisible] = useState(readPreviewPreference);
  const [liveStatus, setLiveStatus] = useState<LiveHandStatus>({ phase: 'connecting' });
  const [calibrationProgress, setCalibrationProgress] = useState<LiveHandStatus | null>(null);
  const [calibrationBusy, setCalibrationBusy] = useState(false);
  const [calibrationError, setCalibrationError] = useState<string | null>(null);
  const [calibrationSaved, setCalibrationSaved] = useState(false);
  const [hasExplored, setHasExplored] = useState(false);
  const [selected, setSelected] = useState<string | null>(null);
  const [pose, setPose] = useState<GestureFeedback | null>(null);
  const [, setRevision] = useState(0);
  const calibrating = operatorMode && calibrationProgress !== null;
  const setupOpen = operatorMode ? !calibrating : simulatorSetupOpen;
  const headPerspective = useHeadPerspective(runtime, setupOpen);
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

  function setCurrentExhibit(next: Exhibit) {
    currentExhibit.current = next;
    setExhibit(next);
  }

  function displayName(node: CadOccurrence) {
    const custom = exhibit?.displayNames[node.id] ?? exhibit?.defaultDisplayNames[node.id];
    return typeof custom === 'string' && custom ? custom : node.name;
  }

  function savePartNames(current: Exhibit, updates: Readonly<Record<string, string>>) {
    const names: Record<string, string> = Object.assign(Object.create(null), current.displayNames);
    for (const [id, name] of Object.entries(updates)) {
      if (name === (current.defaultDisplayNames[id] ?? '')) delete names[id];
      else names[id] = name;
    }
    setCurrentExhibit({ ...current, displayNames: names });
    if (current.namesKey) {
      try { localStorage.setItem(current.namesKey, JSON.stringify(names)); }
      catch { /* Names still apply in this tab. */ }
    }
  }

  useEffect(() => {
    function syncPartNames(event: StorageEvent) {
      const current = currentExhibit.current;
      if (!current?.namesKey || (event.key !== current.namesKey && event.key !== null)) return;
      const names = readPartNames(current.namesKey);
      if (names) setCurrentExhibit({ ...current, displayNames: names });
    }
    window.addEventListener('storage', syncPartNames);
    return () => window.removeEventListener('storage', syncPartNames);
  }, []);

  async function load(name: string, read: (signal: AbortSignal) => Promise<LoadedAssembly>, finishes?: SurfaceFinishes) {
    importAbort.current?.abort();
    const controller = new AbortController();
    importAbort.current = controller;
    runtime.current?.gestures.resetStream();
    setLoading(name);
    setError(null);
    try {
      const { assembly, sourceSha256, defaultDisplayNames = {} } = await read(controller.signal);
      const active = runtime.current;
      if (controller.signal.aborted || !active) return;
      active.gestures.cancel();
      active.scene.load(assembly, finishes);
      active.hasModel = true;
      const namesKey = partNamesKey(sourceSha256);
      setCurrentExhibit({ name, roots: assembly.roots, revision: ++modelRevision.current, namesKey, defaultDisplayNames, displayNames: readPartNames(namesKey) ?? {} });
      setNamesMessage(null);
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

  function downloadPartNames() {
    if (!exhibit) return;
    const rows = collectPartNameRows(exhibit.roots, { ...exhibit.defaultDisplayNames, ...exhibit.displayNames });
    const url = URL.createObjectURL(new Blob([exportPartNames(rows)], { type: 'text/csv;charset=utf-8' }));
    const link = document.createElement('a');
    link.href = url;
    link.download = `${exhibit.name.replace(/\.(step|stp)$/i, '')}-part-display-names.csv`;
    link.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  async function receivePartNames(file: File) {
    const revision = currentExhibit.current?.revision;
    if (revision === undefined) return;
    const request = ++csvRequest.current;
    try {
      const text = await file.text();
      const current = currentExhibit.current;
      if (csvRequest.current !== request || current?.revision !== revision) return;
      const updates = importPartNames(text, collectPartNameRows(current.roots, {}));
      savePartNames(current, updates);
      const count = Object.keys(updates).length;
      setNamesMessage({ text: `Imported ${count} ${count === 1 ? 'row' : 'rows'} of display names.` });
    } catch (cause) {
      if (csvRequest.current === request && currentExhibit.current?.revision === revision) {
        setNamesMessage({ text: cause instanceof Error ? cause.message : 'The display names could not be imported.', error: true });
      }
    }
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
      cursorControls.current?.update(state);
      const { status, target, held, hovered, resetProgress } = state;
      setPose(previous => previous?.status === status && previous.target === target && previous.held === held
        && previous.hovered === hovered && previous.resetProgress === resetProgress
        ? previous : { status, target, held, hovered, resetProgress });
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
    const active: Runtime = { scene, gestures, setupOpen, calibrating: false, contextLost: false, hasModel: false, nativeUntil: 0 };
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
    let latestReceipt: number | undefined;
    const disconnect = connectLiveHands({
      navigation,
      receive: (frame, receivedAt) => {
        if (operatorSession.current) {
          latestReceipt = receivedAt;
          handPerformance.receive(frame.sequence, receivedAt);
        }
        if (active.setupOpen || active.calibrating || active.contextLost || !active.hasModel || importAbort.current || receivedAt < active.nativeUntil) return;
        active.gestures.receive(frame);
        if (operatorSession.current) active.scene.requestRenderMeasurement(submittedAt => {
          if (latestReceipt === receivedAt) handPerformance.submitted(frame, receivedAt, submittedAt);
        });
      },
      reset: reason => {
        latestReceipt = undefined;
        handPerformance.reset();
        active.gestures.resetStream(reason === 'depth-range' ? false : undefined);
      },
      status: setLiveStatus,
    });
    return () => {
      disconnect();
      latestReceipt = undefined;
      handPerformance.reset();
      if (runtime.current === active) active.gestures.cancel();
    };
  }, [handPerformance, liveEnabled, navigation]);

  function receiveFile(file?: File) {
    if (!file) return;
    if (!/\.st(?:e)?p$/i.test(file.name)) {
      setError('Choose a STEP assembly or part with a .step or .stp extension.');
      return;
    }
    void load(file.name, async signal => {
      const data = await readStepFile(file, signal);
      const sourceSha256 = globalThis.crypto?.subtle ? await sha256(data) : null;
      return { assembly: await importStep(data, signal), sourceSha256 };
    });
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
      scene.resetExhibit();
      scene.select(null);
      setSelected(null);
    }, cancelGestures);
  }

  function closeSetup() {
    setSimulatorSetupOpen(false);
    requestAnimationFrame(() => viewport.current?.focus({ preventScroll: true }));
  }

  useEffect(() => {
    const active = runtime.current;
    if (active && active.setupOpen !== setupOpen) {
      active.setupOpen = setupOpen;
      active.gestures.resetStream();
    }
    active?.simulator?.setEnabled(enabled && !setupOpen);
  }, [enabled, setupOpen]);

  function syncRoute() {
    const next = isOperatorPage();
    if (next) operatorSession.current = true;
    else leaveCalibration();
    setOperatorMode(next);
  }

  useEffect(() => {
    const url = new URL(window.location.href);
    if (!simulationMode && (legacyOperatorUrl(url) || url.pathname === `${operatorPath}/`)) {
      url.pathname = operatorPath;
      url.searchParams.delete('operator');
      window.history.replaceState(window.history.state, '', url);
    }
    window.addEventListener('popstate', syncRoute);
    return () => window.removeEventListener('popstate', syncRoute);
  }, []);

  useEffect(() => {
    document.title = operatorMode ? 'Operator settings · Spatial CAD Explorer' : 'Spatial CAD Explorer';
    if (setupOpen) (operatorMode ? operatorTitle.current : setupClose.current)?.focus({ preventScroll: true });
  }, [operatorMode, setupOpen]);

  function openExhibit(event: MouseEvent<HTMLAnchorElement>) {
    if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    event.preventDefault();
    window.history.pushState(null, '', event.currentTarget.href);
    syncRoute();
    requestAnimationFrame(() => canvas.current?.focus({ preventScroll: true }));
  }

  useEffect(() => {
    function syncPreview(event: StorageEvent) {
      if (event.key === previewPreference || event.key === null) setVisitorPreviewVisible(readPreviewPreference());
    }
    window.addEventListener('storage', syncPreview);
    return () => window.removeEventListener('storage', syncPreview);
  }, []);

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

  function changePreview(next: boolean) {
    try { localStorage.setItem(previewPreference, next ? 'on' : 'off'); }
    catch { /* The preference still applies to this tab. */ }
    setVisitorPreviewVisible(next);
  }

  function changeNavigation(next: HandNavigation) {
    try { localStorage.setItem(navigationPreference, next); }
    catch { /* This mode remains available for the current session. */ }
    setNavigation(next);
    runtime.current?.gestures.resetStream();
    setCalibrationError(null);
  }

  function finishCalibration() {
    calibrationAttempt.current = null;
    const active = runtime.current;
    if (active) {
      active.calibrating = false;
      active.gestures.resetStream();
    }
    setCalibrationProgress(null);
  }

  function leaveCalibration() {
    const attempt = calibrationAttempt.current;
    if (!attempt) return;
    attempt.abandoned = true;
    setNavigation(attempt.previousNavigation);
    finishCalibration();
    if (!attempt.pending) {
      setCalibrationBusy(true);
      void calibrateHands('cancel', attempt.display).catch(() => {}).finally(() => setCalibrationBusy(false));
    }
  }

  async function changeCalibration(action: 'begin' | 'capture' | 'cancel') {
    if (!operatorMode || calibrationBusy) return;
    const attempt = action === 'begin'
      ? { display: handDisplayKey(), previousNavigation: navigation, pending: false, abandoned: false }
      : calibrationAttempt.current;
    if (!attempt || attempt.pending) return;
    calibrationAttempt.current = attempt;
    attempt.pending = true;
    setCalibrationBusy(true);
    setCalibrationError(null);
    try {
      const next = await calibrateHands(action, attempt.display);
      if (attempt.abandoned) return;
      if (action === 'begin') {
        if (next.phase !== 'calibration') throw new Error(next.message ?? 'The hand camera is unavailable.');
        setCalibrationSaved(false);
        setNavigation('calibrated');
        const active = runtime.current;
        if (active) active.calibrating = true;
        setCalibrationProgress(next);
      } else if (action === 'cancel' || next.phase === 'ready') {
        if (action === 'capture') {
          setCalibrationSaved(true);
          changeNavigation('calibrated');
        } else setNavigation(attempt.previousNavigation);
        finishCalibration();
      } else if (next.phase === 'calibration') {
        setCalibrationProgress(current => ({ ...next, previewToken: current?.previewToken }));
      } else {
        setCalibrationError(next.message ?? 'The hand camera is unavailable.');
      }
    } catch (cause) {
      if (attempt.abandoned) return;
      if (action === 'cancel') {
        setNavigation(attempt.previousNavigation);
        finishCalibration();
      } else {
        if (action === 'begin') calibrationAttempt.current = null;
        setCalibrationError(cause instanceof Error ? cause.message : 'The camera did not respond.');
      }
    } finally {
      if (attempt.abandoned && action !== 'cancel') await calibrateHands('cancel', attempt.display).catch(() => {});
      attempt.pending = false;
      setCalibrationBusy(false);
    }
  }

  const graphicsIssue = rendererError ?? (contextLost ? 'The graphics connection was interrupted. Waiting for it to recover.' : null);
  const showStageActions = setupOpen || simulationMode;
  const stageMessage = graphicsIssue ? <div className="message error" role="alert"><TriangleAlert size={18} /><span>{graphicsIssue}</span>
    {showStageActions && <button className="button" type="button" onClick={() => window.location.reload()}>Reload</button>}
  </div> : loading ? <div className="message loading" role="status"><LoaderCircle size={18} /> <span>Loading {loading}</span>
    {showStageActions && <button className="button" type="button" onClick={() => importAbort.current?.abort()}>Cancel</button>}
  </div> : error ? <div className="message error" role="alert"><TriangleAlert size={18} /> <span>{error}</span>
    {showStageActions && <button type="button" className="icon-button close-message" aria-label="Dismiss import error" onClick={() => setError(null)}><X size={16} /></button>}
  </div> : null;
  const handsEnabled = simulationMode ? enabled : liveEnabled && liveStatus.phase === 'ready';
  const showPose = handsEnabled && !setupOpen && !calibrating && !contextLost && !!exhibit;
  const dualViewGrip = pose?.target === 'view' && pose.status === 'both';
  const resetProgress = showPose && !loading && !graphicsIssue ? Math.max(0, Math.min(1, pose?.resetProgress ?? 0)) : 0;
  const resetSeconds = Math.max(0.1, (1 - resetProgress) * RESET_HOLD_MS / 1000).toFixed(1);
  const immersiveHeadView = headPerspective.enabled && !setupOpen;
  const showCameraPreview = liveEnabled && !calibrating
    && (operatorMode || (visitorPreviewVisible && liveStatus.phase === 'ready'));
  const visibleHint = !immersiveHeadView && !resetProgress && !setupOpen && !calibrating && exhibit && !contextLost && (calibrationSaved || !hasExplored) && (!simulationMode || enabled);
  const hint = calibrationSaved ? 'Calibration saved. Point at the model to try it.'
    : handsEnabled ? visitorCue(pose, simulationMode) : 'Drag to look around. Tap a part to inspect it.';

  return <main className={`explorer${setupOpen ? operatorMode ? ' is-operator' : ' is-setup-open' : ''}`} onKeyDown={event => {
    if (event.key !== 'Escape' || event.defaultPrevented) return;
    if (calibrating) void changeCalibration('cancel');
    else if (simulatorSetupOpen) closeSetup();
  }}>
    {(!operatorMode || calibrating) && <h1 className="sr-only">Spatial CAD Explorer</h1>}
    <section ref={viewport} className="exhibit" tabIndex={simulationMode ? 0 : undefined} aria-label="Interactive CAD exhibit" aria-describedby={simulationMode ? 'visitor-cue' : undefined}>
      <div className="stage-light" />
      <canvas ref={canvas} className="model" tabIndex={simulationMode ? undefined : 0} aria-label="3D CAD assembly" aria-describedby={simulationMode ? undefined : 'canvas-help'} />
      {!simulationMode && <p id="canvas-help" className="sr-only">Use the arrow keys to orbit, plus and minus to zoom, Enter to select the part in the center, Escape to clear selection, or Home to reset the view.</p>}
      {simulationMode && !setupOpen && <button className="setup-trigger" type="button" onClick={() => setSimulatorSetupOpen(true)}><Settings size={22} /> Setup</button>}
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
      {showCameraPreview && <CameraPreview className="navigation-preview" display={liveStatus.display ?? handDisplayKey()} />}
      {!setupOpen && !calibrating && (stageMessage || !exhibit) && <div className="stage-message">
        {stageMessage}
        {!exhibit && !stageMessage && <div className="empty-stage">
          <Box size={35} /><p>{simulationMode ? 'Open Setup to load an assembly.' : 'Preparing the exhibit…'}</p>
        </div>}
      </div>}
      <HandCursors visible={showPose} controls={cursorControls} />
      {!setupOpen && !calibrating && (simulationMode || focusPart || pose?.target === 'view') && <footer className="exhibit-footer">
        {pose?.target === 'view'
          ? <div className="part-caption">{dualViewGrip ? <Move size={16} /> : <Orbit size={16} />}<span>{dualViewGrip ? 'Pan · Zoom' : 'Exhibit view'}</span></div>
          : focusPart && <div className="part-caption" aria-live="polite"><Box size={16} /><span>{displayName(focusPart)}</span></div>}
        {simulationMode && <span className="input-status" data-active={enabled && pose?.status !== 'lost' && !rendererError}><i />{enabled ? 'Simulated hands' : 'Simulation paused'}</span>}
      </footer>}
      {calibrating && calibrationProgress && <CalibrationPanel step={calibrationProgress.step ?? 0} message={calibrationError ?? calibrationProgress.message ?? null} busy={calibrationBusy}
        captureReady={captureReady} captureHint={captureHint} previewDisplay={calibrationProgress.display} previewToken={calibrationProgress.previewToken}
        onCapture={() => void changeCalibration('capture')} onCancel={() => void changeCalibration('cancel')} />}
    </section>

    {setupOpen && <aside id="exhibit-setup" className="setup-panel" aria-labelledby="setup-title">
      <div className="setup-head">
        <div>
          {operatorMode ? <h1 id="setup-title" ref={operatorTitle} tabIndex={-1}>Operator settings</h1> : <h2 id="setup-title">Exhibit setup</h2>}
          <p>Choose an assembly and prepare the exhibit.</p>
        </div>
        {operatorMode ? <a className="button exhibit-link" href="/" onClick={openExhibit}>Open exhibit</a>
          : <button ref={setupClose} className="button icon-button" type="button" aria-label="Close setup" onClick={closeSetup}><X size={20} /></button>}
      </div>
      <div className="setup-scroll" onDragOver={event => { if (event.dataTransfer.types.includes('Files')) event.preventDefault(); }} onDrop={onDrop}>
        {operatorMode && <HeadPerspectivePanel controls={headPerspective} unavailable={!!graphicsIssue || !exhibit} />}
        {operatorMode && <section className="panel" aria-label="Live hand camera">
          <div className="panel-heading"><h2>Hand camera</h2>
            <div className="hand-camera-actions">
              <a className="button" href="/diagnostics" target="_blank" rel="noopener">Diagnostics<span className="sr-only"> (opens in a new tab)</span></a>
              <button className="button" type="button" onClick={toggleLive}>{liveEnabled ? 'Disconnect' : 'Connect'}</button>
            </div>
          </div>
          <p className="setup-hint" role="status">{!liveEnabled ? 'Touch, mouse, and keyboard controls are ready.'
            : liveStatus.phase === 'ready' ? navigation === 'fuzzy' ? 'Fuzzy navigation is ready. Open the exhibit and move an open hand into view.' : 'Connected. Open the exhibit to use hand gestures.'
            : liveStatus.phase === 'calibration' ? liveStatus.message ?? 'This display needs a reach profile. Choose fuzzy navigation or calibrate below.'
            : liveStatus.phase === 'connecting' ? 'Connecting to the local hand camera.' : liveStatus.message ?? 'Camera unavailable. Start the local camera service and check the USB connection. Retrying automatically.'}</p>
          {liveEnabled && <label className="navigation-choice">Hand navigation
            <select value={navigation} onChange={event => changeNavigation(event.target.value as HandNavigation)}>
              <option value="fuzzy">Fuzzy · no calibration</option>
              <option value="calibrated">Calibrated reach</option>
            </select>
          </label>}
          <label className="preview-choice">
            <input type="checkbox" checked={visitorPreviewVisible} onChange={event => changePreview(event.target.checked)} />
            Show hand camera preview in the exhibit
          </label>
          {liveEnabled && (liveStatus.phase === 'ready' || liveStatus.phase === 'calibration') && <button className="button camera-calibrate" type="button" disabled={calibrationBusy || !!graphicsIssue} onClick={() => void changeCalibration('begin')}>{navigation === 'fuzzy' ? 'Optional reach calibration' : 'Calibrate reach'}</button>}
          {calibrationError && <p className="setup-hint" role="alert">{calibrationError}</p>}
          {calibrationSaved && <p className="setup-hint" role="status">Reach calibration saved.</p>}
          {liveEnabled && <DepthRangePanel />}
          <CameraSettings navigation={navigation} performance={handPerformance} />
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
        <AssemblyPanel key={exhibit?.revision ?? 0} roots={roots} index={index} selected={selected}
          displayName={displayName} onExportNames={downloadPartNames} onImportNames={file => void receivePartNames(file)} namesMessage={namesMessage} onSelect={id => edit(scene => {
          scene.select(id);
          setSelected(id);
        })} />
        <PartPanel part={part} displayName={part && displayName(part)} onDisplayName={name => {
          const current = currentExhibit.current;
          if (part && current && current.revision === exhibit?.revision) savePartNames(current, { [part.id]: name });
        }} details={details} canUndo={scene?.canUndo() ?? false} onColor={color => {
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
            <div><dt><kbd>V</kbd> <kbd>B</kbd></dt><dd>Hold one fist to orbit and zoom with depth. With both, move together to pan, or spread and close to zoom.</dd></div>
            <div><dt><kbd>Q</kbd> <kbd>E</kbd> <span>/</span> <kbd>[</kbd> <kbd>]</kbd></dt><dd>Adjust left / right depth</dd></div>
            <div><dt><kbd>Esc</kbd></dt><dd>Drop tracking</dd></div>
          </dl>
          <p className="setup-hint">Shift-drag to orbit with the pointer. On touchscreens, drag a part to move it or drag empty space to orbit. Add a second finger to resize a part or pan and zoom the view.</p>
        </section>}
      </div>
    </aside>}
  </main>;
}
