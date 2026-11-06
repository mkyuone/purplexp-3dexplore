import { Box, Crosshair, Expand, Hand, LoaderCircle, Orbit, Settings, TriangleAlert, Upload, X } from 'lucide-react';
import { useEffect, useMemo, useRef, useState, type ChangeEvent, type DragEvent } from 'react';
import { importStep, type CadAssembly } from './cad';
import { CadScene } from './scene';
import { GestureController, type GestureState } from './gestures';
import { HandSimulator } from './simulator';
import { AssemblyPanel, indexOccurrences, PartPanel } from './panels';

const sampleName = 'suspension.step';
const emptyRoots: CadAssembly['roots'] = [];
type Runtime = { scene: CadScene; gestures: GestureController; simulator: HandSimulator };
type Exhibit = { name: string; assembly: CadAssembly };

function visitorCue(state: GestureState | null) {
  if (state?.status === 'lost') return 'Move your hands back into view and start with an open hand.';
  if (state?.status === 'both') return state.target === 'part' ? 'Move together, spread to resize, or turn to rotate.' : 'Spread your hands to zoom in. Bring them together to zoom out.';
  if (state?.status === 'holding') return state.target === 'part' ? 'Move the part, or pinch at the ring with your other hand.' : 'Move to look around. Make a second fist at the ring to zoom.';
  return 'Pinch a part to move it. Make a fist to look around.';
}

export function App() {
  const viewport = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const setup = useRef<HTMLDialogElement>(null);
  const picker = useRef<HTMLInputElement>(null);
  const runtime = useRef<Runtime | null>(null);
  const importAbort = useRef<AbortController | null>(null);
  const [exhibit, setExhibit] = useState<Exhibit | null>(null);
  const [loading, setLoading] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [rendererError, setRendererError] = useState<string | null>(null);
  const [setupOpen, setSetupOpen] = useState(false);
  const [enabled, setEnabled] = useState(true);
  const [selected, setSelected] = useState<string | null>(null);
  const [pose, setPose] = useState<GestureState | null>(null);
  const [, setRevision] = useState(0);
  const roots = exhibit?.assembly.roots ?? emptyRoots;
  const index = useMemo(() => indexOccurrences(roots), [roots]);
  const part = selected === null ? undefined : index.byId.get(selected);
  const focusId = pose?.held ?? pose?.hovered;
  const focusPart = focusId ? index.byId.get(focusId) : undefined;
  const scene = runtime.current?.scene;
  const details = setupOpen && part ? scene?.describe(part.id) : null;

  async function load(name: string, read: (signal: AbortSignal) => Promise<ArrayBuffer>) {
    importAbort.current?.abort();
    const controller = new AbortController();
    importAbort.current = controller;
    setLoading(name);
    setError(null);
    try {
      const data = await read(controller.signal);
      controller.signal.throwIfAborted();
      const assembly = await importStep(data, controller.signal);
      const active = runtime.current;
      if (controller.signal.aborted || !active) return;
      active.gestures.cancel();
      active.scene.load(assembly);
      setExhibit({ name, assembly });
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

  function loadSample() {
    return load(sampleName, async signal => {
      const response = await fetch(`${import.meta.env.BASE_URL}samples/${sampleName}`, { signal });
      if (!response.ok) throw new Error('The bundled sample could not be opened.');
      return response.arrayBuffer();
    });
  }

  useEffect(() => {
    if (!viewport.current || !canvas.current) return;
    let scene: CadScene;
    try {
      scene = new CadScene(canvas.current);
    } catch {
      setRendererError('A WebGL 2 context could not be created. Open the explorer in a browser with hardware acceleration enabled.');
      return;
    }
    const gestures = new GestureController(scene, state => {
      runtime.current?.simulator.setJoinTarget(state.join);
      setSelected(scene.getSelected());
      setPose(state);
    });
    const simulator = new HandSimulator(viewport.current, frame => gestures.receive(frame), (x, y) => scene.pick(x, y) ? 'part' : 'view');
    runtime.current = { scene, gestures, simulator };
    simulator.setEnabled(true);
    void loadSample();
    return () => {
      importAbort.current?.abort();
      importAbort.current = null;
      simulator.dispose();
      gestures.dispose();
      scene.dispose();
      runtime.current = null;
    };
  }, []);

  function receiveFile(file?: File) {
    if (!file) return;
    if (!/\.st(?:e)?p$/i.test(file.name)) {
      setError('Choose a STEP assembly or part with a .step or .stp extension.');
      return;
    }
    void load(file.name, () => file.arrayBuffer());
  }

  function onFile(event: ChangeEvent<HTMLInputElement>) {
    receiveFile(event.currentTarget.files?.[0]);
    event.currentTarget.value = '';
  }

  function onDrop(event: DragEvent<HTMLElement>) {
    if (!event.dataTransfer.types.includes('Files')) return;
    event.preventDefault();
    if (!rendererError) receiveFile(event.dataTransfer.files[0]);
  }

  function edit(change: (scene: CadScene) => void) {
    const active = runtime.current;
    if (!active) return;
    active.gestures.cancel();
    change(active.scene);
    setRevision(value => value + 1);
  }

  function openSetup() {
    runtime.current?.simulator.setEnabled(false);
    setup.current?.showModal();
    setSetupOpen(true);
  }

  function closeSetup() {
    setSetupOpen(false);
    const active = runtime.current;
    active?.gestures.cancel();
    active?.simulator.setEnabled(enabled);
    viewport.current?.focus({ preventScroll: true });
  }

  const stageMessage = loading ? <div className="message loading" role="status"><LoaderCircle size={18} /> <span>Importing {loading}</span></div>
    : error || rendererError ? <div className="message error" role="alert"><TriangleAlert size={18} /> <span>{rendererError ?? error}</span>
      {!rendererError && <button type="button" className="icon-button close-message" aria-label="Dismiss import error" onClick={() => setError(null)}><X size={16} /></button>}
    </div> : null;

  return <main className="explorer">
    <section ref={viewport} className="exhibit" tabIndex={0} aria-label="Interactive CAD exhibit" aria-describedby="visitor-cue">
      <div className="stage-light" />
      <canvas ref={canvas} className="model" aria-label="3D CAD assembly" />
      <header className="masthead">
        <div className="brand"><Box aria-hidden="true" /><span>Spatial CAD Explorer</span></div>
        <button className="setup-trigger" type="button" aria-haspopup="dialog" onClick={openSetup}><Settings size={22} /> Setup</button>
      </header>
      <div className="intro">
        <h1>Reach in and explore</h1>
        <p id="visitor-cue" aria-live="polite">{enabled ? visitorCue(pose) : 'Open Setup to enable simulated hands.'}</p>
      </div>
      {!setupOpen && stageMessage && <div className="stage-message">{stageMessage}</div>}
      {!exhibit && !loading && !rendererError && <div className="empty-stage">
        <Box size={35} /><p>Open Setup to load an assembly.</p>
      </div>}
      {enabled && !setupOpen && exhibit && <div className="hands-layer" aria-hidden="true">
        {pose?.hands.map(hand => <div key={hand.id}
          className={`hand-cursor${hand.id === 'right' ? ' second' : ''}${hand.pinched ? ' is-pinched' : ''}${hand.fist ? ' is-view-grip' : ''}`}
          style={{ left: `${hand.x * 100}%`, top: `${hand.y * 100}%`, transform: `translate(-50%, -50%) scale(${Math.max(0.65, Math.min(1.45, 1 + hand.depth))})` }}>
          {hand.fist ? <Orbit size={25} /> : <Hand size={25} />}
        </div>)}
        {pose?.join && <div className="join-target" style={{ left: `${pose.join.x * 100}%`, top: `${pose.join.y * 100}%` }}>
          <span className="join-ring">{pose.target === 'view' ? <Orbit size={24} /> : <Crosshair size={24} />}</span>
          <span className="join-label">{pose.target === 'view' ? 'Make a fist here' : 'Pinch here'}</span>
        </div>}
      </div>}
      <footer className="exhibit-footer">
        {enabled && !setupOpen && (pose?.target === 'view'
          ? <div className="part-caption"><Orbit size={16} /><span>Exhibit view</span></div>
          : focusPart && <div className="part-caption"><Box size={16} /><span>{focusPart.name}</span></div>)}
        <span className="input-status" data-active={enabled && pose?.status !== 'lost' && !rendererError}><i />{enabled ? 'Simulated hands' : 'Simulation paused'}</span>
      </footer>
    </section>

    <dialog ref={setup} className="setup-panel" aria-labelledby="setup-title" onClose={closeSetup}>
      <div className="setup-head">
        <div><h2 id="setup-title">Exhibit setup</h2><p>Choose an assembly and prepare the exhibit.</p></div>
        <button className="button icon-button" type="button" aria-label="Close setup" onClick={() => setup.current?.close()}><X size={20} /></button>
      </div>
      {setupOpen && <div className="setup-scroll" onDragOver={event => { if (event.dataTransfer.types.includes('Files')) event.preventDefault(); }} onDrop={onDrop}>
        <section className="panel file-panel" aria-label="STEP import">
          <h2>Model</h2>
          <p className="file-name" title={exhibit?.name}>{exhibit?.name ?? 'No model loaded'} <span>mm · Z up</span></p>
          <div className="file-actions">
            <button className="button button-primary" type="button" onClick={() => picker.current?.click()} disabled={!!rendererError}><Upload size={16} /> Import STEP</button>
            <button className="button" type="button" onClick={() => void loadSample()} disabled={!!rendererError}>Load sample</button>
          </div>
          <input ref={picker} type="file" aria-label="STEP file" accept=".step,.stp" onChange={onFile} hidden />
          {stageMessage}
        </section>
        <AssemblyPanel key={exhibit?.name ?? ''} roots={roots} index={index} selected={selected} onSelect={id => edit(scene => {
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
        <section className="panel sim-panel" aria-label="Simulator controls">
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
        </section>
      </div>}
    </dialog>
  </main>;
}
