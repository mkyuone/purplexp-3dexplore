import { useEffect, useRef, useState, type RefObject } from 'react';
import type { HeadCameraDiagnostics, HeadCameraState } from './head-camera';
import { HeadPoseTracker } from './head-pose';
import type { CadScene } from './scene';

const neutral = { x: 0, y: 0 };
const statusText = {
  off: 'Use a separate RGB webcam to look around the exhibit. Hand controls stay available.',
  starting: 'Starting the webcam and face tracker…',
  suspended: 'Webcam paused while this tab is hidden.',
  error: 'The webcam is unavailable. Choose another camera or stop and start again.',
} as const;

type TrackingPhase = 'centering' | 'tracking' | 'lost';
const trackingText: Record<TrackingPhase, string> = {
  centering: 'Face the webcam and hold still briefly to center the view.',
  tracking: 'Tracking. Close Setup, then move your head sideways or up and down.',
  lost: 'Face lost. Look toward the webcam to resume.',
};

type TrackingSession = { start: (device?: string) => void; recenter: () => void };
type RenderTiming = { median: number; p95: number };

export function useHeadPerspective(runtime: RefObject<{ scene: CadScene } | null>, setupOpen: boolean) {
  const session = useRef<TrackingSession | null>(null);
  const preferredDevice = useRef<string | undefined>(undefined);
  const setupVisible = useRef(setupOpen);
  const [enabled, setEnabled] = useState(false);
  const [camera, setCamera] = useState<HeadCameraState>({ phase: 'off', devices: [] });
  const [phase, setPhase] = useState<TrackingPhase>('centering');
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [diagnostics, setDiagnostics] = useState<HeadCameraDiagnostics | null>(null);
  const [timing, setTiming] = useState<RenderTiming | null>(null);

  useEffect(() => { setupVisible.current = setupOpen; }, [setupOpen]);

  useEffect(() => {
    const scene = runtime.current?.scene;
    if (!enabled || !scene) return;

    const tracker = new HeadPoseTracker();
    const latency: number[] = [];
    let disposed = false;
    let capture: ReturnType<typeof import('./head-camera').createHeadCamera> | undefined;
    let timeout: number | undefined;
    let animation: number | undefined;
    let lastPhase: TrackingPhase = 'centering';

    function cancelSample() {
      window.clearTimeout(timeout);
      if (animation !== undefined) window.cancelAnimationFrame(animation);
      timeout = undefined;
      animation = undefined;
    }

    function sample(now: number, capturedAt?: number) {
      cancelSample();
      if (disposed) return;
      const pose = tracker.sample(now);
      if (lastPhase !== pose.phase) {
        lastPhase = pose.phase;
        setPhase(pose.phase);
      }
      const measure = capturedAt === undefined || pose.phase !== 'tracking' ? undefined : (renderedAt: number) => {
        latency.push(renderedAt - capturedAt);
        if (latency.length > 120) latency.shift();
      };
      scene?.setViewerOffset(pose, measure);
      if (pose.animating) animation = window.requestAnimationFrame(time => sample(time));
      else if (pose.nextSampleAt !== null) timeout = window.setTimeout(
        () => sample(performance.now()), Math.max(1, pose.nextSampleAt - performance.now()),
      );
    }

    function resetTracking() {
      cancelSample();
      tracker.reset();
      lastPhase = 'centering';
      latency.length = 0;
      setPhase('centering');
      setDiagnostics(null);
      setTiming(null);
      scene?.setViewerOffset(neutral);
    }

    scene.setHeadPerspective(true);
    void import('./head-camera').then(({ createHeadCamera }) => {
      if (disposed) return;
      capture = createHeadCamera({
        onState: state => {
          if (disposed) return;
          setCamera(state);
          if (state.phase !== 'ready') resetTracking();
        },
        onStream: value => { if (!disposed) setStream(value); },
        onFrame: ({ faces, at }) => {
          const now = performance.now();
          if (disposed || now - at > 150) return;
          tracker.receive(faces, at);
          sample(now, at);
        },
        onDiagnostics: value => {
          if (disposed || !setupVisible.current) return;
          setDiagnostics(value);
          if (!latency.length) return;
          const ordered = [...latency].sort((left, right) => left - right);
          setTiming({
            median: ordered[Math.ceil(ordered.length * 0.5) - 1],
            p95: ordered[Math.ceil(ordered.length * 0.95) - 1],
          });
        },
      });
      session.current = {
        start: capture.start,
        recenter: () => {
          const now = performance.now();
          tracker.recenter(now);
          sample(now);
        },
      };
      capture.start(preferredDevice.current);
    }).catch(() => {
      if (disposed) return;
      setCamera(previous => ({ ...previous, phase: 'error', message: 'The face tracker could not load. Stop and start again to retry.' }));
    });

    return () => {
      disposed = true;
      cancelSample();
      capture?.dispose();
      session.current = null;
      scene.setHeadPerspective(false);
    };
  }, [enabled, runtime]);

  return {
    enabled, camera, phase, stream, diagnostics, timing,
    toggle() {
      setCamera(previous => ({ phase: enabled ? 'off' : 'starting', devices: previous.devices }));
      setStream(null);
      setDiagnostics(null);
      setTiming(null);
      setPhase('centering');
      setEnabled(!enabled);
    },
    selectDevice(id: string) {
      const deviceId = id || undefined;
      preferredDevice.current = deviceId;
      setCamera(previous => ({ ...previous, deviceId }));
      if (enabled) session.current?.start(deviceId);
    },
    recenter() { session.current?.recenter(); },
    selectedDevice: enabled ? camera.deviceId ?? preferredDevice.current : preferredDevice.current,
  };
}

export function HeadPerspectivePanel({ controls, unavailable }: {
  controls: ReturnType<typeof useHeadPerspective>;
  unavailable: boolean;
}) {
  const { enabled, camera, phase, stream, diagnostics, timing } = controls;
  const preview = useRef<HTMLVideoElement>(null);
  useEffect(() => {
    const video = preview.current;
    if (!video || !stream) return;
    video.srcObject = stream;
    return () => { video.srcObject = null; };
  }, [stream]);

  return <section className="panel" aria-labelledby="head-perspective-title">
    <div className="panel-heading">
      <h2 id="head-perspective-title">Head-coupled perspective</h2>
      <button className="button" type="button" disabled={!enabled && unavailable} onClick={controls.toggle}>
        {enabled ? 'Stop' : 'Start'}
      </button>
    </div>
    <p className="setup-hint" role={camera.phase === 'error' ? 'alert' : 'status'}>
      {camera.phase === 'ready' ? trackingText[phase] : camera.message ?? statusText[camera.phase]}
    </p>
    {(enabled || camera.devices.length > 0) && <div className="head-camera-actions">
      <label className="navigation-choice">Webcam
        <select value={controls.selectedDevice ?? ''} onChange={event => controls.selectDevice(event.target.value)}>
          <option value="">Automatic</option>
          {camera.devices.map(device => <option key={device.id} value={device.id}>{device.label}</option>)}
        </select>
      </label>
      <button className="button" type="button" disabled={camera.phase !== 'ready'} onClick={controls.recenter}>Recenter</button>
    </div>}
    {stream && <video ref={preview} className="head-camera-preview" autoPlay muted playsInline aria-label="Mirrored head camera preview" />}
    {diagnostics && <dl className="camera-diagnostics" aria-label="Head tracking diagnostics">
      <div><dt>Webcam</dt><dd>{diagnostics.captureFps.toFixed(1)} fps</dd></div>
      <div><dt>Pose rate</dt><dd>{diagnostics.poseFps.toFixed(1)} fps</dd></div>
      <div><dt>Inference</dt><dd>{diagnostics.inferenceMs.toFixed(1)} ms</dd></div>
      <div><dt>Frame to render</dt><dd>{timing ? `${timing.median.toFixed(0)} ms median · ${timing.p95.toFixed(0)} ms p95` : 'Move your head to measure'}</dd></div>
    </dl>}
  </section>;
}
