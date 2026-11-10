import { memo, useEffect, useState, useSyncExternalStore } from 'react';
import { handCameraUrl } from './live-hands';
import './hand-diagnostics.css';

const gripNames = {
  open: 'Open',
  pinch: 'Pinching',
  'pinch pending': 'Pinch pending',
  fist: 'Fist',
  'thumbs up': 'Thumbs up',
  untracked: 'Detected, untracked',
} as const;
const sides = ['Right', 'Left'] as const;
const pollInterval = 200;
const statusInterval = 100;
const staleFrameAge = 1100;

type Side = typeof sides[number];
type Stream = 'rgb' | 'ir1' | 'ir2';
type HandReading = {
  side: Side;
  grip: keyof typeof gripNames;
  depthMm: number | null;
  gapMm?: number | null;
};
type DiagnosticsStatus = {
  status: 'ready' | 'offline' | 'waiting';
  message?: string;
  infraredCount?: number;
  infraredMessage?: string;
  unassignedHands?: number;
  resetSuppressed?: boolean;
  hands: HandReading[];
};

function subscribeToVisibility(update: () => void) {
  document.addEventListener('visibilitychange', update);
  return () => document.removeEventListener('visibilitychange', update);
}

function pageIsVisible() {
  return !document.hidden;
}

function pollWhileVisible(read: (signal: AbortSignal) => Promise<number>) {
  const polling = new AbortController();
  let timer: ReturnType<typeof setTimeout>;

  async function poll() {
    if (polling.signal.aborted || document.hidden) return;
    const delay = await read(polling.signal);
    if (!polling.signal.aborted && !document.hidden) timer = setTimeout(poll, delay);
  }

  void poll();
  return () => {
    polling.abort();
    clearTimeout(timer);
  };
}

function diagnosticsFetch(url: URL, signal: AbortSignal) {
  return fetch(url, { mode: 'cors', cache: 'no-store', signal: AbortSignal.any([signal, AbortSignal.timeout(1500)]) });
}

function useDiagnosticsStatus(visible: boolean) {
  const [status, setStatus] = useState<DiagnosticsStatus | null>(null);

  useEffect(() => {
    if (!visible) return;
    setStatus(null);
    const url = handCameraUrl('./diagnostics');

    return pollWhileVisible(async signal => {
      try {
        const response = await diagnosticsFetch(url, signal);
        if (!response.ok) throw new Error('Diagnostics request failed');
        const reading = await response.json() as DiagnosticsStatus;
        if (!reading || !['ready', 'waiting', 'offline'].includes(reading.status) || !Array.isArray(reading.hands)) {
          throw new Error('Diagnostics response was invalid');
        }
        if (!signal.aborted) setStatus(reading);
        return reading.status === 'offline' ? 900 : statusInterval;
      } catch {
        if (!signal.aborted) {
          setStatus({ status: 'offline', message: 'Cannot reach the local camera bridge.', hands: [] });
        }
        return 900;
      }
    });
  }, [visible]);

  return status;
}

const CameraTile = memo(function CameraTile({
  stream, title, alt, enabled, placeholder, unavailable,
}: {
  stream: Stream;
  title: string;
  alt: string;
  enabled: boolean;
  placeholder: string;
  unavailable?: string;
}) {
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [stalled, setStalled] = useState(false);

  useEffect(() => {
    setImageUrl(null);
    setStalled(false);
    if (!enabled) return;

    const url = handCameraUrl('./diagnostics/frame');
    url.searchParams.set('stream', stream);
    let currentUrl: string | null = null;
    let pendingUrl: string | null = null;
    let lastFrameAt = performance.now();

    const stop = pollWhileVisible(async signal => {
      try {
        const response = await diagnosticsFetch(url, signal);
        if (response.status === 200 && response.headers.get('Content-Type')?.startsWith('image/jpeg')) {
          const frame = await response.blob();
          if (signal.aborted) return pollInterval;
          pendingUrl = URL.createObjectURL(frame);
          const image = new Image();
          image.src = pendingUrl;
          await image.decode();
          if (signal.aborted) return pollInterval;

          if (currentUrl) URL.revokeObjectURL(currentUrl);
          currentUrl = pendingUrl;
          pendingUrl = null;
          setImageUrl(currentUrl);
          setStalled(false);
          lastFrameAt = performance.now();
          return pollInterval;
        }
      } catch {
        // Keep the latest decoded frame during a short capture or network gap.
      } finally {
        if (pendingUrl) URL.revokeObjectURL(pendingUrl);
        pendingUrl = null;
      }

      if (!signal.aborted && performance.now() - lastFrameAt >= staleFrameAge) {
        if (currentUrl) URL.revokeObjectURL(currentUrl);
        currentUrl = null;
        setImageUrl(null);
        setStalled(true);
        return 500;
      }
      return pollInterval;
    });

    return () => {
      stop();
      if (pendingUrl) URL.revokeObjectURL(pendingUrl);
      if (currentUrl) URL.revokeObjectURL(currentUrl);
      pendingUrl = null;
    };
  }, [enabled, stream]);

  return <section className="diagnostics-tile" aria-labelledby={`diagnostics-${stream}`}>
    <h2 id={`diagnostics-${stream}`}>{title}</h2>
    <div className={`diagnostics-feed diagnostics-feed-${stream}`}>
      {enabled && imageUrl
        ? <img src={imageUrl} alt={alt} />
        : <p>{enabled ? unavailable || (stalled ? 'Feed unavailable' : 'Waiting for frames…') : placeholder}</p>}
    </div>
  </section>;
});

function millimeters(value: number | null | undefined) {
  return typeof value === 'number' && Number.isFinite(value) && value >= 0 ? `${Math.round(value)} mm` : '—';
}

function HandStatus({ reading }: { reading: DiagnosticsStatus | null }) {
  return <section className="diagnostics-tile diagnostics-hands" aria-labelledby="diagnostics-hands-title">
    <h2 id="diagnostics-hands-title">Hand status</h2>
    <table>
      <thead><tr><th scope="col">Hand</th><th scope="col">Grip</th><th scope="col">Palm depth</th><th scope="col">Pinch gap</th></tr></thead>
      <tbody>{sides.map(side => {
        const hand = reading?.status === 'ready' ? reading.hands.find(item => item?.side === side) : undefined;
        const grip = hand && Object.hasOwn(gripNames, hand.grip) ? gripNames[hand.grip] : undefined;

        return <tr key={side}>
          <th scope="row"><span className="diagnostics-hand-dot" data-side={side} aria-hidden="true" />{side}</th>
          <td>{grip ?? (reading?.status === 'ready' ? 'Not detected' : '—')}</td>
          <td>{millimeters(hand?.depthMm)}</td>
          <td>{millimeters(hand?.gapMm)}</td>
        </tr>;
      })}</tbody>
    </table>
    <p className="diagnostics-hand-note">Pinch gap measures the distance between the thumb and index fingertips.</p>
    <p className="diagnostics-hand-note">
      Grip shows the current camera frame. To reset, hold a thumbs-up for 1.5 seconds with the hand camera connected and the <a href="/">exhibit</a> visible.
    </p>
    {reading?.resetSuppressed === true && <p className="diagnostics-hand-note" role="status">
      Thumbs-up reset is paused for depth setup. Show an open hand after setup to enable it again.
    </p>}
    {!!reading?.unassignedHands && <p className="diagnostics-hand-note">
      {reading.unassignedHands === 1 ? 'One detected hand has an uncertain side label.' : `${reading.unassignedHands} detected hands have uncertain side labels.`}
    </p>}
  </section>;
}

export default function HandDiagnostics() {
  const visible = useSyncExternalStore(subscribeToVisibility, pageIsVisible);
  const reading = useDiagnosticsStatus(visible);
  const enabled = visible && reading?.status !== 'offline';
  const placeholder = visible ? 'Camera offline' : 'Preview paused';
  const hasSecondInfrared = reading?.infraredCount !== 1;
  const secondInfraredPlaceholder = !enabled ? placeholder
    : hasSecondInfrared ? 'Waiting for frames…' : 'This camera has one infrared sensor.';
  const phase = !visible ? 'paused' : reading?.status ?? 'waiting';
  const status = !visible ? 'Previews paused' : reading?.message || (
    phase === 'ready' ? 'Camera connected' : phase === 'offline' ? 'Camera offline' : 'Connecting to the camera…'
  );

  return <main className="hand-diagnostics">
    <header className="diagnostics-header">
      <div>
        <h1>Hand diagnostics</h1>
        <p className="diagnostics-connection" data-phase={phase} role="status">{status}</p>
      </div>
      <a className="button exhibit-link" href="/operator">Operator settings</a>
    </header>
    <div className="diagnostics-grid">
      <CameraTile stream="rgb" title="RGB + MediaPipe" alt="Live RGB view with MediaPipe hand landmarks"
        enabled={enabled} placeholder={placeholder} />
      <HandStatus reading={visible ? reading : null} />
      <CameraTile stream="ir1" title="Infrared 1" alt="Live preview from infrared sensor 1"
        enabled={enabled} placeholder={placeholder} unavailable={reading?.infraredMessage} />
      <CameraTile stream="ir2" title="Infrared 2" alt="Live preview from infrared sensor 2"
        enabled={enabled && reading?.status === 'ready' && hasSecondInfrared}
        placeholder={secondInfraredPlaceholder} unavailable={reading?.infraredMessage} />
    </div>
  </main>;
}
