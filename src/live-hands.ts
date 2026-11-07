import type { HandPose, PoseFrame } from './gestures';

export type LiveHandStatus = { phase: 'connecting' | 'ready' | 'offline' | 'calibration'; step?: number; message?: string };

const poseUrl = import.meta.env.VITE_HAND_POSE_URL || 'http://127.0.0.1:4182/poses';

export function handDisplayKey() {
  const { innerWidth, innerHeight, devicePixelRatio, screen, screenX, screenY } = window;
  return `${innerWidth}x${innerHeight}@${Number(devicePixelRatio.toFixed(3))}:${screen.width}x${screen.height}:${Math.round(screenX)},${Math.round(screenY)}`;
}

function readStatus(value: { status?: unknown; step?: unknown; message?: unknown } | null): LiveHandStatus {
  const message = typeof value?.message === 'string' ? value.message : undefined;
  if (value?.status === 'calibration' && typeof value.step === 'number' && Number.isInteger(value.step) && value.step >= 0 && value.step < 5) {
    return { phase: 'calibration', step: value.step, message };
  }
  if (value?.status === 'ready' || value?.status === 'offline') return { phase: value.status, message };
  throw new Error('The local hand camera returned an invalid response.');
}

export async function calibrateHands(action: 'begin' | 'capture' | 'cancel', display: string): Promise<LiveHandStatus> {
  if (action !== 'cancel' && display !== handDisplayKey()) throw new Error('The display changed. Cancel and start calibration again in Setup.');
  const response = await fetch(new URL('./calibration', new URL(poseUrl, window.location.href)), {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ display, action }), signal: AbortSignal.timeout(1800),
  });
  const value = await response.json();
  if (!response.ok) throw new Error(typeof value?.message === 'string' ? value.message : 'The camera did not accept the calibration request.');
  return readStatus(value);
}

function validHand(value: unknown): value is HandPose {
  if (!value || typeof value !== 'object') return false;
  const hand = value as Partial<HandPose>;
  return typeof hand.id === 'string' && typeof hand.pinched === 'boolean'
    && (hand.fist === undefined || typeof hand.fist === 'boolean') && typeof hand.depthValid === 'boolean'
    && [hand.x, hand.y, hand.depth].every(Number.isFinite);
}

export function connectLiveHands(options: {
  receive: (frame: PoseFrame) => void;
  reset: () => void;
  status: (status: LiveHandStatus) => void;
}) {
  const session = new AbortController();
  let timer: ReturnType<typeof setTimeout>;
  let sourceId: string | undefined;
  let lastSequence = -1;
  let status: LiveHandStatus = { phase: 'connecting' };
  let display: string | undefined;

  function setStatus(next: LiveHandStatus) {
    if (next.phase === status.phase && next.step === status.step && next.message === status.message) return;
    status = next;
    options.status(next);
  }

  async function poll() {
    let delay = 33;
    const startedAt = performance.now();
    try {
      const currentDisplay = handDisplayKey();
      const url = new URL(poseUrl, window.location.href);
      url.searchParams.set('display', currentDisplay);
      const response = await fetch(url, {
        cache: 'no-store', signal: AbortSignal.any([session.signal, AbortSignal.timeout(800)]),
      });
      if (!response.ok) throw new Error('Hand camera unavailable.');
      const value = await response.json();
      if (session.signal.aborted || currentDisplay !== handDisplayKey()) return;
      if (display !== currentDisplay) {
        display = currentDisplay;
        sourceId = undefined;
        options.reset();
      }
      const next = readStatus(value);
      if (next.phase !== 'ready') {
        if (sourceId !== undefined) options.reset();
        sourceId = undefined;
        setStatus(next);
        delay = next.phase === 'calibration' ? 150 : 1000;
        return;
      }
      if (typeof value?.session !== 'string' || !Number.isSafeInteger(value.sequence)
        || !Number.isFinite(value.ageMs) || value.ageMs < 0 || value.ageMs >= 220
        || !['tracking', 'lost', 'ambiguous'].includes(value.tracking)
        || !Array.isArray(value.hands) || value.hands.length > 2 || !value.hands.every(validHand)) {
        throw new Error('Invalid hand camera frame.');
      }
      if (sourceId !== value.session) {
        sourceId = value.session;
        lastSequence = -1;
        options.reset();
      }
      setStatus(next);
      if (value.sequence > lastSequence) {
        lastSequence = value.sequence;
        options.receive({ sequence: value.sequence, capturedAt: startedAt - value.ageMs, tracking: value.tracking, hands: value.hands });
      }
    } catch {
      if (session.signal.aborted) return;
      if (sourceId !== undefined) options.reset();
      sourceId = undefined;
      setStatus({ phase: 'offline' });
      delay = 1500;
    } finally {
      if (!session.signal.aborted) timer = setTimeout(poll, delay);
    }
  }

  void poll();
  return () => {
    session.abort();
    clearTimeout(timer);
  };
}
