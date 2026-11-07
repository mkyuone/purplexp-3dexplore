import type { HandPose, PoseFrame } from './gestures';

export type HandNavigation = 'fuzzy' | 'calibrated';

export type HandCameraSelection = { source: 'rgb'; tracker: string; fps: number };
export type HandCameraSettings = HandCameraSelection & {
  sources: { value: string; label: string }[];
  trackers: { value: string; label: string; available: boolean; reason?: string }[];
  frameRates: number[];
  applying: boolean;
  error?: string;
  diagnostics?: {
    poseFps?: number; processingMs?: number; frameAgeMs?: number;
    estimatedHands?: number; depthHands?: number; trackedHands?: number;
  };
};

export type LiveHandStatus = {
  phase: 'connecting' | 'ready' | 'offline' | 'calibration';
  display?: string;
  step?: number;
  message?: string;
  captureReady?: boolean;
  captureHint?: string;
  previewToken?: string;
};

const poseUrl = import.meta.env?.VITE_HAND_POSE_URL || 'http://127.0.0.1:4182/poses';

function readCameraSettings(value: unknown): HandCameraSettings {
  if (!value || typeof value !== 'object') throw new Error('The local hand camera returned invalid settings.');
  const settings = value as HandCameraSettings;
  const choice = (item: { value?: unknown; label?: unknown }) => item && typeof item.value === 'string' && typeof item.label === 'string';
  if (settings.source !== 'rgb' || typeof settings.tracker !== 'string'
    || !Number.isSafeInteger(settings.fps) || settings.fps <= 0 || typeof settings.applying !== 'boolean'
    || !Array.isArray(settings.sources) || !settings.sources.every(choice)
    || !Array.isArray(settings.trackers) || !settings.trackers.every(item => choice(item) && typeof item.available === 'boolean')
    || !Array.isArray(settings.frameRates) || !settings.frameRates.every(fps => Number.isSafeInteger(fps) && fps > 0)
    || !settings.sources.some(item => item.value === settings.source)
    || !settings.trackers.some(item => item.value === settings.tracker)
    || !settings.frameRates.includes(settings.fps)) {
    throw new Error('The local hand camera returned invalid settings.');
  }
  const metrics = settings.diagnostics;
  const finiteMetric = (value: unknown) => typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : undefined;
  const handCount = (value: unknown) => typeof value === 'number' && Number.isSafeInteger(value) && value >= 0 ? value : undefined;
  return {
    source: settings.source, tracker: settings.tracker, fps: settings.fps,
    sources: settings.sources.map(({ value, label }) => ({ value, label })),
    trackers: settings.trackers.map(({ value, label, available, reason }) => ({
      value, label, available, reason: typeof reason === 'string' ? reason : undefined,
    })),
    frameRates: settings.frameRates,
    applying: settings.applying, error: typeof settings.error === 'string' ? settings.error : undefined,
    diagnostics: metrics && typeof metrics === 'object' ? {
      poseFps: finiteMetric(metrics.poseFps), processingMs: finiteMetric(metrics.processingMs), frameAgeMs: finiteMetric(metrics.frameAgeMs),
      estimatedHands: handCount(metrics.estimatedHands), depthHands: handCount(metrics.depthHands), trackedHands: handCount(metrics.trackedHands),
    } : undefined,
  };
}

async function cameraSettingsRequest(selection: HandCameraSelection | undefined, signal: AbortSignal): Promise<HandCameraSettings> {
  const response = await fetch(new URL('./settings', new URL(poseUrl, window.location.href)), {
    method: selection ? 'POST' : 'GET', cache: 'no-store',
    ...(selection ? {
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ source: 'rgb', tracker: selection.tracker, fps: selection.fps }),
    } : {}),
    signal: AbortSignal.any([signal, AbortSignal.timeout(2200)]),
  });
  const value = await response.json();
  if (!response.ok) throw new Error(typeof value?.error === 'string' ? value.error
    : typeof value?.message === 'string' ? value.message : 'The camera did not accept the settings request.');
  return readCameraSettings(value);
}

export function fetchHandCameraSettings(signal: AbortSignal) {
  return cameraSettingsRequest(undefined, signal);
}

export function applyHandCameraSettings(selection: HandCameraSelection, signal: AbortSignal) {
  return cameraSettingsRequest(selection, signal);
}

export function handDisplayKey() {
  const { innerWidth, innerHeight, devicePixelRatio, screen, screenX, screenY } = window;
  return `${innerWidth}x${innerHeight}@${Number(devicePixelRatio.toFixed(3))}:${screen.width}x${screen.height}:${Math.round(screenX)},${Math.round(screenY)}`;
}

function readStatus(
  value: { status?: unknown; step?: unknown; message?: unknown; captureReady?: unknown; captureHint?: unknown; previewToken?: unknown } | null,
  display: string,
): LiveHandStatus {
  const message = typeof value?.message === 'string' ? value.message : undefined;
  if (value?.status === 'calibration' && typeof value.step === 'number' && Number.isInteger(value.step) && value.step >= 0 && value.step < 5) {
    return {
      phase: 'calibration', display, step: value.step, message,
      captureReady: typeof value.captureReady === 'boolean' ? value.captureReady : undefined,
      captureHint: typeof value.captureHint === 'string' ? value.captureHint : undefined,
      previewToken: typeof value.previewToken === 'string' ? value.previewToken : undefined,
    };
  }
  if (value?.status === 'ready' || value?.status === 'offline') return { phase: value.status, display, message };
  throw new Error('The local hand camera returned an invalid response.');
}

export function fetchCalibrationPreview(display: string, token: string, signal: AbortSignal) {
  const url = new URL('./preview', new URL(poseUrl, window.location.href));
  url.searchParams.set('display', display);
  return fetch(url, { headers: { 'X-Calibration-Preview': token }, cache: 'no-store', signal });
}

export async function beginCameraPreview(display: string, signal: AbortSignal): Promise<string> {
  const response = await fetch(new URL('./preview', new URL(poseUrl, window.location.href)), {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ display, action: 'begin' }), signal,
  });
  const value = await response.json();
  if (!response.ok || typeof value?.previewToken !== 'string') throw new Error('Camera preview unavailable.');
  return value.previewToken;
}

export async function endCameraPreview(display: string, token: string) {
  await fetch(new URL('./preview', new URL(poseUrl, window.location.href)), {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ display, action: 'end', previewToken: token }),
    signal: AbortSignal.timeout(1200), keepalive: true,
  }).catch(() => {});
}

export async function calibrateHands(action: 'begin' | 'capture' | 'cancel', display: string): Promise<LiveHandStatus> {
  if (action !== 'cancel' && display !== handDisplayKey()) throw new Error('The display changed. Cancel and start calibration again in Setup.');
  const response = await fetch(new URL('./calibration', new URL(poseUrl, window.location.href)), {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ display, action }), signal: AbortSignal.timeout(1800),
  });
  const value = await response.json();
  if (!response.ok) throw new Error(typeof value?.message === 'string' ? value.message : 'The camera did not accept the calibration request.');
  return readStatus(value, display);
}

function validHand(value: unknown): value is HandPose {
  if (!value || typeof value !== 'object') return false;
  const hand = value as Partial<HandPose>;
  return typeof hand.id === 'string' && typeof hand.pinched === 'boolean'
    && (hand.fist === undefined || typeof hand.fist === 'boolean')
    && (hand.thumbsUp === undefined || typeof hand.thumbsUp === 'boolean') && typeof hand.depthValid === 'boolean'
    && [hand.x, hand.y, hand.depth].every(Number.isFinite);
}

export function connectLiveHands(options: {
  navigation: HandNavigation;
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
    if (next.phase === status.phase && next.display === status.display && next.step === status.step && next.message === status.message
      && next.captureReady === status.captureReady && next.captureHint === status.captureHint) return;
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
      url.searchParams.set('mode', options.navigation);
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
      // The bridge is still connected. Leave the controller's stall timer in
      // charge and keep polling at full speed for the next fresh pose.
      if (value?.status === 'recovering') return;
      const next = readStatus(value, currentDisplay);
      if (next.phase !== 'ready') {
        if (sourceId !== undefined) options.reset();
        sourceId = undefined;
        setStatus(next);
        delay = next.phase === 'calibration' ? 150 : 1000;
        return;
      }
      if (typeof value?.session !== 'string' || !Number.isSafeInteger(value.sequence)
        || !Number.isFinite(value.ageMs) || value.ageMs < 0
        || !['tracking', 'lost', 'ambiguous'].includes(value.tracking)
        || !Array.isArray(value.hands) || value.hands.length > 2 || !value.hands.every(validHand)) {
        throw new Error('Invalid hand camera frame.');
      }
      if (value.ageMs >= 220) return;
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
      setStatus({ phase: 'offline', display });
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
