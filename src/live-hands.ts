import type { HandPose, PoseFrame } from './gestures';

export type HandNavigation = 'fuzzy' | 'calibrated';

export type HandCameraSettings = {
  fps: number;
  frameRates: number[];
  applying: boolean;
  error?: string;
  diagnostics?: {
    poseFps?: number; processingMs?: number; frameAgeMs?: number;
    estimatedHands?: number; depthHands?: number; trackedHands?: number;
    queueMs?: number; inferenceMs?: number; alignmentMs?: number; fusionMs?: number;
    processingP95Ms?: number; captureDroppedFrames?: number; depthFrameAvailable?: boolean;
    frameAgeSource?: 'hostArrival' | 'dequeue';
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
const POSE_POLL_MS = 1000 / 60;
const POSE_RETRY_WINDOW_MS = 500;

function readCameraSettings(value: unknown): HandCameraSettings {
  if (!value || typeof value !== 'object') throw new Error('The local hand camera returned invalid settings.');
  const settings = value as HandCameraSettings;
  const isCaptureRate = (fps: unknown) => fps === 30 || fps === 60;
  if (!isCaptureRate(settings.fps) || typeof settings.applying !== 'boolean'
    || !Array.isArray(settings.frameRates) || !settings.frameRates.every(isCaptureRate)
    || !settings.frameRates.includes(settings.fps)) {
    throw new Error('The local hand camera returned invalid settings.');
  }
  const metrics = settings.diagnostics;
  const finiteMetric = (value: unknown) => typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : undefined;
  const handCount = (value: unknown) => typeof value === 'number' && Number.isSafeInteger(value) && value >= 0 ? value : undefined;
  return {
    fps: settings.fps,
    frameRates: settings.frameRates,
    applying: settings.applying, error: typeof settings.error === 'string' ? settings.error : undefined,
    diagnostics: metrics && typeof metrics === 'object' ? {
      poseFps: finiteMetric(metrics.poseFps), processingMs: finiteMetric(metrics.processingMs), frameAgeMs: finiteMetric(metrics.frameAgeMs),
      estimatedHands: handCount(metrics.estimatedHands), depthHands: handCount(metrics.depthHands), trackedHands: handCount(metrics.trackedHands),
      queueMs: finiteMetric(metrics.queueMs), inferenceMs: finiteMetric(metrics.inferenceMs),
      alignmentMs: finiteMetric(metrics.alignmentMs), fusionMs: finiteMetric(metrics.fusionMs),
      processingP95Ms: finiteMetric(metrics.processingP95Ms), captureDroppedFrames: handCount(metrics.captureDroppedFrames),
      depthFrameAvailable: typeof metrics.depthFrameAvailable === 'boolean' ? metrics.depthFrameAvailable : undefined,
      frameAgeSource: metrics.frameAgeSource === 'hostArrival' || metrics.frameAgeSource === 'dequeue' ? metrics.frameAgeSource : undefined,
    } : undefined,
  };
}

async function cameraSettingsRequest(fps: number | undefined, signal: AbortSignal): Promise<HandCameraSettings> {
  const updating = fps !== undefined;
  const response = await fetch(new URL('./settings', new URL(poseUrl, window.location.href)), {
    method: updating ? 'POST' : 'GET', cache: 'no-store',
    ...(updating ? {
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ source: 'rgb', tracker: 'mediapipe', fps }),
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

export function applyHandCameraSettings(fps: number, signal: AbortSignal) {
  return cameraSettingsRequest(fps, signal);
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
  receive: (frame: PoseFrame, receivedAt: number) => void;
  reset: () => void;
  status: (status: LiveHandStatus) => void;
}) {
  const session = new AbortController();
  let timer: ReturnType<typeof setTimeout>;
  let sourceId: string | undefined;
  let waitForPose = false;
  let lastSequence = -1;
  let lastCapturedAt = -Infinity;
  let lastContactAt = -Infinity;
  let status: LiveHandStatus = { phase: 'connecting' };
  let display: string | undefined;

  function setStatus(next: LiveHandStatus) {
    if (next.phase === status.phase && next.display === status.display && next.step === status.step && next.message === status.message
      && next.captureReady === status.captureReady && next.captureHint === status.captureHint) return;
    status = next;
    options.status(next);
  }

  async function poll() {
    let interval = POSE_POLL_MS;
    const startedAt = performance.now();
    try {
      const currentDisplay = handDisplayKey();
      const url = new URL(poseUrl, window.location.href);
      url.searchParams.set('display', currentDisplay);
      url.searchParams.set('mode', options.navigation);
      if (waitForPose && sourceId && currentDisplay === display) {
        url.searchParams.set('session', sourceId);
        url.searchParams.set('after', String(lastSequence));
      }
      const response = await fetch(url, {
        cache: 'no-store', signal: AbortSignal.any([session.signal, AbortSignal.timeout(sourceId === undefined ? 800 : 160)]),
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
      if (value?.status === 'recovering') {
        lastContactAt = performance.now();
        return;
      }
      const next = readStatus(value, currentDisplay);
      if (next.phase !== 'ready') {
        if (sourceId !== undefined) options.reset();
        sourceId = undefined;
        setStatus(next);
        interval = next.phase === 'calibration' ? 150 : 1000;
        return;
      }
      if (typeof value?.session !== 'string' || !Number.isSafeInteger(value.sequence) || value.sequence < 0
        || !Number.isFinite(value.ageMs) || value.ageMs < 0
        || !['tracking', 'lost', 'ambiguous'].includes(value.tracking)
        || !Array.isArray(value.hands) || value.hands.length > 2 || !value.hands.every(validHand)) {
        throw new Error('Invalid hand camera frame.');
      }
      const receivedAt = performance.now();
      lastContactAt = receivedAt;
      if (value.ageMs >= 220) return;
      if (sourceId !== value.session) {
        sourceId = value.session;
        lastSequence = -1;
        lastCapturedAt = -Infinity;
        options.reset();
      }
      setStatus(next);
      if (value.sequence > lastSequence) {
        lastSequence = value.sequence;
        waitForPose = value.waitForPose === true;
        if (waitForPose) interval = 0;
        // Remove the time waiting for a pose on the bridge before estimating
        // return transit. Older bridges respond immediately and omit bridgeMs.
        const bridgeMs = Number.isFinite(value.bridgeMs) && value.bridgeMs >= 0 ? value.bridgeMs : 0;
        const transitMs = Math.max(0, receivedAt - startedAt - bridgeMs) / 2;
        const capturedAt = receivedAt - transitMs - value.ageMs;
        lastCapturedAt = Math.max(lastCapturedAt, capturedAt);
        options.receive({ sequence: value.sequence, capturedAt: lastCapturedAt, tracking: value.tracking, hands: value.hands }, receivedAt);
      }
    } catch {
      if (session.signal.aborted) return;
      // One failed poll can recover in the same camera session. The gesture
      // controller pauses stale motion while we wait for a fresh observation.
      if (sourceId !== undefined && performance.now() - lastContactAt < POSE_RETRY_WINDOW_MS) return;
      if (sourceId !== undefined) options.reset();
      sourceId = undefined;
      setStatus({ phase: 'offline', display });
      interval = 1500;
    } finally {
      if (!session.signal.aborted) timer = setTimeout(poll, Math.max(0, startedAt + interval - performance.now()));
    }
  }

  void poll();
  return () => {
    session.abort();
    clearTimeout(timer);
  };
}
