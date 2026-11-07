export type HeadCameraFace = {
  x: number;
  y: number;
  width: number;
  height: number;
  score: number;
};

export type HeadCameraFrame = { at: number; faces: readonly HeadCameraFace[]; inferenceMs: number };
export type HeadCameraDevice = { id: string; label: string };
export type HeadCameraDiagnostics = { captureFps: number; poseFps: number; inferenceMs: number };
export type HeadCameraState = {
  phase: 'off' | 'starting' | 'ready' | 'suspended' | 'error';
  devices: readonly HeadCameraDevice[];
  deviceId?: string;
  message?: string;
};

export type HeadCameraRequest = { at: number; bitmap: ImageBitmap };
export type HeadCameraReply = { type: 'ready' } | { type: 'frame'; frame: HeadCameraFrame } | { type: 'error'; message: string };

type Callbacks = {
  onFrame: (frame: HeadCameraFrame) => void;
  onState: (state: HeadCameraState) => void;
  onStream: (stream: MediaStream | null) => void;
  onDiagnostics?: (stats: HeadCameraDiagnostics) => void;
};

type Session = {
  video: HTMLVideoElement;
  stream?: MediaStream;
  worker?: Worker;
  frameCallback?: number;
  timeout?: number;
};

const otherSensor = /real\s*sense|\bd435i?\b|infrared|\bir\b|\bdepth\b|\bintegrated i\b/i;
const integrated = /integrated|built[ -]?in|facetime|front facing/i;

async function webcamList(media: MediaDevices) {
  const devices = await media.enumerateDevices().catch(() => []);
  return devices.filter(({ kind, deviceId, label }) => kind === 'videoinput' && deviceId && !otherSensor.test(label))
    .map(({ deviceId, label }, index) => ({ id: deviceId, label: label || `Webcam ${index + 1}` }))
    .sort((a, b) => Number(integrated.test(b.label)) - Number(integrated.test(a.label)));
}

function cameraError(error: unknown) {
  switch (error instanceof DOMException ? error.name : '') {
    case 'NotAllowedError': return 'Allow webcam access in the browser, then start the experiment again.';
    case 'NotFoundError':
    case 'OverconstrainedError': return 'The selected webcam is unavailable. Connect or choose an RGB webcam.';
    case 'NotReadableError':
    case 'AbortError': return 'The browser could not open the webcam. Check camera access and whether it is in use, then try again.';
    default: return error instanceof Error ? error.message : 'The webcam could not start.';
  }
}

export function createHeadCamera(callbacks: Callbacks) {
  const media = navigator.mediaDevices;
  const events = new AbortController();
  let state: HeadCameraState = { phase: 'off', devices: [] };
  let enabled = false;
  let requestedDevice: string | undefined;
  let session: Session | null = null;
  let disposed = false;

  function publish(update: Partial<HeadCameraState>) {
    state = { ...state, ...update };
    if (!disposed) callbacks.onState(state);
  }

  function release() {
    const active = session;
    session = null;
    if (!active) return;
    window.clearTimeout(active.timeout);
    if (active.frameCallback !== undefined) active.video.cancelVideoFrameCallback(active.frameCallback);
    active.worker?.terminate();
    active.video.pause();
    active.video.srcObject = null;
    active.video.remove();
    active.stream?.getTracks().forEach(track => track.stop());
    if (!disposed) callbacks.onStream(null);
  }

  function fail(active: Session, message: string) {
    if (session !== active) return;
    enabled = false;
    release();
    publish({ phase: 'error', message });
  }

  async function refreshDevices() {
    const devices = await webcamList(media);
    if (disposed) return;
    if (devices.length === state.devices.length && devices.every((device, index) => (
      device.id === state.devices[index].id && device.label === state.devices[index].label
    ))) return;
    publish({ devices });
  }

  function connectWorker(active: Session, worker: Worker) {
    const { video } = active;
    let busy = false;
    let captureCount = 0;
    let poseCount = 0;
    let inferenceTotal = 0;
    let intervalStart = 0;
    let receivedFrame = false;

    function takeFrame(at: number) {
      if (session !== active) return;
      active.frameCallback = video.requestVideoFrameCallback(takeFrame);
      intervalStart ||= at;
      captureCount++;
      const elapsed = at - intervalStart;
      if (elapsed >= 500) {
        callbacks.onDiagnostics?.({
          captureFps: 1000 * captureCount / elapsed,
          poseFps: 1000 * poseCount / elapsed,
          inferenceMs: poseCount ? inferenceTotal / poseCount : 0,
        });
        intervalStart = at;
        captureCount = poseCount = inferenceTotal = 0;
      }
      if (busy || !video.videoWidth || !video.videoHeight) return;
      busy = true;
      active.timeout = window.setTimeout(() => fail(active, 'The head tracker stopped responding. Start the experiment again.'), receivedFrame ? 2500 : 8000);
      const resizeWidth = Math.min(256, video.videoWidth);
      const resizeHeight = Math.max(1, Math.round(video.videoHeight * resizeWidth / video.videoWidth));
      // BlazeFace takes 128×128 input. Reduce transfer size while preserving the source aspect ratio.
      void createImageBitmap(video, { resizeWidth, resizeHeight, resizeQuality: 'low' }).then(bitmap => {
        if (session !== active) {
          bitmap.close();
          return;
        }
        try {
          worker.postMessage({ at, bitmap } satisfies HeadCameraRequest, [bitmap]);
        } catch (error) {
          bitmap.close();
          throw error;
        }
      }).catch(() => fail(active, 'The webcam frame could not be read. Start the experiment again.'));
    }

    active.timeout = window.setTimeout(() => fail(active, 'The local head tracker could not load. Start the experiment again.'), 15000);
    worker.onerror = () => fail(active, 'The local head tracker could not run in this browser.');
    worker.onmessage = ({ data }: MessageEvent<HeadCameraReply>) => {
      if (session !== active) return;
      window.clearTimeout(active.timeout);
      if (data.type === 'error') {
        fail(active, data.message);
      } else if (data.type === 'ready') {
        publish({ phase: 'ready' });
        if (session === active) active.frameCallback = video.requestVideoFrameCallback(takeFrame);
      } else {
        receivedFrame = true;
        busy = false;
        poseCount++;
        inferenceTotal += data.frame.inferenceMs;
        callbacks.onFrame(data.frame);
      }
    };
  }

  async function open(active: Session) {
    let startingVideo = false;
    try {
      const devices = await webcamList(media);
      if (session !== active) return;
      const deviceId = requestedDevice || devices[0]?.id;
      const stream = await media.getUserMedia({ video: {
        width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 60 },
        ...(deviceId ? { deviceId: { exact: deviceId } } : { facingMode: { ideal: 'user' } }),
      }, audio: false });
      if (session !== active) {
        stream.getTracks().forEach(track => track.stop());
        return;
      }
      active.stream = stream;
      const track = stream.getVideoTracks()[0];
      if (!track) throw new Error('The selected camera did not provide a video stream.');
      if (otherSensor.test(track.label)) throw new Error('Choose an RGB webcam separate from the RealSense hand camera.');
      const connectedId = track.getSettings().deviceId || deviceId;
      track.addEventListener('ended', () => fail(active, 'The webcam disconnected. Reconnect it or choose another webcam.'), { once: true });
      void refreshDevices();
      publish({ deviceId: connectedId });
      if (session !== active) return;
      // Chrome can abort playback on a detached video element.
      document.body.append(active.video);
      active.video.srcObject = stream;
      callbacks.onStream(stream);
      startingVideo = true;
      active.timeout = window.setTimeout(() => fail(active, 'No webcam video arrived. Check whether another tab or app is using it.'), 10000);
      await active.video.play();
      if (session !== active) return;
      window.clearTimeout(active.timeout);
      active.worker = new Worker(new URL('./head-camera.worker.ts', import.meta.url), { type: 'module', name: 'exhibit-head-tracker' });
      connectWorker(active, active.worker);
    } catch (error) {
      if (session !== active) return;
      console.warn(startingVideo ? 'Head webcam playback failed' : 'Head webcam access failed', error);
      fail(active, startingVideo ? 'The webcam opened, but video playback failed. Try starting the webcam again.' : cameraError(error));
      if (!disposed) void refreshDevices();
    }
  }

  function start(deviceId?: string) {
    if (disposed) return;
    release();
    enabled = true;
    requestedDevice = deviceId;
    if (document.hidden) {
      publish({ phase: 'suspended', message: undefined, deviceId });
      return;
    }
    const video = document.createElement('video');
    if (!media?.getUserMedia || typeof video.requestVideoFrameCallback !== 'function' || typeof createImageBitmap !== 'function') {
      enabled = false;
      publish({ phase: 'error', message: 'This browser cannot use the webcam here. Open the exhibit in Chrome on localhost or HTTPS.' });
      return;
    }
    video.muted = true;
    video.playsInline = true;
    video.hidden = true;
    const active = { video };
    session = active;
    publish({ phase: 'starting', message: undefined, deviceId });
    void open(active);
  }

  function stop() {
    if (disposed) return;
    enabled = false;
    release();
    publish({ phase: 'off', message: undefined });
  }

  document.addEventListener('visibilitychange', () => {
    if (!enabled) return;
    if (document.hidden) {
      release();
      publish({ phase: 'suspended' });
    } else {
      start(state.deviceId || requestedDevice);
    }
  }, { signal: events.signal });
  media?.addEventListener('devicechange', refreshDevices, { signal: events.signal });
  if (media) void refreshDevices();

  return { start, stop, dispose: () => {
    disposed = true;
    enabled = false;
    events.abort();
    release();
  } };
}
