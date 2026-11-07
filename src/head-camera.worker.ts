/// <reference lib="webworker" />

import { FaceDetector } from '@mediapipe/tasks-vision';
import wasmLoaderUrl from '@mediapipe/tasks-vision/vision_wasm_module_internal.js?url&no-inline';
import wasmBinaryUrl from '@mediapipe/tasks-vision/vision_wasm_module_internal.wasm?url';
import modelUrl from './models/blaze-face-short-range.tflite?url';
import type { HeadCameraFace, HeadCameraReply, HeadCameraRequest } from './head-camera';

const scope = self as unknown as DedicatedWorkerGlobalScope;
const localUrl = (value: string) => new URL(value, scope.location.href);
const fetchAsset = scope.fetch.bind(scope);

// The detector uses same-origin assets. Keep its optional SDK metrics local too.
scope.fetch = (input, init) => {
  const url = localUrl(input instanceof Request ? input.url : String(input));
  if (url.origin !== scope.location.origin) return Promise.reject(new TypeError('Only local head tracker assets are allowed.'));
  return fetchAsset(input, { ...init, redirect: 'error' });
};

function send(reply: HeadCameraReply) {
  scope.postMessage(reply);
}

async function initialize() {
  try {
    const detector = await FaceDetector.createFromOptions({
      wasmLoaderPath: localUrl(wasmLoaderUrl).href,
      wasmBinaryPath: localUrl(wasmBinaryUrl).href,
    }, {
      baseOptions: { modelAssetPath: localUrl(modelUrl).href, delegate: 'CPU' },
      runningMode: 'VIDEO',
      minDetectionConfidence: 0.6,
    });

    scope.onmessage = ({ data: { at, bitmap } }: MessageEvent<HeadCameraRequest>) => {
      const start = performance.now();
      try {
        const result = detector.detectForVideo(bitmap, at);
        const faces: HeadCameraFace[] = [];
        for (const { keypoints, boundingBox, categories } of result.detections) {
          const [rightEye, leftEye] = keypoints;
          if (!rightEye || !leftEye || !boundingBox || boundingBox.width <= 0 || boundingBox.height <= 0) continue;
          faces.push({
            x: (rightEye.x + leftEye.x) / 2,
            y: (rightEye.y + leftEye.y) / 2,
            width: boundingBox.width / bitmap.width,
            height: boundingBox.height / bitmap.height,
            score: categories[0]?.score ?? 0,
          });
        }
        send({ type: 'frame', frame: { at, faces, inferenceMs: performance.now() - start } });
      } catch (error) {
        console.error('Head tracking failed', error);
        send({ type: 'error', message: 'Head tracking could not process the webcam image. Start the experiment again.' });
      } finally {
        bitmap.close();
      }
    };
    send({ type: 'ready' });
  } catch (error) {
    console.error('Head tracker initialization failed', error);
    send({ type: 'error', message: 'The local head tracker could not load. Reload the exhibit and try again.' });
  }
}

void initialize();
