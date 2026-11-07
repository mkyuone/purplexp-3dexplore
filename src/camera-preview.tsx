import { useEffect, useState } from 'react';
import { beginCameraPreview, endCameraPreview, fetchCalibrationPreview } from './live-hands';
import './camera-preview.css';

export function CameraPreview({ display, token, className = '', below = false }: {
  display: string;
  token?: string;
  className?: string;
  below?: boolean;
}) {
  const [imageUrl, setImageUrl] = useState<string | null>(null);

  useEffect(() => {
    const preview = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    let currentUrl: string | null = null;
    let currentToken = token;
    let lastFrameAt = performance.now();

    function showFrame(frame: Blob | null) {
      const previousUrl = currentUrl;
      currentUrl = frame ? URL.createObjectURL(frame) : null;
      setImageUrl(currentUrl);
      if (previousUrl) URL.revokeObjectURL(previousUrl);
    }

    async function poll() {
      if (document.hidden) {
        showFrame(null);
        timer = setTimeout(poll, 1000);
        return;
      }
      let frame: Blob | null = null;
      try {
        const signal = AbortSignal.any([preview.signal, AbortSignal.timeout(1200)]);
        if (!currentToken) {
          currentToken = await beginCameraPreview(display, signal);
          if (preview.signal.aborted) {
            void endCameraPreview(display, currentToken);
            return;
          }
          lastFrameAt = performance.now();
        }
        const response = await fetchCalibrationPreview(display, currentToken, signal);
        if (response.status === 200 && response.headers.get('Content-Type')?.startsWith('image/jpeg')) {
          frame = await response.blob();
          lastFrameAt = performance.now();
        }
      } catch {
        // The next poll can recover from a camera or bridge interruption.
      }
      if (preview.signal.aborted) return;
      showFrame(frame);
      if (!token && currentToken && performance.now() - lastFrameAt > 2500) {
        void endCameraPreview(display, currentToken);
        currentToken = undefined;
      }
      timer = setTimeout(poll, frame ? 180 : 350);
    }

    void poll();
    return () => {
      preview.abort();
      clearTimeout(timer);
      if (currentUrl) URL.revokeObjectURL(currentUrl);
      if (!token && currentToken) void endCameraPreview(display, currentToken);
    };
  }, [display, token]);

  return <figure className={`camera-preview ${className}`} data-below={below}>
    <div className="camera-preview-frame">
      {imageUrl ? <img src={imageUrl} alt="Mirrored camera view with hand detection markers" /> : <span>Waiting for camera preview</span>}
    </div>
    <figcaption>
      <span>Live camera · mirrored</span>
      <span className="preview-legend"><span>Depth found</span><span>Hand detected</span></span>
    </figcaption>
  </figure>;
}
