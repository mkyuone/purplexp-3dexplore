import { useEffect, useRef, useState } from 'react';
import { CameraPreview } from './camera-preview';
import { CALIBRATION_INSET_X, CALIBRATION_INSET_Y } from './reach';
import './calibration.css';

const left = CALIBRATION_INSET_X * 100;
const top = CALIBRATION_INSET_Y * 100;
const captureDelaySeconds = 3;
const targets = [
  { x: left, y: top, name: 'upper left' },
  { x: 100 - left, y: top, name: 'upper right' },
  { x: 100 - left, y: 100 - top, name: 'lower right' },
  { x: left, y: 100 - top, name: 'lower left' },
  { x: 50, y: 50, name: 'center' },
];

type CalibrationPanelProps = {
  step: number;
  message: string | null;
  captureReady?: boolean;
  captureHint?: string;
  previewDisplay?: string;
  previewToken?: string;
  busy: boolean;
  onCapture: () => void;
  onCancel: () => void;
};

export function CalibrationPanel({ step, message, captureReady, captureHint, previewDisplay, previewToken, busy, onCapture, onCancel }: CalibrationPanelProps) {
  const capture = useRef<HTMLButtonElement>(null);
  const cancel = useRef<HTMLButtonElement>(null);
  const timer = useRef<number | undefined>(undefined);
  const [countdown, setCountdown] = useState<number | null>(null);
  const target = targets[step];

  useEffect(() => {
    capture.current?.focus({ preventScroll: true });
    return () => window.clearInterval(timer.current);
  }, []);

  function stopCountdown() {
    window.clearInterval(timer.current);
    timer.current = undefined;
    setCountdown(null);
  }

  function startCountdown() {
    if (busy || timer.current !== undefined) return;
    let seconds = captureDelaySeconds;
    setCountdown(seconds);
    timer.current = window.setInterval(() => {
      seconds--;
      if (seconds > 0) {
        setCountdown(seconds);
      } else {
        stopCountdown();
        onCapture();
      }
    }, 1000);
  }

  function cancelCalibration() {
    if (busy) return;
    stopCountdown();
    onCancel();
  }

  return <section className="calibration-overlay" role="dialog" aria-modal="true" aria-labelledby="calibration-title" aria-describedby="calibration-instructions" onKeyDown={event => {
    if (event.key === 'Escape') {
      event.preventDefault();
      event.stopPropagation();
      cancelCalibration();
    } else if (event.key === 'Tab') {
      const first = event.shiftKey ? cancel.current : capture.current;
      const last = event.shiftKey ? capture.current : cancel.current;
      if (event.target === last) {
        event.preventDefault();
        first?.focus();
      }
    }
  }}>
    {previewDisplay && previewToken && <CameraPreview className="calibration-preview" key={previewToken} display={previewDisplay} token={previewToken} below={target.y > 50} />}
    <div className="calibration-target" data-ready={captureReady === true} style={{ left: `${target.x}%`, top: `${target.y}%` }} aria-hidden="true" />
    <footer className="calibration-prompt" data-target-position={target.y > 50 ? 'lower' : target.y === 50 ? 'center' : 'upper'}>
      <div>
        <h2 id="calibration-title">Hand camera calibration</h2>
        <p id="calibration-instructions">Move your whole hand toward the ring with your palm facing the camera. Select Capture, then use the {captureDelaySeconds}-second countdown to hold your fingertip steady.</p>
        <p className="calibration-status" role="status" aria-live="polite" aria-atomic="true">
          Target {step + 1} of {targets.length} · {target.name}{message ? ` · ${message}` : ''}
        </p>
        <p className="calibration-readiness" data-ready={captureReady === true} role="status" aria-live="polite">
          {captureReady && countdown !== null ? 'Steady. Keep your hand in place.'
            : captureHint ?? (captureReady ? 'Ready to capture.' : 'Waiting for a steady hand.')}
        </p>
      </div>
      <div className="calibration-actions">
        <button ref={capture} className="button button-primary" type="button" aria-disabled={busy || countdown !== null} onClick={startCountdown}>
          {busy ? 'Capturing…' : countdown !== null ? `Capture in ${countdown}` : 'Capture'}
        </button>
        <button ref={cancel} className="button" type="button" aria-disabled={busy} onClick={cancelCalibration}>Cancel</button>
        <span className="sr-only" role="status">{countdown !== null ? `Capturing in ${countdown} ${countdown === 1 ? 'second' : 'seconds'}.` : ''}</span>
      </div>
    </footer>
  </section>;
}
