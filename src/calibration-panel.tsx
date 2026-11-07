import { useEffect, useRef } from 'react';
import { CALIBRATION_INSET_X, CALIBRATION_INSET_Y } from './reach';
import './calibration.css';

const left = CALIBRATION_INSET_X * 100;
const top = CALIBRATION_INSET_Y * 100;
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
  busy: boolean;
  onCapture: () => void;
  onCancel: () => void;
};

export function CalibrationPanel({ step, message, busy, onCapture, onCancel }: CalibrationPanelProps) {
  const capture = useRef<HTMLButtonElement>(null);
  const cancel = useRef<HTMLButtonElement>(null);
  const target = targets[step];

  useEffect(() => {
    capture.current?.focus({ preventScroll: true });
  }, []);

  return <section className="calibration-overlay" role="dialog" aria-modal="true" aria-labelledby="calibration-title" aria-describedby="calibration-instructions" onKeyDown={event => {
    if (event.key === 'Escape') {
      event.preventDefault();
      event.stopPropagation();
      if (!busy) onCancel();
    } else if (event.key === 'Tab') {
      const first = event.shiftKey ? cancel.current : capture.current;
      const last = event.shiftKey ? capture.current : cancel.current;
      if (event.target === last) {
        event.preventDefault();
        first?.focus();
      }
    }
  }}>
    <div className="calibration-target" style={{ left: `${target.x}%`, top: `${target.y}%` }} aria-hidden="true" />
    <footer className="calibration-prompt" data-target-position={target.y > 50 ? 'lower' : target.y === 50 ? 'center' : 'upper'}>
      <div>
        <h2 id="calibration-title">Hand camera calibration</h2>
        <p id="calibration-instructions">Aim one index finger at the ring from a comfortable position. Hold steady, then select Capture.</p>
        <p className="calibration-status" role="status" aria-live="polite" aria-atomic="true">
          Target {step + 1} of {targets.length} · {target.name}{message ? ` · ${message}` : ''}
        </p>
      </div>
      <div className="calibration-actions">
        <button ref={capture} className="button button-primary" type="button" aria-disabled={busy} onClick={() => {
          if (!busy) onCapture();
        }}>{busy ? 'Capturing…' : 'Capture'}</button>
        <button ref={cancel} className="button" type="button" aria-disabled={busy} onClick={() => {
          if (!busy) onCancel();
        }}>Cancel</button>
      </div>
    </footer>
  </section>;
}
