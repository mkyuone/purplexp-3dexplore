import { useEffect, useRef, useState } from 'react';
import { requestDepthRange } from './live-hands';

type Status = Awaited<ReturnType<typeof requestDepthRange>>;
type Action = 'begin' | 'cancel' | 'reset';
const centimeters = (meters: number) => Math.round(meters * 100);

export function DepthRangePanel() {
  const [status, setStatus] = useState<Status | null>(null);
  const [busy, setBusy] = useState(false);
  const [pollError, setPollError] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const activeRequest = useRef<AbortController | null>(null);
  const ownedSession = useRef<string | null>(null);
  const capturing = status?.step !== undefined;

  useEffect(() => {
    if (busy) return;
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    async function poll() {
      try {
        if (document.hidden) return;
        const next = await requestDepthRange(controller.signal);
        if (controller.signal.aborted || activeRequest.current) return;
        if (!next.step || next.session !== ownedSession.current) ownedSession.current = null;
        if (next.confirmed) setNotice('Depth range saved.');
        setStatus(next);
        setPollError(null);
      } catch (cause) {
        if (!controller.signal.aborted && !activeRequest.current) {
          setPollError(cause instanceof Error && cause.name === 'Error' ? cause.message : 'Waiting for the hand camera service.');
        }
      } finally {
        if (!controller.signal.aborted) timer = setTimeout(poll, capturing ? 220 : 1500);
      }
    }
    void poll();
    return () => {
      controller.abort();
      clearTimeout(timer);
    };
  }, [busy, capturing]);

  useEffect(() => () => {
    activeRequest.current?.abort();
    const session = ownedSession.current;
    if (session) void requestDepthRange(AbortSignal.timeout(1400), { session, action: 'cancel' }).catch(() => {});
  }, []);

  async function update(action: Action) {
    const session = status?.session;
    if (!session || activeRequest.current) return;
    const controller = new AbortController();
    activeRequest.current = controller;
    setBusy(true);
    setNotice(null);
    setActionError(null);
    try {
      const next = await requestDepthRange(controller.signal, { action, session });
      if (controller.signal.aborted) return;
      ownedSession.current = action === 'begin' && next.step ? session : null;
      setStatus(next);
      setPollError(null);
      if (action === 'reset') setNotice('Automatic depth range restored.');
    } catch (cause) {
      if (!controller.signal.aborted) setActionError(cause instanceof Error && cause.name === 'Error' ? cause.message : 'Could not reach the hand camera service.');
    } finally {
      activeRequest.current = null;
      if (!controller.signal.aborted) setBusy(false);
    }
  }

  const issue = actionError ?? pollError;
  const range = status?.range;
  return <section className="depth-range-settings" aria-label="Hand depth range">
    <h3>Hand depth range</h3>
    {capturing ? <>
      <p className="depth-range-step">{status.step === 'near' ? '1 of 2 · Closest' : '2 of 2 · Farthest'}</p>
      <p className="camera-option-note">
        {status.step === 'near' ? 'Hold one hand at your closest comfortable distance.' : 'Open your hand again and move it to your farthest comfortable distance.'}{' '}
        Point your thumb up with your other fingers open. When the distance is steady, curl your fingers into a thumbs-up and hold it without moving your thumb tip.
      </p>
      <dl className="depth-range-readings">
        {status.nearM !== undefined && <div><dt>Closest captured</dt><dd>{centimeters(status.nearM)} cm</dd></div>}
        <div><dt>Thumb distance</dt><dd>{status.captureM === undefined ? '—' : `${centimeters(status.captureM)} cm`}</dd></div>
      </dl>
      <p className="camera-option-note depth-range-hint" data-ready={status.captureM !== undefined} role="status">
        {status.captureHint ?? (status.confirming ? 'Keep your thumbs-up steady.' : 'Waiting for a steady thumb.')}
      </p>
      <button className="button" type="button" disabled={busy} onClick={() => void update('cancel')}>Cancel</button>
    </> : <>
      <p className="camera-option-note">
        {range ? `Custom range: ${centimeters(range.nearM)}–${centimeters(range.farM)} cm from the camera.`
          : status?.session ? 'Using automatic depth range.' : 'Waiting for the hand camera.'}{' '}
        Set the comfortable range for push/pull zoom and moving parts in depth.
      </p>
      <div className="depth-range-actions">
        <button className="button" type="button" disabled={busy || !status?.session || !!pollError} onClick={() => void update('begin')}>
          {range ? 'Set new depth range' : 'Set depth range'}
        </button>
        {range && <button className="button" type="button" disabled={busy || !status?.session || !!pollError} onClick={() => void update('reset')}>Use automatic range</button>}
      </div>
      {(notice || status?.captureHint) && <p className="camera-option-note" role="status">{notice ?? status?.captureHint}</p>}
    </>}
    {issue && <p className="camera-settings-error" role="alert">{issue}</p>}
  </section>;
}
