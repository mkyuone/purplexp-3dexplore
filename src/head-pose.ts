export interface HeadFace {
  // Eye midpoint and face-box size in normalized, unmirrored webcam coordinates.
  x: number;
  y: number;
  width: number;
  height: number;
}

interface Point {
  x: number;
  y: number;
}

export interface HeadPoseSample extends Point {
  phase: 'centering' | 'tracking' | 'lost';
  animating: boolean;
  nextSampleAt: number | null;
}

interface CenterCandidate {
  first: HeadFace;
  center: Point;
  since: number;
  at: number;
  count: number;
}

interface Viewer {
  face: HeadFace;
  center: Point;
  offset: Point;
  seenAt: number;
  recoveringUntil: number;
}

const CENTER_MS = 250;
const CENTER_MAX_GAP_MS = 120;
const CENTER_RADIUS = 0.035;
const LOSS_HOLD_MS = 150;
const LOSS_EASE_MS = 250;
const VIEWER_LOCK_MS = 500;
const ZERO = { x: 0, y: 0 };

function validFace(face: HeadFace): boolean {
  return face.x >= 0 && face.x <= 1 && face.y >= 0 && face.y <= 1
    && face.width > 0 && face.width <= 1 && face.height > 0 && face.height <= 1;
}

function distance(a: Point, b: Point): number {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function shapeDifference(a: HeadFace, b: HeadFace): number {
  return Math.max(Math.abs(Math.log(a.width / b.width)), Math.abs(Math.log(a.height / b.height)));
}

function matchViewer(faces: readonly HeadFace[], previous: HeadFace, elapsed: number): HeadFace | undefined {
  const radius = Math.min(0.12, 0.045 + elapsed * 0.00055);
  let closest: HeadFace | undefined;
  let best = Infinity;
  let second = Infinity;
  for (const face of faces) {
    const travel = distance(face, previous);
    const shape = shapeDifference(face, previous);
    if (travel > radius || shape > 0.55) continue;
    const cost = travel + shape * 0.045;
    if (cost < best) {
      second = best;
      best = cost;
      closest = face;
    } else {
      second = Math.min(second, cost);
    }
  }
  // With two similarly plausible matches, wait for the current viewer to separate.
  return second - best > 0.018 ? closest : undefined;
}

function boundedOffset(displacement: number, limit: number): number {
  const travel = Math.max(0, Math.abs(displacement) - 0.0015);
  return travel === 0 ? 0 : Math.sign(displacement) * Math.min(limit, 0.95 * travel);
}

function visibleOffset(viewer: Viewer, age: number): Point {
  if (age <= LOSS_HOLD_MS) return viewer.offset;
  const t = Math.min(1, (age - LOSS_HOLD_MS) / LOSS_EASE_MS);
  if (t === 1) return ZERO;
  const remaining = 1 - t * t * (3 - 2 * t);
  return { x: viewer.offset.x * remaining, y: viewer.offset.y * remaining };
}

export class HeadPoseTracker {
  private candidate: CenterCandidate | null = null;
  private viewer: Viewer | null = null;
  private lastFrameAt = -Infinity;
  private seenViewer = false;
  private detected = false;

  receive(faces: readonly HeadFace[], now: number): void {
    if (!Number.isFinite(now) || now <= this.lastFrameAt) return;
    this.lastFrameAt = now;
    const valid = faces.filter(validFace);
    const viewer = this.viewer;
    if (viewer && now - viewer.seenAt < VIEWER_LOCK_MS) {
      this.updateViewer(viewer, valid, now);
      return;
    }

    this.viewer = null;
    this.detected = false;
    if (valid.length === 1) this.findCenter(valid[0], now);
    else this.candidate = null;
  }

  sample(now: number): HeadPoseSample {
    const viewer = this.viewer;
    if (!viewer) {
      const phase = this.candidate || !this.seenViewer ? 'centering' : 'lost';
      return { ...ZERO, phase, animating: false, nextSampleAt: null };
    }
    const age = Math.max(0, now - viewer.seenAt);
    const held = age < LOSS_HOLD_MS;
    const phase = this.detected && held ? 'tracking' : 'lost';
    const offset = visibleOffset(viewer, age);
    return {
      ...offset,
      phase,
      animating: !held && (offset.x !== 0 || offset.y !== 0),
      nextSampleAt: held ? viewer.seenAt + LOSS_HOLD_MS : null,
    };
  }

  recenter(now = this.lastFrameAt): void {
    if (this.viewer && this.detected && now - this.viewer.seenAt < LOSS_HOLD_MS) {
      this.viewer.center = this.viewer.face;
      this.viewer.offset = ZERO;
      this.viewer.recoveringUntil = 0;
      return;
    }
    this.viewer = null;
    this.candidate = null;
    this.seenViewer = false;
    this.detected = false;
  }

  reset(): void {
    this.viewer = null;
    this.candidate = null;
    this.lastFrameAt = -Infinity;
    this.seenViewer = false;
    this.detected = false;
  }

  private findCenter(face: HeadFace, now: number): void {
    const candidate = this.candidate;
    if (!candidate || now - candidate.at > CENTER_MAX_GAP_MS
      || distance(face, candidate.first) > CENTER_RADIUS || shapeDifference(face, candidate.first) > 0.35) {
      this.candidate = { first: face, center: { x: face.x, y: face.y }, since: now, at: now, count: 1 };
      return;
    }

    candidate.at = now;
    candidate.count += 1;
    candidate.center.x += (face.x - candidate.center.x) / candidate.count;
    candidate.center.y += (face.y - candidate.center.y) / candidate.count;
    if (now - candidate.since < CENTER_MS || candidate.count < 5) return;

    this.viewer = { face, center: candidate.center, offset: ZERO, seenAt: now, recoveringUntil: 0 };
    this.candidate = null;
    this.seenViewer = true;
    this.detected = true;
  }

  private updateViewer(viewer: Viewer, faces: readonly HeadFace[], now: number): void {
    const elapsed = now - viewer.seenAt;
    const face = matchViewer(faces, viewer.face, elapsed);
    this.detected = face !== undefined;
    if (!face) return;

    if (elapsed >= LOSS_HOLD_MS) {
      // Recover from the pose already on screen, including a partially eased view.
      viewer.offset = visibleOffset(viewer, elapsed);
      viewer.recoveringUntil = now + 120;
    } else {
      const target = {
        x: boundedOffset(viewer.center.x - face.x, 0.18),
        y: boundedOffset(viewer.center.y - face.y, 0.12),
      };
      const smoothingMs = now < viewer.recoveringUntil || distance(viewer.offset, target) <= 0.012 ? 55 : 20;
      const alpha = 1 - Math.exp(-elapsed / smoothingMs);
      const x = viewer.offset.x + (target.x - viewer.offset.x) * alpha;
      const y = viewer.offset.y + (target.y - viewer.offset.y) * alpha;
      viewer.offset = {
        x: Math.abs(x - target.x) < 0.00025 ? target.x : x,
        y: Math.abs(y - target.y) < 0.00025 ? target.y : y,
      };
    }
    viewer.face = face;
    viewer.seenAt = now;
  }
}
