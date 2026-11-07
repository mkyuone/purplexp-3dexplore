import { Quaternion, Vector3 } from 'three'
import { CALIBRATION_INSET_X, CALIBRATION_INSET_Y } from './reach.ts'

export interface HandPose {
  readonly id: string
  readonly x: number
  readonly y: number
  // The source supplies a finite placeholder when depth is unavailable.
  readonly depth: number
  readonly depthValid: boolean
  readonly pinched: boolean
  readonly fist?: boolean
  readonly thumbsUp?: boolean
}

// A source timestamps frames against the receiver's monotonic clock. Positive depth
// is toward the viewer, in normalized viewport widths from the neutral plane.
export interface PoseFrame {
  readonly sequence: number
  readonly capturedAt: number
  readonly hands: readonly HandPose[]
  readonly tracking?: 'tracking' | 'lost' | 'ambiguous'
}

export type GestureStatus = 'ready' | 'aim' | 'holding' | 'both' | 'lost'
export interface JoinTarget { x: number; y: number; radiusX: number; radiusY: number; radiusPx: number }

export function isInsideJoin(target: JoinTarget, x: number, y: number) {
  return Math.hypot((x - target.x) / target.radiusX, (y - target.y) / target.radiusY) <= 1
}

export interface GestureState {
  status: GestureStatus
  hands: readonly HandPose[]
  activeHands: readonly string[]
  resetProgress: number
  hovered: string | null
  held: string | null
  target: 'part' | 'view' | null
  join: JoinTarget | null
}

export interface ObjectPose {
  position: Vector3
  rotation: Quaternion
  scale: number
}

interface GestureScene {
  pick(x: number, y: number): string | null
  select(id: string | null): void
  hover(ids: string[]): void
  getPose(id: string): ObjectPose | null
  setPose(id: string, pose: ObjectPose): void
  screenPoint(x: number, y: number, reference: Vector3, depth?: number): Vector3
  orbit(dx: number, dy: number): void
  zoom(factor: number): void
  commitGrip(id: string, startingPose: ObjectPose): void
}

interface PartGrip {
  id: string
  original: ObjectPose
  base: ObjectPose
  startFirst: Vector3
  startSecond: Vector3 | null
}

interface ActiveGrip {
  first: HandPose
  second: HandPose | null
  viewAnchor: { x: number; y: number; depth: number }
  joinSide: number
  joinVertical: number
  pairBlocked: boolean
  part: PartGrip | null
}

export const FRAME_MAX_AGE_MS = 240
export const RESET_HOLD_MS = 1500
const MIN_HAND_SEPARATION = 0.06
const RESUME_HAND_SEPARATION = 0.08
const TARGET_GRACE_MS = 200
const TARGET_GRACE_PX = 28
const TRACKING_GRACE_MS = 150
const CLOSED_GRIP_CONFIRM_MS = 100
const PAIR_RECOVERY_MS = 100
const VIEW_ORBIT_DEADZONE = 0.005
const VIEW_DEPTH_DEADZONE = 0.006
const VIEW_DEPTH_ZOOM_GAIN = 2

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value))
}

function isUsableHand(hand: HandPose) {
  return Number.isFinite(hand.depth)
    && hand.x >= 0 && hand.x <= 1 && hand.y >= 0 && hand.y <= 1
    && !(hand.pinched && hand.fist)
}

function hasDepth(first: HandPose, second: HandPose | null) {
  return first.depthValid && (second === null || second.depthValid)
}

function isNeutral(hand: HandPose) {
  return !hand.pinched && !hand.fist && !hand.thumbsUp
}

function matchesGrip(grip: ActiveGrip, hand: HandPose) {
  return grip.part ? hand.pinched : hand.fist === true
}

function sameHands(left: readonly HandPose[], right: readonly HandPose[]) {
  return left.length === right.length && left.every((hand, index) => {
    const other = right[index]
    return hand.id === other.id && hand.x === other.x && hand.y === other.y
      && hand.depth === other.depth && hand.depthValid === other.depthValid
      && hand.pinched === other.pinched && Boolean(hand.fist) === Boolean(other.fist)
      && Boolean(hand.thumbsUp) === Boolean(other.thumbsUp)
  })
}

function samePosition(left: HandPose, right: HandPose, useDepth: boolean) {
  return left.x === right.x && left.y === right.y && (!useDepth || left.depth === right.depth)
}

function crossedBetweenFrames(previous: Vector3, next: Vector3) {
  const travel = next.clone().sub(previous)
  const distance = travel.lengthSq()
  if (!distance) return false
  const closest = -previous.dot(travel) / distance
  return closest > 0 && closest < 1
    && travel.multiplyScalar(closest).add(previous).lengthSq() < MIN_HAND_SEPARATION ** 2
}

export class GestureController {
  private readonly scene: GestureScene
  private readonly onState: (state: GestureState) => void
  private readonly joinAnywhere: boolean
  private readonly liveTracking: boolean
  private readonly onReset?: () => void
  private viewportWidth = 1000
  private viewportHeight = 1000
  private grip: ActiveGrip | null = null
  private interruption: { capturedAt: number; receivedAt: number } | null = null
  private armed = new Set<string>()
  private lastSequence = -1
  private lastCapture = -Infinity
  private lastReceivedAt = -Infinity
  private timeout: ReturnType<typeof setTimeout> | undefined
  private recovering = false
  private recoverClosed = false
  private retryPinch: string | null = null
  private closedCandidate: { hand: HandPose; target: string | null; capturedAt: number } | null = null
  private hovered: string[] = []
  private hoverCandidates = new Map<string, { id: string; x: number; y: number; capturedAt: number }>()
  private resetCandidate: { id: string; capturedAt: number; elapsed: number } | null = null
  // Keep this latch across source loss and cancellation: missing frames are not
  // evidence that the visitor released a thumbs-up that already fired.
  private resetOwner: string | null = null
  private state: GestureState = {
    status: 'ready', hands: [], activeHands: [], resetProgress: 0, hovered: null, held: null, target: null, join: null,
  }

  constructor(scene: GestureScene, onState: (state: GestureState) => void, options: {
    joinAnywhere?: boolean
    liveTracking?: boolean
    onReset?: () => void
  } = {}) {
    this.scene = scene
    this.onState = onState
    this.joinAnywhere = options.joinAnywhere ?? false
    this.liveTracking = options.liveTracking ?? false
    this.onReset = options.onReset
    this.recoverClosed = this.liveTracking
  }

  setViewport(width: number, height: number) {
    if (!Number.isFinite(width) || width <= 0 || !Number.isFinite(height) || height <= 0) return
    if (width === this.viewportWidth && height === this.viewportHeight) return
    this.viewportWidth = width
    this.viewportHeight = height
    this.cancel()
  }

  // Call on source replacement, after detaching the previous stream's listener.
  resetStream(allowClosedRecovery = this.liveTracking) {
    this.lastSequence = -1
    this.lastCapture = -Infinity
    this.lastReceivedAt = -Infinity
    this.loseTracking(false)
    this.recoverClosed = this.liveTracking && allowClosedRecovery
  }

  cancel() {
    this.resetCandidate = null
    this.endGrip()
    this.armed.clear()
    this.hoverCandidates.clear()
    this.recoverClosed = false
    this.recovering = false
    this.updateState(this.state.hands)
  }

  dispose() {
    clearTimeout(this.timeout)
    this.timeout = undefined
    this.interruption = null
    this.closedCandidate = null
    this.retryPinch = null
    this.hoverCandidates.clear()
    this.resetCandidate = null
  }

  receive(frame: PoseFrame) {
    if (!Number.isSafeInteger(frame.sequence) || frame.sequence <= this.lastSequence
      || !Number.isFinite(frame.capturedAt) || frame.capturedAt < this.lastCapture) return
    const receivedAt = performance.now()
    const age = receivedAt - frame.capturedAt
    if (age < -20 || age >= FRAME_MAX_AGE_MS) {
      if (this.resetCandidate) {
        this.resetCandidate = null
        this.updateState(this.state.hands)
      }
      return
    }

    const stalled = this.state.hands.length && (receivedAt - this.lastReceivedAt >= FRAME_MAX_AGE_MS
      || frame.capturedAt - this.lastCapture >= FRAME_MAX_AGE_MS)
    const interruption = this.interruption
    const grace = this.liveTracking ? FRAME_MAX_AGE_MS : TRACKING_GRACE_MS
    if (stalled || (interruption && (receivedAt - interruption.receivedAt >= grace
      || frame.capturedAt - interruption.capturedAt >= grace))) this.loseTracking()
    this.lastSequence = frame.sequence
    this.lastCapture = frame.capturedAt
    this.lastReceivedAt = receivedAt
    const hands = frame.hands.filter(isUsableHand)
    if (frame.tracking === 'lost' || frame.tracking === 'ambiguous' || frame.hands.length > 2
      || new Set(frame.hands.map(hand => hand.id)).size !== frame.hands.length) {
      this.interruptTracking()
      return
    }

    if (this.updateReset(hands, frame.capturedAt)) {
      if (this.timeout === undefined) this.checkTimeout()
      return
    }

    const grip = this.grip
    if (grip) {
      const first = hands.find(hand => hand.id === grip.first.id)
      const second = grip.second && hands.find(hand => hand.id === grip.second?.id)
      if (this.interruption || !first || (grip.second && !second)) {
        if ((first && isNeutral(first)) || (second && isNeutral(second))) {
          const remaining = [first, second].find(hand => hand && matchesGrip(grip, hand))
          this.interruption = null
          if (remaining) this.rebase(grip, remaining, null)
          else this.endGrip()
          this.armed = new Set(hands.filter(isNeutral).map(hand => hand.id))
          this.recovering = false
          this.updateState(hands)
          return
        }
        if (first && matchesGrip(grip, first)
          && (!grip.second || (second && matchesGrip(grip, second)))) {
          this.interruption = null
          this.rebase(grip, first, second || null)
          this.updateState(hands)
          return
        }
        if (this.liveTracking && grip.second && interruption
          && (receivedAt - interruption.receivedAt >= PAIR_RECOVERY_MS
            || frame.capturedAt - interruption.capturedAt >= PAIR_RECOVERY_MS)) {
          const remaining = [first, second].find(hand => hand && matchesGrip(grip, hand))
          if (remaining) {
            this.interruption = null
            this.rebase(grip, remaining, null)
            this.updateState(hands)
            return
          }
        }
        this.interruptTracking()
        return
      }
    }

    if (this.liveTracking && !hands.length && this.state.hands.length) {
      this.loseTracking()
      return
    }

    for (const [id, candidate] of this.hoverCandidates) {
      const hand = hands.find(hand => hand.id === id)
      if (!hand || frame.capturedAt - candidate.capturedAt > TARGET_GRACE_MS
        || Math.hypot(
          (hand.x - candidate.x) * this.viewportWidth,
          (hand.y - candidate.y) * this.viewportHeight,
        ) > TARGET_GRACE_PX) this.hoverCandidates.delete(id)
    }

    if (hands.length) {
      if (this.timeout === undefined) this.checkTimeout()
    } else {
      clearTimeout(this.timeout)
      this.timeout = undefined
    }
    const neutral = hands.filter(isNeutral)
    if (neutral.length) this.recovering = false
    if (this.retryPinch !== null && hands.some(hand => hand.id === this.retryPinch && isNeutral(hand))) this.retryPinch = null
    const fresh = hands.filter(hand => (hand.pinched || hand.fist) && this.armed.has(hand.id))
    // Require an open hand between grips, including a direct pinch-to-fist change.
    this.armed = new Set(neutral.map(hand => hand.id))

    if (this.grip) {
      this.updateGrip(hands, fresh)
    } else {
      for (const hand of fresh) {
        if (this.startGrip(hand)) break
        if (this.liveTracking && hand.pinched && fresh.length === 1 && this.closedCandidate === null) this.retryPinch = hand.id
      }
      if (!this.grip && (this.recoverClosed || this.retryPinch !== null)) this.tryClosedRecovery(hands)
    }
    this.updateState(hands)
  }

  private checkTimeout() {
    const deadline = this.interruption
      ? this.interruption.receivedAt + (this.liveTracking ? FRAME_MAX_AGE_MS : TRACKING_GRACE_MS)
      : this.lastReceivedAt + FRAME_MAX_AGE_MS
    const remaining = deadline - performance.now()
    if (remaining <= 0) {
      this.loseTracking()
      return
    }
    this.timeout = setTimeout(() => this.checkTimeout(), remaining)
  }

  private loseTracking(allowClosedRecovery = true) {
    clearTimeout(this.timeout)
    this.timeout = undefined
    this.resetCandidate = null
    this.recoverClosed = this.liveTracking && allowClosedRecovery
      && (this.recoverClosed || this.grip !== null || this.armed.size > 0 || this.retryPinch !== null)
    this.endGrip()
    this.armed.clear()
    this.hoverCandidates.clear()
    this.closedCandidate = null
    this.recovering = true
    this.updateState([])
  }

  private interruptTracking() {
    this.resetCandidate = null
    if (!this.grip) {
      this.loseTracking()
      return
    }
    if (!this.interruption) {
      // Camera processing can delay a report. The browser gives it a full
      // grace period while capture timestamps still bound the physical gap.
      this.interruption = { capturedAt: this.lastCapture, receivedAt: this.lastReceivedAt }
      clearTimeout(this.timeout)
      this.timeout = undefined
    }
    this.armed.clear()
    this.hoverCandidates.clear()
    this.closedCandidate = null
    if (this.timeout === undefined) this.checkTimeout()
    this.updateState([])
  }

  private updateReset(hands: readonly HandPose[], capturedAt: number) {
    const thumbs = hands.filter(hand => hand.depthValid && hand.thumbsUp)
    const firstThumb = () => thumbs.reduce((first, hand) => hand.id < first.id ? hand : first)
    if (this.resetOwner !== null) {
      const owner = hands.find(hand => hand.id === this.resetOwner)
      if (owner?.depthValid && !owner.thumbsUp && !thumbs.length) this.resetOwner = null
      else if (!owner && thumbs.length) {
        // A restarted source may rename a still-held hand. Retain the lock and
        // require this newly observed owner to release before another reset.
        this.resetOwner = firstThumb().id
      }
    }
    if (!thumbs.length) {
      this.resetCandidate = null
      return false
    }

    this.endGrip()
    this.armed.clear()
    this.hoverCandidates.clear()
    this.recoverClosed = false
    this.recovering = false
    if (this.resetOwner !== null) {
      this.resetCandidate = null
      this.updateState(hands)
      return true
    }
    const hand = thumbs.find(hand => hand.id === this.resetCandidate?.id) ?? firstThumb()
    if (this.resetCandidate?.id === hand.id) {
      this.resetCandidate.elapsed += capturedAt - this.resetCandidate.capturedAt
      this.resetCandidate.capturedAt = capturedAt
    } else this.resetCandidate = { id: hand.id, capturedAt, elapsed: 0 }
    if (this.resetCandidate.elapsed >= RESET_HOLD_MS) {
      this.resetOwner = hand.id
      this.resetCandidate = null
      this.updateState(hands)
      this.onReset?.()
    } else this.updateState(hands)
    return true
  }

  private startGrip(first: HandPose) {
    let part: PartGrip | null = null
    if (first.pinched) {
      // Closing the fingers can move the aim just beyond the highlighted surface.
      const id = this.scene.pick(first.x, first.y) ?? this.hoverCandidates.get(first.id)?.id
      if (!id) return false
      const pose = this.scene.getPose(id)
      if (!pose) return false
      part = { id, original: pose, base: pose, startFirst: this.worldPoint(first, pose), startSecond: null }
      this.scene.select(id)
    }

    this.grip = {
      first, second: null, viewAnchor: { x: first.x, y: first.y, depth: first.depth },
      joinSide: first.x <= 0.64 ? 1 : -1, joinVertical: first.y <= 0.5 ? 1 : -1,
      pairBlocked: false, part,
    }
    this.hoverCandidates.clear()
    this.recoverClosed = false
    this.retryPinch = null
    this.recovering = false
    this.closedCandidate = null
    return true
  }

  private matchesClosedCandidate(hand: HandPose, target: string | null, requireSteadyAim: boolean) {
    const candidate = this.closedCandidate
    if (!candidate || candidate.hand.id !== hand.id || candidate.hand.pinched !== hand.pinched || candidate.target !== target) return false
    if (!requireSteadyAim) return true
    const travel = Math.hypot(
      (hand.x - candidate.hand.x) * this.viewportWidth,
      (hand.y - candidate.hand.y) * this.viewportHeight,
    )
    const steadyDepth = !hand.depthValid || !candidate.hand.depthValid || Math.abs(hand.depth - candidate.hand.depth) <= 0.035
    return travel <= TARGET_GRACE_PX && steadyDepth
  }

  private confirmClosed(hand: HandPose, target: string | null, requireSteadyAim: boolean) {
    if (!this.matchesClosedCandidate(hand, target, requireSteadyAim)) {
      this.closedCandidate = { hand, target, capturedAt: this.lastCapture }
      return false
    }
    return this.lastCapture - this.closedCandidate!.capturedAt >= CLOSED_GRIP_CONFIRM_MS
  }

  private tryClosedRecovery(hands: readonly HandPose[]) {
    const closed = hands.filter(hand => this.retryPinch !== null
      ? hand.id === this.retryPinch && hand.pinched : this.recoverClosed && (hand.pinched || hand.fist))
    if (closed.length > 1 && closed.some(hand => hand.fist)) {
      this.closedCandidate = null
      return
    }
    const candidates = closed.map(hand => ({
      hand, target: hand.pinched ? this.scene.pick(hand.x, hand.y) : null,
    })).filter(({ hand, target }) => target !== null || hand.fist)
    const existing = candidates.find(({ hand, target }) => this.matchesClosedCandidate(hand, target, true))
    const sharedPart = candidates.length > 0 && candidates.every(({ hand, target }) => hand.pinched && target === candidates[0].target)
    const selected = existing ?? (candidates.length === 1 || sharedPart
      ? candidates.sort((left, right) => left.hand.id.localeCompare(right.hand.id))[0] : undefined)
    if (!selected) {
      this.closedCandidate = null
      return
    }
    const { hand, target } = selected
    if (this.confirmClosed(hand, target, true)) this.startGrip(hand)
  }

  private updateGrip(hands: readonly HandPose[], fresh: readonly HandPose[]) {
    const grip = this.grip!
    const first = hands.find(hand => hand.id === grip.first.id)
    const second = (grip.second && hands.find(hand => hand.id === grip.second?.id)) ?? null
    if (!first || (grip.second && !second)) {
      this.endGrip()
      this.armed.clear()
      this.recovering = true
      return
    }

    const firstHolding = matchesGrip(grip, first)
    const secondHolding = second && matchesGrip(grip, second) ? second : null
    if (!firstHolding || (second && !secondHolding)) {
      const remaining = firstHolding ? first : secondHolding
      if (remaining) this.rebase(grip, remaining, null)
      else this.endGrip()
      return
    }

    const useDepth = hasDepth(first, second)
    if (useDepth !== hasDepth(grip.first, grip.second)) {
      if (grip.part) {
        // Keep the old XY anchors so this frame still moves in the screen plane.
        // Anchor depth to this sample so its change in availability adds no motion.
        const anchorFirst = { ...first, x: grip.first.x, y: grip.first.y }
        const anchorSecond = second && grip.second
          ? { ...second, x: grip.second.x, y: grip.second.y } : null
        if (!this.rebasePart(grip.part, anchorFirst, anchorSecond)) return
        grip.first = anchorFirst
        grip.second = anchorSecond
        if (anchorSecond && this.handAxis(anchorFirst, anchorSecond).lengthSq() < MIN_HAND_SEPARATION ** 2) {
          grip.pairBlocked = true
        }
      } else grip.viewAnchor.depth = first.depth
    }

    if (!samePosition(grip.first, first, useDepth) || (grip.second && second && !samePosition(grip.second, second, useDepth))) {
      if (second) {
        this.updatePair(grip, first, second)
      } else if (grip.part) {
        this.movePart(grip.part, first, null)
      } else {
        const anchor = grip.viewAnchor
        const dx = first.x - anchor.x
        const dy = first.y - anchor.y
        const travel = Math.hypot(dx, dy)
        if (travel > VIEW_ORBIT_DEADZONE) {
          const fraction = 1 - VIEW_ORBIT_DEADZONE / travel
          this.scene.orbit(dx * fraction, dy * fraction)
          anchor.x += dx * fraction
          anchor.y += dy * fraction
        }
        if (useDepth) {
          const depth = first.depth - anchor.depth
          const excess = Math.max(0, Math.abs(depth) - VIEW_DEPTH_DEADZONE)
          if (excess > 0) {
            const applied = Math.sign(depth) * excess
            this.scene.zoom(Math.exp(applied * VIEW_DEPTH_ZOOM_GAIN))
            anchor.depth += applied
          }
        }
      }
    }

    grip.first = first
    grip.second = second
    if (grip.part && !second) {
      const join = this.joinTarget(grip)
      const joining = hands.find(hand => hand.id !== first.id && hand.pinched)
      if (!joining) {
        this.closedCandidate = null
        return
      }
      const confirmed = this.liveTracking && this.confirmClosed(joining, grip.part.id, false)
      if ((fresh.includes(joining) || confirmed)
        && this.handAxis(first, joining).lengthSq() >= MIN_HAND_SEPARATION ** 2
        && (this.joinAnywhere || isInsideJoin(join, joining.x, joining.y))) this.rebase(grip, first, joining)
    }
  }

  private handAxis(first: HandPose, second: HandPose) {
    // Depth and both screen axes are expressed in viewport widths.
    return new Vector3(
      second.x - first.x,
      (second.y - first.y) * this.viewportHeight / this.viewportWidth,
      hasDepth(first, second) ? second.depth - first.depth : 0,
    )
  }

  private updatePair(grip: ActiveGrip, first: HandPose, second: HandPose) {
    const previous = this.handAxis(grip.first, grip.second!)
    const axis = this.handAxis(first, second)
    const separation = axis.length()
    const interrupted = grip.pairBlocked || crossedBetweenFrames(previous, axis)
    const blocked = separation < (interrupted ? RESUME_HAND_SEPARATION : MIN_HAND_SEPARATION)
    const recover = !blocked && interrupted
    if (grip.part) {
      this.movePart(grip.part, first, second, blocked || recover)
    }
    if (recover) this.rebase(grip, first, second)
    grip.pairBlocked = blocked
  }

  private rebase(grip: ActiveGrip, first: HandPose, second: HandPose | null) {
    this.closedCandidate = null
    grip.first = first
    grip.second = second
    grip.viewAnchor = { x: first.x, y: first.y, depth: first.depth }
    grip.pairBlocked = second !== null && this.handAxis(first, second).lengthSq() < MIN_HAND_SEPARATION ** 2
    if (!second) {
      grip.joinSide = first.x <= 0.64 ? 1 : -1
      grip.joinVertical = first.y <= 0.5 ? 1 : -1
    }
    if (grip.part) this.rebasePart(grip.part, first, second)
  }

  private rebasePart(part: PartGrip, first: HandPose, second: HandPose | null) {
    const pose = this.scene.getPose(part.id)
    if (!pose) {
      this.endGrip()
      return false
    }
    const useDepth = hasDepth(first, second)
    part.base = pose
    part.startFirst = this.worldPoint(first, pose, useDepth)
    part.startSecond = second && this.worldPoint(second, pose, useDepth)
    return true
  }

  private worldPoint(hand: HandPose, pose: ObjectPose, useDepth = hand.depthValid) {
    return this.scene.screenPoint(hand.x, hand.y, pose.position, useDepth ? hand.depth : 0)
  }

  private movePart(part: PartGrip, first: HandPose, second: HandPose | null, preserveOrientation = false) {
    const useDepth = hasDepth(first, second)
    const point = this.worldPoint(first, part.base, useDepth)
    if (!second || !part.startSecond) {
      this.scene.setPose(part.id, { ...part.base, position: point.sub(part.startFirst).add(part.base.position) })
      return
    }

    const secondPoint = this.worldPoint(second, part.base, useDepth)
    const midpointOffset = point.clone().add(secondPoint).sub(part.startFirst).sub(part.startSecond).multiplyScalar(0.5)
    const position = part.base.position.clone().add(midpointOffset)
    if (preserveOrientation) {
      const current = this.scene.getPose(part.id)
      if (current) this.scene.setPose(part.id, { ...current, position })
      return
    }

    const startAxis = part.startSecond.clone().sub(part.startFirst)
    const axis = secondPoint.sub(point)
    const startLength = startAxis.length()
    const length = axis.length()
    if (startLength <= 1e-8 || length <= 1e-8) return
    const scale = clamp(part.base.scale * length / startLength, Math.min(0.2, part.base.scale), Math.max(5, part.base.scale))
    const rotation = new Quaternion().setFromUnitVectors(startAxis.divideScalar(startLength), axis.divideScalar(length))
      .multiply(part.base.rotation)
    this.scene.setPose(part.id, { position, rotation, scale })
  }

  private joinTarget(grip: ActiveGrip): JoinTarget {
    const { viewportWidth: width, viewportHeight: height } = this
    const radiusPx = Math.min(
      clamp(Math.min(width, height) * 0.028, 21, 35),
      width * (0.5 - CALIBRATION_INSET_X),
      height * (0.5 - CALIBRATION_INSET_Y),
    )
    const radiusX = radiusPx / width
    const radiusY = radiusPx / height
    const insetX = Math.min(0.5, CALIBRATION_INSET_X + radiusX)
    const insetY = Math.min(0.5, CALIBRATION_INSET_Y + radiusY)
    const offset = 0.19
    const edgeX = clamp(grip.first.x + grip.joinSide * offset, insetX, 1 - insetX)
    // Slide along the edge instead of flipping to the other side of the hand.
    const overflow = Math.max(0, offset - Math.max(0, (edgeX - grip.first.x) * grip.joinSide))
    const y = clamp(grip.first.y + grip.joinVertical * overflow * width / height, insetY, 1 - insetY)
    // At a corner, use any remaining reach to slide inward on the next edge.
    const usedVertical = Math.max(0, (y - grip.first.y) * grip.joinVertical) * height / width
    const x = clamp(edgeX - grip.joinSide * Math.max(0, overflow - usedVertical), insetX, 1 - insetX)
    return { x, y, radiusX, radiusY, radiusPx }
  }

  private endGrip() {
    const part = this.grip?.part
    this.grip = null
    this.interruption = null
    this.closedCandidate = null
    this.retryPinch = null
    if (part) this.scene.commitGrip(part.id, part.original)
  }

  private updateState(hands: readonly HandPose[]) {
    const grip = this.grip
    const activeHands = grip && !this.interruption ? hands.filter(hand =>
      (hand.id === grip.first.id || hand.id === grip.second?.id) && matchesGrip(grip, hand)).map(hand => hand.id) : []
    const hovered: string[] = []
    if (!grip) {
      for (const hand of hands.filter(isNeutral)) {
        const id = this.scene.pick(hand.x, hand.y)
        if (id === null) continue
        if (this.armed.has(hand.id)) {
          this.hoverCandidates.set(hand.id, { id, x: hand.x, y: hand.y, capturedAt: this.lastCapture })
        }
        if (!hovered.includes(id)) hovered.push(id)
      }
    }
    const state: GestureState = {
      status: grip?.second ? 'both' : grip ? 'holding' : this.recovering ? 'lost' : hands.length ? 'aim' : 'ready',
      hands,
      activeHands,
      resetProgress: this.resetCandidate ? clamp(this.resetCandidate.elapsed / RESET_HOLD_MS, 0, 1) : 0,
      hovered: hovered[0] ?? null,
      held: grip?.part?.id ?? null,
      target: grip ? grip.part ? 'part' : 'view' : null,
      join: grip?.part && !grip.second ? this.joinTarget(grip) : null,
    }
    const previous = this.state
    if (hovered.length !== this.hovered.length || hovered.some((id, index) => id !== this.hovered[index])) {
      this.scene.hover(hovered)
      this.hovered = hovered
    }
    if (previous.status === state.status && previous.hovered === state.hovered && previous.resetProgress === state.resetProgress
      && previous.held === state.held && previous.target === state.target && sameHands(previous.hands, hands)
      && previous.activeHands.length === state.activeHands.length
      && previous.activeHands.every((id, index) => id === state.activeHands[index])) return
    this.state = state
    this.onState(state)
  }
}
