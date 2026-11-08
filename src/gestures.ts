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
  readonly pinchPending?: boolean
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
  pan(dx: number, dy: number): void
  zoom(factor: number): void
  commitGrip(id: string, startingPose: ObjectPose): void
}

interface PartGrip {
  id: string
  original: ObjectPose
  base: ObjectPose
  startFirst: Vector3
  pair: { startSecond: Vector3; direction: Vector3; rotation: Quaternion } | null
  useDepth: boolean
  depthCandidate: { capturedAt: number; first: number; second: number | null } | null
}

interface SecondaryGrip {
  id: string
  observedAt: number
  heldMs: number
  missing: boolean
  recoverDepth?: boolean
}

interface ClosedCandidate {
  hand: HandPose
  target: string | null
  secondFistId?: string
  parallelPinch?: HandPose
  capturedAt: number
}

interface ActiveGrip {
  first: HandPose
  second: HandPose | null
  secondaryGrip: SecondaryGrip | null
  viewAnchor: { x: number; y: number; depth: number }
  viewPairLogSeparation: number | null
  joinSide: number
  joinVertical: number
  pairBlocked: boolean
  part: PartGrip | null
}

export const FRAME_MAX_AGE_MS = 240
export const RESET_HOLD_MS = 1500
const RESET_DROPOUT_GRACE_MS = 180
const RESET_RELEASE_CONFIRM_MS = 400
export const LIVE_PINCH_CONTINUITY_MS = 320
const MIN_HAND_SEPARATION = 0.06
const RESUME_HAND_SEPARATION = 0.08
const TARGET_GRACE_MS = 200
const TARGET_GRACE_PX = 28
const LIVE_TARGET_GRACE_FRACTION = 0.025
const TRACKING_GRACE_MS = 150
const CLOSED_GRIP_CONFIRM_MS = 100
const PINCH_REIDENTIFY_DISTANCE = 0.08
const DEPTH_RECOVERY_TOLERANCE = 0.05 // In viewport widths, like HandPose.depth.
const DEPTH_RECOVERY_MAX_GAP_MS = 100
const VIEW_ORBIT_DEADZONE = 0.005
const VIEW_DEPTH_DEADZONE = 0.006
const VIEW_DEPTH_ZOOM_GAIN = 2
const VIEW_PAIR_SPREAD_DEADZONE = 0.045

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
      && Boolean(hand.thumbsUp) === Boolean(other.thumbsUp) && Boolean(hand.pinchPending) === Boolean(other.pinchPending)
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
  private interruption: { capturedAt: number; receivedAt: number; knownIds: Set<string>; allowNewId: boolean } | null = null
  private armed = new Set<string>()
  private lastSequence = -1
  private lastCapture = -Infinity
  private lastReceivedAt = -Infinity
  private timeout: ReturnType<typeof setTimeout> | undefined
  private recovering = false
  private recoverClosed = false
  private retryPinch: string | null = null
  private closedCandidate: ClosedCandidate | null = null
  private hovered: string[] = []
  private hoverCandidates = new Map<string, { id: string; x: number; y: number; capturedAt: number }>()
  private resetCandidate: { id: string; capturedAt: number; elapsed: number; paused: boolean } | null = null
  // Keep the latch through a dropped frame or stream restart.
  private resetOwner: string | null = null
  private resetReleaseSince: number | null = null
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
    this.resetReleaseSince = null
    this.lastSequence = -1
    this.lastCapture = -Infinity
    this.lastReceivedAt = -Infinity
    this.loseTracking(false)
    this.recoverClosed = this.liveTracking && allowClosedRecovery
  }

  cancel() {
    this.resetCandidate = null
    this.resetReleaseSince = null
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

    const gap = Math.max(receivedAt - this.lastReceivedAt, frame.capturedAt - this.lastCapture)
    const quiet = this.state.hands.length > 0 && gap >= FRAME_MAX_AGE_MS
    const interruption = this.interruption
    const continuity = this.liveTracking && this.grip?.part ? LIVE_PINCH_CONTINUITY_MS : FRAME_MAX_AGE_MS
    const grace = this.liveTracking ? continuity : TRACKING_GRACE_MS
    if ((quiet && gap >= continuity) || (interruption && (receivedAt - interruption.receivedAt >= grace
      || frame.capturedAt - interruption.capturedAt >= grace))) this.loseTracking()
    const previousCapture = this.lastCapture
    const previousReceipt = this.lastReceivedAt
    this.lastSequence = frame.sequence
    this.lastCapture = frame.capturedAt
    this.lastReceivedAt = receivedAt
    const hands = frame.hands.filter(isUsableHand)
    const ambiguous = frame.tracking === 'ambiguous' || frame.hands.length > 2
      || new Set(frame.hands.map(hand => hand.id)).size !== frame.hands.length
    this.observeResetRelease(hands, frame.capturedAt, gap, ambiguous)
    if (frame.tracking === 'lost' || ambiguous) {
      if (!ambiguous && this.pauseReset(frame.capturedAt)) {
        this.updateState([])
        if (this.timeout === undefined) this.checkTimeout()
        return
      }
      this.interruptTracking(previousCapture, previousReceipt, ambiguous)
      return
    }

    if (this.updateReset(hands, frame.capturedAt)) {
      if (this.timeout === undefined) this.checkTimeout()
      return
    }

    const grip = this.grip
    if (grip) {
      const pending = grip.secondaryGrip
      if (pending && hands.some(hand => hand.id === pending.id && !matchesGrip(grip, hand))) grip.secondaryGrip = null
      const first = hands.find(hand => hand.id === grip.first.id)
      const second = grip.second && hands.find(hand => hand.id === grip.second?.id)
      if (this.interruption || quiet || !first || (grip.second && !second)) {
        const recoverDepth = this.interruption !== null || quiet
        if ((first && !matchesGrip(grip, first)) || (second && !matchesGrip(grip, second))) {
          const remaining = [first, second].find(hand => hand && matchesGrip(grip, hand))
          this.interruption = null
          if (remaining) this.rebase(grip, remaining, null, recoverDepth)
          else this.endGrip()
          this.armed = new Set(hands.filter(isNeutral).map(hand => hand.id))
          this.recovering = false
          this.updateState(hands)
          return
        }
        if (first && matchesGrip(grip, first)
          && (!grip.second || (second && matchesGrip(grip, second)))) {
          this.interruption = null
          this.rebase(grip, first, second || null, true)
          if (!grip.second) {
            if (grip.secondaryGrip) grip.secondaryGrip.missing = true
            this.tryJoin(grip, hands, [])
          }
          this.updateState(hands)
          return
        }
        if (grip.second && (this.liveTracking || !grip.part)) {
          const remaining = [first, second].find(hand => hand && matchesGrip(grip, hand))
          if (remaining) {
            const missing = first ? grip.second : grip.first
            grip.secondaryGrip = {
              id: missing.id, observedAt: interruption?.capturedAt ?? previousCapture,
              heldMs: CLOSED_GRIP_CONFIRM_MS, missing: true, recoverDepth: true,
            }
            this.interruption = null
            this.rebase(grip, remaining, null, recoverDepth)
            this.updateState(hands)
            return
          }
        }
        if (this.liveTracking && grip.part && !grip.second && !first && interruption?.allowNewId !== false
          && frame.capturedAt - (interruption?.capturedAt ?? previousCapture) < LIVE_PINCH_CONTINUITY_MS) {
          const knownIds = interruption?.knownIds ?? new Set(this.state.hands.map(hand => hand.id))
          const nearby = hands.filter(hand => hand.id !== grip.secondaryGrip?.id && !knownIds.has(hand.id)
            && hand.pinched && this.handAxis(grip.first, hand).lengthSq() <= PINCH_REIDENTIFY_DISTANCE ** 2)
          if (nearby.length === 1) {
            this.interruption = null
            this.rebase(grip, nearby[0], null, true)
            this.updateState(hands)
            return
          }
        }
        this.interruptTracking(previousCapture, previousReceipt)
        return
      }
    }

    if (this.liveTracking && !hands.length && this.state.hands.length) {
      this.loseTracking()
      return
    }

    const aimTolerance = this.liveTracking ? LIVE_TARGET_GRACE_FRACTION : 0
    const radiusX = Math.max(TARGET_GRACE_PX / this.viewportWidth, aimTolerance)
    const radiusY = Math.max(TARGET_GRACE_PX / this.viewportHeight, aimTolerance)
    for (const [id, candidate] of this.hoverCandidates) {
      const hand = hands.find(hand => hand.id === id)
      if (!hand || frame.capturedAt - candidate.capturedAt > TARGET_GRACE_MS
        || Math.hypot((hand.x - candidate.x) / radiusX, (hand.y - candidate.y) / radiusY) > 1) {
        this.hoverCandidates.delete(id)
      }
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
        if (this.startGrip(hand, hands, fresh)) break
        if (this.liveTracking && hand.pinched && fresh.length === 1 && this.closedCandidate === null) this.retryPinch = hand.id
      }
      if (!this.grip && (this.recoverClosed || this.retryPinch !== null)) this.tryClosedRecovery(hands)
    }
    this.updateState(hands)
  }

  private checkTimeout() {
    const continuity = this.liveTracking && this.grip?.part ? LIVE_PINCH_CONTINUITY_MS : FRAME_MAX_AGE_MS
    const deadline = this.interruption
      ? this.interruption.receivedAt + (this.liveTracking ? continuity : TRACKING_GRACE_MS)
      : this.lastReceivedAt + continuity
    const remaining = deadline - performance.now()
    if (remaining <= 0) {
      this.loseTracking()
      return
    }
    this.timeout = setTimeout(() => this.checkTimeout(), Math.min(remaining, FRAME_MAX_AGE_MS))
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

  private interruptTracking(previousCapture: number, previousReceipt: number, ambiguous = false) {
    this.resetCandidate = null
    if (!this.grip) {
      this.loseTracking()
      return
    }
    if (!this.interruption) {
      // Live part grips keep their original deadline across repeated misses.
      const livePart = this.liveTracking && this.grip.part !== null
      this.interruption = {
        capturedAt: livePart ? previousCapture : this.lastCapture,
        receivedAt: livePart ? previousReceipt : this.lastReceivedAt,
        knownIds: new Set(this.state.hands.map(hand => hand.id)), allowNewId: !ambiguous,
      }
      clearTimeout(this.timeout)
      this.timeout = undefined
    } else if (ambiguous) this.interruption.allowNewId = false
    this.armed.clear()
    this.hoverCandidates.clear()
    this.closedCandidate = null
    if (this.grip.secondaryGrip) this.grip.secondaryGrip.missing = true
    if (this.timeout === undefined) this.checkTimeout()
    this.updateState([])
  }

  private observeResetRelease(hands: readonly HandPose[], capturedAt: number, gap: number, ambiguous: boolean) {
    if (this.resetOwner === null) return
    const owner = hands.find(hand => hand.id === this.resetOwner)
    if (ambiguous || (owner && !owner.depthValid) || hands.some(hand => hand.thumbsUp)) {
      this.resetReleaseSince = null
      return
    }
    // An open hand releases immediately. When it leaves the frame, require a
    // series of fresh observations without the gesture before rearming.
    if (this.resetReleaseSince === null || gap >= FRAME_MAX_AGE_MS) this.resetReleaseSince = capturedAt
    if ((owner && isNeutral(owner)) || capturedAt - this.resetReleaseSince >= RESET_RELEASE_CONFIRM_MS) {
      this.resetOwner = null
      this.resetReleaseSince = null
    }
  }

  private pauseReset(capturedAt: number) {
    const candidate = this.resetCandidate
    if (!this.liveTracking || !candidate || capturedAt - candidate.capturedAt > RESET_DROPOUT_GRACE_MS) {
      this.resetCandidate = null
      return false
    }
    candidate.paused = true
    return true
  }

  private updateReset(hands: readonly HandPose[], capturedAt: number) {
    const thumbs = hands.filter(hand => hand.depthValid && hand.thumbsUp)
    if (!thumbs.length) {
      const candidate = this.resetCandidate
      const hand = candidate && hands.find(hand => hand.id === candidate.id)
      if ((!hand || !hand.depthValid || hand.fist) && this.pauseReset(capturedAt)) {
        this.updateState(hands)
        return true
      }
      this.resetCandidate = null
      return false
    }

    this.endGrip()
    this.armed.clear()
    this.hoverCandidates.clear()
    this.recoverClosed = false
    this.recovering = false
    const firstThumb = () => thumbs.reduce((first, hand) => hand.id < first.id ? hand : first)
    if (this.resetOwner !== null) {
      // Tracking may rename a still-held hand during a source restart.
      if (!hands.some(hand => hand.id === this.resetOwner)) this.resetOwner = firstThumb().id
      this.resetCandidate = null
      this.updateState(hands)
      return true
    }
    const hand = thumbs.find(hand => hand.id === this.resetCandidate?.id) ?? firstThumb()
    let candidate = this.resetCandidate
    if (candidate?.id === hand.id && (!candidate.paused || capturedAt - candidate.capturedAt <= RESET_DROPOUT_GRACE_MS)) {
      if (!candidate.paused) candidate.elapsed += capturedAt - candidate.capturedAt
      candidate.capturedAt = capturedAt
      candidate.paused = false
    } else {
      candidate = { id: hand.id, capturedAt, elapsed: 0, paused: false }
      this.resetCandidate = candidate
    }
    if (candidate.elapsed >= RESET_HOLD_MS) {
      this.resetOwner = hand.id
      this.resetCandidate = null
      this.updateState(hands)
      this.onReset?.()
    } else this.updateState(hands)
    return true
  }

  private startGrip(first: HandPose, hands: readonly HandPose[], fresh: readonly HandPose[] = []) {
    let part: PartGrip | null = null
    if (first.pinched) {
      // Use the highlighted part when the live fingertip shifts a little as it closes.
      const aimedPart = this.hoverCandidates.get(first.id)?.id
      const id = this.liveTracking && aimedPart ? aimedPart : this.scene.pick(first.x, first.y) ?? aimedPart
      if (!id) return false
      const pose = this.scene.getPose(id)
      if (!pose) return false
      part = {
        id, original: pose, base: pose, startFirst: this.worldPoint(first, pose), pair: null,
        useDepth: first.depthValid, depthCandidate: null,
      }
      this.scene.select(id)
    }

    this.grip = {
      first, second: null, secondaryGrip: null, viewAnchor: { x: first.x, y: first.y, depth: first.depth },
      viewPairLogSeparation: null, joinSide: first.x <= 0.64 ? 1 : -1, joinVertical: first.y <= 0.5 ? 1 : -1,
      pairBlocked: false, part,
    }
    this.hoverCandidates.clear()
    this.recoverClosed = false
    this.retryPinch = null
    this.recovering = false
    this.closedCandidate = null
    this.tryJoin(this.grip, hands, fresh)
    return true
  }

  private matchesClosedCandidate(hand: HandPose, target: string | null) {
    const candidate = this.closedCandidate
    return candidate !== null && candidate.target === target && this.isSteadyClosedHand(hand, candidate.hand)
  }

  private isSteadyClosedHand(hand: HandPose, initial: HandPose) {
    if (initial.id !== hand.id || initial.pinched !== hand.pinched) return false
    const travel = Math.hypot(
      (hand.x - initial.x) * this.viewportWidth,
      (hand.y - initial.y) * this.viewportHeight,
    )
    const steadyDepth = !hand.depthValid || !initial.depthValid || Math.abs(hand.depth - initial.depth) <= 0.035
    return travel <= TARGET_GRACE_PX && steadyDepth
  }

  private tryClosedRecovery(hands: readonly HandPose[]) {
    const closed = hands.filter(hand => this.retryPinch !== null
      ? hand.id === this.retryPinch && hand.pinched : this.recoverClosed && (hand.pinched || hand.fist))
    if (closed.some(hand => hand.fist) && closed.some(hand => hand.pinched)) {
      this.closedCandidate = null
      return
    }
    const candidates = closed.map(hand => ({
      hand, target: hand.pinched ? this.scene.pick(hand.x, hand.y) : null,
    })).filter(({ hand, target }) => target !== null || hand.fist)
    const existing = candidates.find(({ hand, target }) => this.matchesClosedCandidate(hand, target))
    const sharedTarget = candidates.length > 0 && candidates.every(({ target }) => target === candidates[0].target)
    const selected = existing ?? (sharedTarget
      ? candidates.sort((left, right) => left.hand.id.localeCompare(right.hand.id))[0] : undefined)
    if (!selected) {
      this.closedCandidate = null
      return
    }
    const { hand, target } = selected
    const secondFist = hand.fist ? closed.find(other => other.id !== hand.id && other.fist) : undefined
    const secondPinch = hand.pinched && sharedTarget ? candidates.find(other => other.hand.id !== hand.id)?.hand : undefined
    const candidate = this.closedCandidate
    if (!candidate || !existing || candidate.secondFistId !== secondFist?.id) {
      this.closedCandidate = { hand, target, secondFistId: secondFist?.id, parallelPinch: secondPinch, capturedAt: this.lastCapture }
      return
    }
    // A second pinch can share the wait only if it stayed steady on the same
    // exact target throughout the primary's confirmation window.
    if (candidate.parallelPinch && (!secondPinch || !this.isSteadyClosedHand(secondPinch, candidate.parallelPinch))) {
      candidate.parallelPinch = undefined
    }
    if (this.lastCapture - candidate.capturedAt >= CLOSED_GRIP_CONFIRM_MS) {
      const second = secondFist ?? (candidate.parallelPinch && secondPinch)
      this.startGrip(hand, hands, second ? [second] : [])
    }
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

    if (!grip.part && !second) {
      this.tryJoin(grip, hands, fresh, first)
      if (grip.second) return
    }

    if (grip.part) this.updatePartGrip(grip, grip.part, first, second)
    else if (second && grip.second) {
      if (!samePosition(grip.first, first, false) || !samePosition(grip.second, second, false)) {
        this.updateViewPair(grip, first, second)
      }
    } else this.updateViewGrip(grip, first)

    grip.first = first
    grip.second = second
    if (grip.part && !second) this.tryJoin(grip, hands, fresh)
  }

  private tryJoin(grip: ActiveGrip, hands: readonly HandPose[], fresh: readonly HandPose[], first = grip.first) {
    const second = hands.find(hand => hand.id !== first.id)
    const candidate = grip.secondaryGrip
    const grace = grip.part ? LIVE_PINCH_CONTINUITY_MS : this.liveTracking ? FRAME_MAX_AGE_MS : TRACKING_GRACE_MS
    const recent = candidate && this.lastCapture - candidate.observedAt < grace ? candidate : null
    if (!second || !matchesGrip(grip, second)) {
      if (recent && (!second || second.id !== recent.id)) {
        recent.missing = true
        grip.secondaryGrip = recent
      } else grip.secondaryGrip = null
      return
    }

    let confirmed = fresh.includes(second) || (recent?.id === second.id && recent.heldMs >= CLOSED_GRIP_CONFIRM_MS)
    const recoverDepth = recent?.id === second.id && recent.recoverDepth === true
    if (this.liveTracking) {
      if (!confirmed && recent?.missing && recent.heldMs >= CLOSED_GRIP_CONFIRM_MS && recent.id !== second.id) return
      const heldMs = recent?.id === second.id
        ? recent.heldMs + (recent.missing ? 0 : this.lastCapture - recent.observedAt) : 0
      confirmed ||= heldMs >= CLOSED_GRIP_CONFIRM_MS
      grip.secondaryGrip = {
        id: second.id, observedAt: this.lastCapture,
        heldMs: confirmed ? CLOSED_GRIP_CONFIRM_MS : heldMs, missing: false, recoverDepth,
      }
    }
    const useDepth = grip.part?.useDepth === true && hasDepth(first, second)
    const canJoin = !grip.part || (this.handAxis(first, second, useDepth).lengthSq() >= MIN_HAND_SEPARATION ** 2
      && (this.joinAnywhere || isInsideJoin(this.joinTarget(grip), second.x, second.y)))
    if (confirmed && canJoin) {
      grip.secondaryGrip = null
      this.rebase(grip, first, second, recoverDepth)
    }
  }

  private handAxis(first: HandPose, second: HandPose, useDepth = hasDepth(first, second)) {
    // Depth and both screen axes are expressed in viewport widths.
    return new Vector3(
      second.x - first.x,
      (second.y - first.y) * this.viewportHeight / this.viewportWidth,
      useDepth ? second.depth - first.depth : 0,
    )
  }

  private updateViewGrip(grip: ActiveGrip, first: HandPose) {
    const anchor = grip.viewAnchor
    const useDepth = first.depthValid
    if (useDepth !== grip.first.depthValid) anchor.depth = first.depth
    if (samePosition(grip.first, first, useDepth)) return

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

  private updateViewPair(grip: ActiveGrip, first: HandPose, second: HandPose) {
    const axis = this.handAxis(first, second, false)
    const separation = axis.length()
    const spreadAnchor = grip.viewPairLogSeparation
    const minimum = spreadAnchor === null ? RESUME_HAND_SEPARATION : MIN_HAND_SEPARATION
    if (separation < minimum) {
      grip.viewPairLogSeparation = null
    } else {
      const logSeparation = Math.log(separation)
      const previous = this.handAxis(grip.first, grip.second!, false)
      // Re-anchor when the hands cross so the new separation does not jump the zoom.
      if (spreadAnchor === null || crossedBetweenFrames(previous, axis)) {
        grip.viewPairLogSeparation = logSeparation
      } else {
        const spreadDelta = logSeparation - spreadAnchor
        const spread = spreadDelta - clamp(spreadDelta, -VIEW_PAIR_SPREAD_DEADZONE, VIEW_PAIR_SPREAD_DEADZONE)
        if (spread) this.scene.zoom(Math.exp(clamp(spread, -0.2, 0.2)))
        grip.viewPairLogSeparation = spreadAnchor + spread
      }
    }

    const anchor = grip.viewAnchor
    const x = (first.x + second.x) / 2
    const y = (first.y + second.y) / 2
    if (x !== anchor.x || y !== anchor.y) this.scene.pan(x - anchor.x, y - anchor.y)
    anchor.x = x
    anchor.y = y
  }

  private updatePartGrip(grip: ActiveGrip, part: PartGrip, first: HandPose, second: HandPose | null) {
    if (part.useDepth && !hasDepth(first, second)) {
      // Keep the old XY anchors so the first missing-depth frame still moves in the screen plane.
      if (!this.rebasePart(part, grip.first, grip.second, false)) return
      if (grip.second && this.handAxis(grip.first, grip.second, false).lengthSq() < MIN_HAND_SEPARATION ** 2) {
        grip.pairBlocked = true
      }
    }

    if (!samePosition(grip.first, first, part.useDepth)
      || (grip.second && second && !samePosition(grip.second, second, part.useDepth))) {
      if (second) this.updatePair(grip, part, first, second)
      else this.movePart(part, first, null)
    }

    if (!part.useDepth && this.confirmDepth(part, first, second)) {
      // Apply this sample's XY motion before anchoring Z to its confirmed depth.
      if (!this.rebasePart(part, first, second, true)) return
      const minimum = grip.pairBlocked ? RESUME_HAND_SEPARATION : MIN_HAND_SEPARATION
      grip.pairBlocked = second !== null && this.handAxis(first, second, true).lengthSq() < minimum ** 2
    }
  }

  private depthSample(first: HandPose, second: HandPose | null) {
    return hasDepth(first, second) ? { capturedAt: this.lastCapture, first: first.depth, second: second?.depth ?? null } : null
  }

  private confirmDepth(part: PartGrip, first: HandPose, second: HandPose | null) {
    const current = this.depthSample(first, second)
    const previous = part.depthCandidate
    if (current && previous && current.capturedAt <= previous.capturedAt) return false
    part.depthCandidate = current
    return current !== null && previous !== null
      && current.capturedAt - previous.capturedAt <= DEPTH_RECOVERY_MAX_GAP_MS
      && Math.abs(current.first - previous.first) <= DEPTH_RECOVERY_TOLERANCE
      && (current.second === null || (previous.second !== null
        && Math.abs(current.second - previous.second) <= DEPTH_RECOVERY_TOLERANCE))
  }

  private updatePair(grip: ActiveGrip, part: PartGrip, first: HandPose, second: HandPose) {
    const previous = this.handAxis(grip.first, grip.second!, part.useDepth)
    const axis = this.handAxis(first, second, part.useDepth)
    const separation = axis.length()
    const interrupted = grip.pairBlocked || crossedBetweenFrames(previous, axis)
    const blocked = separation < (interrupted ? RESUME_HAND_SEPARATION : MIN_HAND_SEPARATION)
    const recover = !blocked && interrupted
    this.movePart(part, first, second, blocked || recover)
    if (recover) this.rebasePart(part, first, second, part.useDepth)
    grip.pairBlocked = blocked
  }

  private rebase(grip: ActiveGrip, first: HandPose, second: HandPose | null, recoverDepth = false) {
    this.closedCandidate = null
    grip.first = first
    grip.second = second
    grip.viewAnchor = {
      x: second ? (first.x + second.x) / 2 : first.x,
      y: second ? (first.y + second.y) / 2 : first.y,
      depth: first.depth,
    }
    const viewSeparation = !grip.part && second ? this.handAxis(first, second, false).length() : 0
    grip.viewPairLogSeparation = viewSeparation >= RESUME_HAND_SEPARATION ? Math.log(viewSeparation) : null
    if (!second) {
      grip.joinSide = first.x <= 0.64 ? 1 : -1
      grip.joinVertical = first.y <= 0.5 ? 1 : -1
    }
    if (grip.part) {
      const useDepth = !recoverDepth && grip.part.useDepth && hasDepth(first, second)
      grip.part.depthCandidate = useDepth ? null : this.depthSample(first, second)
      grip.pairBlocked = second !== null && this.handAxis(first, second, useDepth).lengthSq() < MIN_HAND_SEPARATION ** 2
      this.rebasePart(grip.part, first, second, useDepth)
    }
  }

  private rebasePart(part: PartGrip, first: HandPose, second: HandPose | null, useDepth: boolean) {
    const pose = this.scene.getPose(part.id)
    if (!pose) {
      this.endGrip()
      return false
    }
    part.useDepth = useDepth
    if (useDepth) part.depthCandidate = null
    part.base = pose
    part.startFirst = this.worldPoint(first, pose, useDepth)
    const startSecond = second && this.worldPoint(second, pose, useDepth)
    part.pair = startSecond && {
      startSecond,
      direction: startSecond.clone().sub(part.startFirst).normalize(),
      rotation: pose.rotation,
    }
    return true
  }

  private worldPoint(hand: HandPose, pose: ObjectPose, useDepth = hand.depthValid) {
    return this.scene.screenPoint(hand.x, hand.y, pose.position, useDepth ? hand.depth : 0)
  }

  private movePart(part: PartGrip, first: HandPose, second: HandPose | null, preserveOrientation = false) {
    const point = this.worldPoint(first, part.base, part.useDepth)
    const pair = part.pair
    if (!second || !pair) {
      this.scene.setPose(part.id, { ...part.base, position: point.sub(part.startFirst).add(part.base.position) })
      return
    }

    const secondPoint = this.worldPoint(second, part.base, part.useDepth)
    const midpointOffset = point.clone().add(secondPoint).sub(part.startFirst).sub(pair.startSecond).multiplyScalar(0.5)
    const position = part.base.position.clone().add(midpointOffset)
    if (preserveOrientation) {
      const current = this.scene.getPose(part.id)
      if (current) this.scene.setPose(part.id, { ...current, position })
      return
    }

    const axis = secondPoint.sub(point)
    const startLength = pair.startSecond.distanceTo(part.startFirst)
    const length = axis.length()
    if (startLength <= 1e-8 || length <= 1e-8) return
    const scale = clamp(part.base.scale * length / startLength, Math.min(0.2, part.base.scale), Math.max(5, part.base.scale))
    // The original direction becomes ambiguous at a half turn. Compose the
    // rotation between consecutive valid directions to carry the turn through it.
    const direction = axis.divideScalar(length)
    const rotation = new Quaternion().setFromUnitVectors(pair.direction, direction)
      .multiply(pair.rotation).normalize()
    pair.direction = direction
    pair.rotation = rotation
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
        // A pinch awaiting confirmation still arms the hand, but must not change its target.
        const id = hand.pinchPending ? this.hoverCandidates.get(hand.id)?.id : this.scene.pick(hand.x, hand.y)
        if (!id) continue
        if (!hand.pinchPending && this.armed.has(hand.id)) {
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
