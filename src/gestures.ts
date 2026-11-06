import { Quaternion, Vector3 } from 'three'

export interface HandPose {
  readonly id: string
  readonly x: number
  readonly y: number
  readonly depth: number
  readonly depthValid: boolean
  readonly pinched: boolean
  readonly fist?: boolean
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
export interface JoinTarget { x: number; y: number; radius: number }

export interface GestureState {
  status: GestureStatus
  hands: readonly HandPose[]
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
  joinSide: number
  part: PartGrip | null
}

export const FRAME_MAX_AGE_MS = 240
const MIN_HAND_SEPARATION = 0.06
const JOIN_RADIUS = 0.065

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value))
}

function isUsableHand(hand: HandPose) {
  return hand.depthValid && Number.isFinite(hand.depth)
    && hand.x >= 0 && hand.x <= 1 && hand.y >= 0 && hand.y <= 1
    && !(hand.pinched && hand.fist)
}

function isNeutral(hand: HandPose) {
  return !hand.pinched && !hand.fist
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
  })
}

function samePosition(left: HandPose, right: HandPose) {
  return left.x === right.x && left.y === right.y && left.depth === right.depth
}

function handSpacing(first: HandPose, second: HandPose) {
  return Math.hypot(first.x - second.x, first.y - second.y, first.depth - second.depth)
}

export class GestureController {
  private readonly scene: GestureScene
  private readonly onState: (state: GestureState) => void
  private grip: ActiveGrip | null = null
  private armed = new Set<string>()
  private lastSequence = -1
  private lastCapture = -Infinity
  private timeout: ReturnType<typeof setTimeout> | undefined
  private recovering = false
  private hovered: string[] = []
  private state: GestureState = {
    status: 'ready', hands: [], hovered: null, held: null, target: null, join: null,
  }

  constructor(scene: GestureScene, onState: (state: GestureState) => void) {
    this.scene = scene
    this.onState = onState
  }

  cancel() {
    this.endGrip()
    this.armed.clear()
    this.recovering = false
    this.updateState(this.state.hands)
  }

  dispose() {
    clearTimeout(this.timeout)
    this.timeout = undefined
  }

  receive(frame: PoseFrame) {
    if (!Number.isSafeInteger(frame.sequence) || frame.sequence <= this.lastSequence
      || !Number.isFinite(frame.capturedAt) || frame.capturedAt < this.lastCapture) return
    const age = performance.now() - frame.capturedAt
    if (age < -20 || age >= FRAME_MAX_AGE_MS) return

    if (this.state.hands.length && frame.capturedAt - this.lastCapture >= FRAME_MAX_AGE_MS) this.loseTracking()
    this.lastSequence = frame.sequence
    this.lastCapture = frame.capturedAt
    const hands = frame.hands.filter(isUsableHand)
    if (frame.tracking === 'lost' || frame.tracking === 'ambiguous' || frame.hands.length > 2
      || new Set(frame.hands.map(hand => hand.id)).size !== frame.hands.length) {
      this.loseTracking()
      return
    }

    if (hands.length) {
      if (this.timeout === undefined) this.checkTimeout()
    } else {
      clearTimeout(this.timeout)
      this.timeout = undefined
    }
    const neutral = hands.filter(isNeutral)
    if (neutral.length) this.recovering = false
    const fresh = hands.filter(hand => !isNeutral(hand) && this.armed.has(hand.id))
    // Require an open hand between grips, including a direct pinch-to-fist change.
    this.armed = new Set(neutral.map(hand => hand.id))

    if (this.grip) {
      this.updateGrip(hands, fresh)
    } else {
      for (const hand of fresh) {
        if (this.startGrip(hand)) break
      }
    }
    this.updateState(hands)
  }

  private checkTimeout() {
    const remaining = FRAME_MAX_AGE_MS - (performance.now() - this.lastCapture)
    if (remaining <= 0) {
      this.loseTracking()
      return
    }
    this.timeout = setTimeout(() => this.checkTimeout(), remaining)
  }

  private loseTracking() {
    clearTimeout(this.timeout)
    this.timeout = undefined
    this.endGrip()
    this.armed.clear()
    this.recovering = true
    this.updateState([])
  }

  private startGrip(first: HandPose) {
    let part: PartGrip | null = null
    if (first.pinched) {
      const id = this.scene.pick(first.x, first.y)
      if (!id) return false
      const pose = this.scene.getPose(id)
      if (!pose) return false
      part = { id, original: pose, base: pose, startFirst: this.worldPoint(first, pose), startSecond: null }
      this.scene.select(id)
    }

    this.grip = { first, second: null, joinSide: first.x <= 0.64 ? 1 : -1, part }
    return true
  }

  private updateGrip(hands: readonly HandPose[], fresh: readonly HandPose[]) {
    const grip = this.grip!
    const first = hands.find(hand => hand.id === grip.first.id)
    const second = grip.second && hands.find(hand => hand.id === grip.second?.id)
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

    if (!samePosition(grip.first, first) || (grip.second && second && !samePosition(grip.second, second))) {
      if (grip.part) this.movePart(grip.part, first, second || null)
      else this.moveView(grip, first, second || null)
    }

    grip.first = first
    grip.second = second || null
    if (!second) {
      const join = this.joinTarget(grip)
      const joining = fresh.find(hand => hand.id !== first.id && matchesGrip(grip, hand)
        && handSpacing(hand, first) >= MIN_HAND_SEPARATION
        && Math.hypot(hand.x - join.x, hand.y - join.y) <= join.radius)
      if (joining) this.rebase(grip, first, joining)
    }
  }

  private rebase(grip: ActiveGrip, first: HandPose, second: HandPose | null) {
    grip.first = first
    grip.second = second
    if (!second) grip.joinSide = first.x <= 0.64 ? 1 : -1
    const part = grip.part
    if (!part) return
    const pose = this.scene.getPose(part.id)
    if (!pose) {
      this.endGrip()
      return
    }
    part.base = pose
    part.startFirst = this.worldPoint(first, pose)
    part.startSecond = second && this.worldPoint(second, pose)
  }

  private worldPoint(hand: HandPose, pose: ObjectPose) {
    return this.scene.screenPoint(hand.x, hand.y, pose.position, hand.depth)
  }

  private movePart(part: PartGrip, first: HandPose, second: HandPose | null) {
    const point = this.worldPoint(first, part.base)
    if (!second || !part.startSecond) {
      this.scene.setPose(part.id, { ...part.base, position: point.sub(part.startFirst).add(part.base.position) })
      return
    }

    const secondPoint = this.worldPoint(second, part.base)
    const midpointOffset = point.clone().add(secondPoint).sub(part.startFirst).sub(part.startSecond).multiplyScalar(0.5)
    const position = part.base.position.clone().add(midpointOffset)
    if (handSpacing(first, second) < MIN_HAND_SEPARATION) {
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

  private moveView(grip: ActiveGrip, first: HandPose, second: HandPose | null) {
    if (grip.second && second) {
      const previous = Math.max(MIN_HAND_SEPARATION, handSpacing(grip.first, grip.second))
      const distance = Math.max(MIN_HAND_SEPARATION, handSpacing(first, second))
      if (distance !== previous) this.scene.zoom(distance / previous)
    } else {
      const dx = first.x - grip.first.x
      const dy = first.y - grip.first.y
      if (dx || dy) this.scene.orbit(dx, dy)
    }
  }

  private joinTarget(grip: ActiveGrip): JoinTarget {
    const side = grip.first.x + grip.joinSide * 0.19
    const x = side < 0.09 || side > 0.91 ? grip.first.x - grip.joinSide * 0.19 : side
    return { x: clamp(x, 0.09, 0.91), y: clamp(grip.first.y, 0.09, 0.91), radius: JOIN_RADIUS }
  }

  private endGrip() {
    const part = this.grip?.part
    this.grip = null
    if (part) this.scene.commitGrip(part.id, part.original)
  }

  private updateState(hands: readonly HandPose[]) {
    const grip = this.grip
    const hovered = !grip
      ? [...new Set(hands.filter(isNeutral).map(hand => this.scene.pick(hand.x, hand.y)).filter(id => id !== null))]
      : []
    const state: GestureState = {
      status: grip?.second ? 'both' : grip ? 'holding' : this.recovering ? 'lost' : hands.length ? 'aim' : 'ready',
      hands,
      hovered: hovered[0] ?? null,
      held: grip?.part?.id ?? null,
      target: grip ? grip.part ? 'part' : 'view' : null,
      join: grip && !grip.second ? this.joinTarget(grip) : null,
    }
    const previous = this.state
    if (hovered.length !== this.hovered.length || hovered.some((id, index) => id !== this.hovered[index])) {
      this.scene.hover(hovered)
      this.hovered = hovered
    }
    if (previous.status === state.status && previous.hovered === state.hovered
      && previous.held === state.held && previous.target === state.target && sameHands(previous.hands, hands)) return
    this.state = state
    this.onState(state)
  }
}
