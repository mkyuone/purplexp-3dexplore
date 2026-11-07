import { Quaternion, Vector3 } from 'three'
import { CALIBRATION_INSET_X, CALIBRATION_INSET_Y } from './reach.ts'

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
export interface JoinTarget { x: number; y: number; radiusX: number; radiusY: number; radiusPx: number }

export function isInsideJoin(target: JoinTarget, x: number, y: number) {
  return Math.hypot((x - target.x) / target.radiusX, (y - target.y) / target.radiusY) <= 1
}

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
  joinVertical: number
  pairBlocked: boolean
  part: PartGrip | null
}

export const FRAME_MAX_AGE_MS = 240
const MIN_HAND_SEPARATION = 0.06
const RESUME_HAND_SEPARATION = 0.08

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
  private viewportWidth = 1000
  private viewportHeight = 1000
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

  setViewport(width: number, height: number) {
    if (!Number.isFinite(width) || width <= 0 || !Number.isFinite(height) || height <= 0) return
    if (width === this.viewportWidth && height === this.viewportHeight) return
    this.viewportWidth = width
    this.viewportHeight = height
    this.cancel()
  }

  // Call on source replacement, after detaching the previous stream's listener.
  resetStream() {
    this.lastSequence = -1
    this.lastCapture = -Infinity
    this.loseTracking()
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

    this.grip = {
      first, second: null, joinSide: first.x <= 0.64 ? 1 : -1, joinVertical: first.y <= 0.5 ? 1 : -1,
      pairBlocked: false, part,
    }
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
      if (second) {
        this.updatePair(grip, first, second)
      } else if (grip.part) {
        this.movePart(grip.part, first, null)
      } else {
        const dx = first.x - grip.first.x
        const dy = first.y - grip.first.y
        if (dx || dy) this.scene.orbit(dx, dy)
      }
    }

    grip.first = first
    grip.second = second || null
    if (!second) {
      const join = this.joinTarget(grip)
      const joining = fresh.find(hand => hand.id !== first.id && matchesGrip(grip, hand)
        && this.handAxis(first, hand).lengthSq() >= MIN_HAND_SEPARATION ** 2
        && isInsideJoin(join, hand.x, hand.y))
      if (joining) this.rebase(grip, first, joining)
    }
  }

  private handAxis(first: HandPose, second: HandPose) {
    // Depth and both screen axes are expressed in viewport widths.
    return new Vector3(second.x - first.x, (second.y - first.y) * this.viewportHeight / this.viewportWidth, second.depth - first.depth)
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
    } else if (!blocked && !recover) {
      const priorSeparation = previous.length()
      if (separation !== priorSeparation) this.scene.zoom(separation / priorSeparation)
    }
    if (recover) this.rebase(grip, first, second)
    grip.pairBlocked = blocked
  }

  private rebase(grip: ActiveGrip, first: HandPose, second: HandPose | null) {
    grip.first = first
    grip.second = second
    grip.pairBlocked = false
    if (!second) {
      grip.joinSide = first.x <= 0.64 ? 1 : -1
      grip.joinVertical = first.y <= 0.5 ? 1 : -1
    }
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

  private movePart(part: PartGrip, first: HandPose, second: HandPose | null, preserveOrientation = false) {
    const point = this.worldPoint(first, part.base)
    if (!second || !part.startSecond) {
      this.scene.setPose(part.id, { ...part.base, position: point.sub(part.startFirst).add(part.base.position) })
      return
    }

    const secondPoint = this.worldPoint(second, part.base)
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
