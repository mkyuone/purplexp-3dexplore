import type { HandPose, JoinTarget, PoseFrame } from './gestures.ts'

export type SimHandId = 'left' | 'right'
type Grip = 'part' | 'view'
type GripSource = 'keyboard' | 'pointer' | 'control'

interface SimHand {
  x: number
  y: number
  depth: number
  visible: boolean
  grip: { kind: Grip; sources: Set<GripSource> } | null
}

interface PointerGrip {
  id: SimHandId
  kind: Grip
  touch: boolean
}

const HAND_IDS: SimHandId[] = ['left', 'right']
const MOVE_KEYS: Record<string, [SimHandId, number, number]> = {
  KeyA: ['left', -1, 0], KeyD: ['left', 1, 0], KeyW: ['left', 0, -1], KeyS: ['left', 0, 1],
  ArrowLeft: ['right', -1, 0], ArrowRight: ['right', 1, 0],
  ArrowUp: ['right', 0, -1], ArrowDown: ['right', 0, 1],
}
const DEPTH_KEYS: Record<string, [SimHandId, number]> = {
  KeyQ: ['left', 1], KeyE: ['left', -1], BracketLeft: ['right', 1], BracketRight: ['right', -1],
}
const GRIP_KEYS: Record<string, [SimHandId, Grip]> = {
  Space: ['left', 'part'], Enter: ['right', 'part'], KeyV: ['left', 'view'], KeyB: ['right', 'view'],
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value))
}

function newHand(x: number): SimHand {
  return { x, y: 0.5, depth: 0, visible: false, grip: null }
}

export class HandSimulator {
  private readonly viewport: HTMLElement
  private readonly onFrame: (frame: PoseFrame) => void
  private readonly touchGripAt?: (x: number, y: number) => Grip
  private readonly window: Window
  private readonly document: Document
  private hands = { left: newHand(0.5), right: newHand(0.7) }
  private pointers = new Map<number, PointerGrip>()
  private join: JoinTarget | null = null
  private sequence = 0
  private enabled = false
  private heartbeat: number | undefined

  constructor(viewport: HTMLElement, onFrame: (frame: PoseFrame) => void, touchGripAt?: (x: number, y: number) => Grip) {
    this.viewport = viewport
    this.onFrame = onFrame
    this.touchGripAt = touchGripAt
    this.document = viewport.ownerDocument
    this.window = this.document.defaultView!
    viewport.addEventListener('pointerdown', this.pointerDown)
    viewport.addEventListener('pointermove', this.pointerMove)
    viewport.addEventListener('pointerup', this.pointerUp)
    viewport.addEventListener('pointercancel', this.pointerCancel)
    viewport.addEventListener('lostpointercapture', this.pointerCancel)
    viewport.addEventListener('pointerleave', this.pointerLeave)
    this.window.addEventListener('keydown', this.keyDown)
    this.window.addEventListener('keyup', this.keyUp)
    this.window.addEventListener('blur', this.reset)
    this.document.addEventListener('visibilitychange', this.visibilityChange)
  }

  setEnabled(enabled: boolean) {
    if (this.enabled === enabled) return
    this.enabled = enabled
    if (enabled) {
      this.hands = { left: newHand(0.5), right: newHand(0.7) }
      this.emit()
      this.heartbeat = this.window.setInterval(() => this.emit(), 80)
    } else {
      this.window.clearInterval(this.heartbeat)
      this.heartbeat = undefined
      this.clearHands()
      this.emit('lost')
    }
  }

  setJoinTarget(join: JoinTarget | null) {
    this.join = join
  }

  aimSecondAtJoin() {
    if (!this.enabled || !this.join || this.hands.right.grip) return
    const { x, y } = this.join
    Object.assign(this.hands.right, { x, y, visible: true })
    this.emit()
  }

  moveHand(id: SimHandId, dx: number, dy: number) {
    if (!this.enabled) return
    this.ensureVisible(id)
    const hand = this.hands[id]
    hand.x = clamp(hand.x + dx, 0, 1)
    hand.y = clamp(hand.y + dy, 0, 1)
    this.emit()
  }

  moveDepth(id: SimHandId, delta: number) {
    if (!this.enabled) return
    this.ensureVisible(id)
    const hand = this.hands[id]
    hand.depth = clamp(hand.depth + delta, -0.4, 0.4)
    this.emit()
  }

  setPinch(id: SimHandId, held: boolean) {
    if (this.enabled) this.updateGrip(id, 'part', 'control', held)
  }

  reset = () => {
    if (!this.enabled) return
    this.clearHands()
    this.emit('lost')
  }

  dispose() {
    this.window.clearInterval(this.heartbeat)
    this.viewport.removeEventListener('pointerdown', this.pointerDown)
    this.viewport.removeEventListener('pointermove', this.pointerMove)
    this.viewport.removeEventListener('pointerup', this.pointerUp)
    this.viewport.removeEventListener('pointercancel', this.pointerCancel)
    this.viewport.removeEventListener('lostpointercapture', this.pointerCancel)
    this.viewport.removeEventListener('pointerleave', this.pointerLeave)
    this.window.removeEventListener('keydown', this.keyDown)
    this.window.removeEventListener('keyup', this.keyUp)
    this.window.removeEventListener('blur', this.reset)
    this.document.removeEventListener('visibilitychange', this.visibilityChange)
    this.clearHands()
    this.enabled = false
  }

  private clearHands() {
    const pointers = [...this.pointers.keys()]
    this.pointers.clear()
    for (const id of pointers) {
      if (this.viewport.hasPointerCapture(id)) this.viewport.releasePointerCapture(id)
    }
    this.hands = { left: newHand(0.5), right: newHand(0.7) }
  }

  private emit(tracking?: 'lost') {
    const hands: HandPose[] = []
    for (const id of HAND_IDS) {
      const { x, y, depth, visible, grip } = this.hands[id]
      if (visible) hands.push({ id, x, y, depth, depthValid: true, pinched: grip?.kind === 'part', fist: grip?.kind === 'view' })
    }
    this.onFrame({ sequence: ++this.sequence, capturedAt: performance.now(), hands, tracking })
  }

  private ensureVisible(id: SimHandId) {
    const hand = this.hands[id]
    if (hand.visible) return
    if (id === 'right' && this.join) {
      hand.x = this.join.x
      hand.y = this.join.y
    }
    hand.visible = true
  }

  private updateGrip(id: SimHandId, kind: Grip, source: GripSource, held: boolean) {
    const hand = this.hands[id]
    const grip = hand.grip
    if (held) {
      if (grip && (grip.kind !== kind || grip.sources.has(source))) return
      if (id === 'right' && source !== 'pointer' && this.join && !grip
        && Math.hypot(hand.x - this.join.x, hand.y - this.join.y) > this.join.radius) this.aimSecondAtJoin()
      if (!hand.visible) {
        this.ensureVisible(id)
        this.emit()
      }
      if (grip) grip.sources.add(source)
      else hand.grip = { kind, sources: new Set([source]) }
    } else {
      if (!grip || grip.kind !== kind || !grip.sources.delete(source)) return
      if (!grip.sources.size) hand.grip = null
    }
    this.emit()
  }

  private pointFromEvent(event: PointerEvent) {
    const bounds = this.viewport.getBoundingClientRect()
    if (bounds.width <= 0 || bounds.height <= 0) return null
    const x = (event.clientX - bounds.left) / bounds.width
    const y = (event.clientY - bounds.top) / bounds.height
    if (x < 0 || x > 1 || y < 0 || y > 1) {
      this.reset()
      return null
    }
    return { x, y }
  }

  private pointerDown = (event: PointerEvent) => {
    if (!this.enabled || event.button !== 0 || this.pointers.has(event.pointerId)) return
    if ((event.target as Element | null)?.closest('button, input, select, textarea, a, [data-simulator-control]')) return
    const active = [...this.pointers.values()]
    const used = active.map(pointer => pointer.id)
    const touch = event.pointerType === 'touch'
    const id = touch ? HAND_IDS.find(candidate => !used.includes(candidate)) : 'left'
    if (!id || used.includes(id)) return
    const point = this.pointFromEvent(event)
    if (!point) return
    const kind = touch
      ? active.find(pointer => pointer.touch)?.kind ?? this.touchGripAt?.(point.x, point.y) ?? 'part'
      : event.shiftKey ? 'view' : 'part'
    const hand = this.hands[id]
    if (hand.grip && hand.grip.kind !== kind) return
    event.preventDefault()
    this.viewport.focus({ preventScroll: true })
    this.pointers.set(event.pointerId, { id, kind, touch })
    this.viewport.setPointerCapture(event.pointerId)
    Object.assign(hand, point, { visible: true })
    this.emit()
    this.updateGrip(id, kind, 'pointer', true)
  }

  private pointerMove = (event: PointerEvent) => {
    if (!this.enabled) return
    const pointer = this.pointers.get(event.pointerId)
    if (!pointer && (event.pointerType === 'touch' || this.hasLeftPointer())) return
    const point = this.pointFromEvent(event)
    if (!point) return
    Object.assign(this.hands[pointer?.id ?? 'left'], point, { visible: true })
    this.emit()
  }

  private pointerUp = (event: PointerEvent) => {
    const pointer = this.pointers.get(event.pointerId)
    if (!pointer) return
    this.pointers.delete(event.pointerId)
    const { id, kind, touch } = pointer
    const point = this.pointFromEvent(event)
    if (!point) {
      this.updateGrip(id, kind, 'pointer', false)
      return
    }
    Object.assign(this.hands[id], point, { visible: true })
    this.updateGrip(id, kind, 'pointer', false)
    if (touch && !this.hands[id].grip) {
      this.hands[id].visible = false
      this.emit()
    }
  }

  private pointerCancel = (event: PointerEvent) => {
    if (this.pointers.has(event.pointerId)) this.reset()
  }

  private pointerLeave = (event: PointerEvent) => {
    if (!this.enabled || event.pointerType === 'touch' || this.hasLeftPointer()) return
    this.hands.left.visible = false
    this.hands.left.grip = null
    this.emit()
  }

  private hasLeftPointer() {
    return [...this.pointers.values()].some(pointer => pointer.id === 'left')
  }

  private keyDown = (event: KeyboardEvent) => {
    if (!this.enabled || event.altKey || event.ctrlKey || event.metaKey) return
    const focus = this.document.activeElement
    if (focus !== this.document.body && focus !== this.viewport && focus !== this.document.documentElement) return
    const { code } = event
    const move = MOVE_KEYS[code]
    const depth = DEPTH_KEYS[code]
    const grip = GRIP_KEYS[code]
    const step = event.shiftKey ? 0.065 : 0.025
    if (move) {
      const [id, x, y] = move
      this.moveHand(id, x * step, y * step)
    } else if (depth) {
      this.moveDepth(depth[0], depth[1] * step)
    } else if (grip) {
      if (!event.repeat) this.updateGrip(grip[0], grip[1], 'keyboard', true)
    } else if (code === 'Escape') {
      if (!event.repeat) this.reset()
    } else {
      return
    }
    event.preventDefault()
  }

  private keyUp = (event: KeyboardEvent) => {
    if (!this.enabled) return
    const grip = GRIP_KEYS[event.code]
    if (grip) this.updateGrip(grip[0], grip[1], 'keyboard', false)
  }

  private visibilityChange = () => {
    if (this.document.hidden) this.reset()
  }
}
