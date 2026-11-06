import assert from 'node:assert/strict'
import { test } from 'node:test'
import { setTimeout as delay } from 'node:timers/promises'
import { Quaternion, Vector3 } from 'three'
import { FRAME_MAX_AGE_MS, GestureController } from '../src/gestures.ts'
import type { GestureState, HandPose, ObjectPose } from '../src/gestures.ts'

function hand(id: string, x: number, y: number, pinched = false, depth = 0): HandPose {
  return { id, x, y, pinched, depth, depthValid: true }
}

function assertVector(actual: Vector3, expected: Vector3) {
  assert.ok(actual.distanceTo(expected) < 1e-8, `${actual.toArray()} differs from ${expected.toArray()}`)
}

function harness(t: { after: (cleanup: () => void) => void }) {
  const initial = { position: new Vector3(1, 2, 3), rotation: new Quaternion(), scale: 1 }
  let pose = initial
  let state: GestureState
  let sequence = 0
  let selected = 'assembly'
  let hover: string[] = []
  const edits: { id: string; before: ObjectPose }[] = []
  const camera = { orbit: new Vector3(), zoom: 1 }
  const controller = new GestureController({
    pick: (x) => x < 0.85 ? 'wheel' : null,
    select: (id) => { selected = id ?? '' },
    hover: (ids) => { hover = ids },
    getPose: () => pose,
    setPose: (_id, value) => { pose = value },
    screenPoint: (x, y, reference, depth = 0) => new Vector3(x, -y, depth).add(reference),
    orbit: (dx, dy) => camera.orbit.add(new Vector3(dx, dy, 0)),
    zoom: (factor) => { camera.zoom *= factor },
    commitGrip: (id, before) => { edits.push({ id, before }) },
  }, value => { state = value })
  t.after(() => controller.dispose())
  return {
    controller, initial, edits, camera,
    pose: () => pose, state: () => state, hover: () => hover, selected: () => selected,
    send: (...hands: HandPose[]) => controller.receive({ sequence: ++sequence, capturedAt: performance.now(), hands }),
    replay: (hands: HandPose[], age: number, offset: number) => controller.receive({
      sequence: sequence + offset, capturedAt: performance.now() - age, hands,
    }),
  }
}

test('grabs an individual part, rotates and scales with both hands, then rebases and records one undo', t => {
  const h = harness(t)
  const first = hand('first', 0.3, 0.4)
  h.send(first)
  assert.deepEqual(h.hover(), ['wheel'])
  h.send({ ...first, pinched: true })
  assert.equal(h.state().target, 'part')
  assert.equal(h.state().held, 'wheel')
  assert.equal(h.selected(), 'wheel')

  const moved = hand('first', 0.4, 0.5, true, 0.12)
  h.send(moved)
  const translated = new Vector3(1.1, 1.9, 3.12)
  assertVector(h.pose().position, translated)

  const join = h.state().join!
  const second = hand('second', join.x, join.y, false, moved.depth)
  h.send(moved, second)
  h.send(moved, { ...second, fist: true })
  assert.equal(h.state().status, 'holding')
  h.send(moved, { ...second, pinched: true })
  assert.equal(h.state().status, 'holding')
  h.send(moved, second)
  h.send(moved, { ...second, pinched: true })
  assert.equal(h.state().status, 'both')
  assertVector(h.pose().position, translated)

  const midpoint = (moved.x + second.x) / 2
  const distance = second.x - moved.x
  const top = hand('first', midpoint, 0.5 - distance, true, moved.depth)
  const bottom = hand('second', midpoint, 0.5 + distance, true, moved.depth)
  h.send(top, bottom)
  assertVector(h.pose().position, translated)
  assertVector(new Vector3(1, 0, 0).applyQuaternion(h.pose().rotation), new Vector3(0, -1, 0))
  assert.ok(Math.abs(h.pose().scale - 2) < 1e-8)

  const rotated = h.pose()
  h.send({ ...top, pinched: false }, bottom)
  assert.equal(h.state().status, 'holding')
  assert.deepEqual(h.pose(), rotated)
  h.send({ ...bottom, x: bottom.x + 0.1 })
  assertVector(h.pose().position, new Vector3(1.2, 1.9, 3.12))
  assert.deepEqual(h.pose().rotation, rotated.rotation)
  h.send({ ...bottom, pinched: false })
  assert.deepEqual(h.edits, [{ id: 'wheel', before: h.initial }])
  assertVector(h.camera.orbit, new Vector3())
  assert.equal(h.camera.zoom, 1)
})

test('freezes on depth loss and ignores held newcomers, stale frames, and sequence replays', t => {
  const h = harness(t)
  const first = hand('first', 0.3, 0.4)
  h.send(first)
  h.send({ ...first, pinched: true })
  const moved = { ...first, x: 0.5, pinched: true }
  h.send(moved)
  const lastPose = h.pose()
  h.replay([{ ...moved, x: 0.8 }], 0, 0)
  h.replay([{ ...moved, x: 0.8 }], FRAME_MAX_AGE_MS + 1, 20)
  assert.deepEqual(h.pose(), lastPose)

  const stranger = hand('stranger', 0.3, 0.4, true)
  h.send({ ...moved, depthValid: false }, stranger)
  assert.equal(h.state().status, 'lost')
  h.send({ ...stranger, x: 0.6 })
  assert.equal(h.state().held, null)
  assert.deepEqual(h.pose(), lastPose)
  assert.equal(h.edits.length, 1)
  h.send({ ...stranger, pinched: false })
  h.send(stranger)
  assert.equal(h.state().held, 'wheel')
})

test('an empty-space pinch stays inert when it moves onto a part', t => {
  const h = harness(t)
  const empty = hand('first', 0.9, 0.4)
  h.send(empty)
  h.send({ ...empty, pinched: true })
  h.send({ ...empty, pinched: true, x: 0.6 })
  assert.equal(h.state().target, null)
  assert.equal(h.state().held, null)
  assert.equal(h.state().hovered, null)
  assertVector(h.camera.orbit, new Vector3())
  assert.deepEqual(h.pose(), h.initial)
})

test('a fist over a part orbits the view and only a second fist can join for zoom', t => {
  const h = harness(t)
  const first = hand('first', 0.3, 0.4)
  h.send(first)
  h.send({ ...first, fist: true })
  assert.equal(h.state().status, 'holding')
  assert.equal(h.state().target, 'view')
  assert.equal(h.state().held, null)

  const moved = { ...first, x: 0.9, y: 0.5, fist: true }
  h.send(moved)
  assertVector(h.camera.orbit, new Vector3(0.6, 0.1, 0))
  assert.equal(h.state().hovered, null)
  const join = h.state().join!
  const second = hand('second', join.x, join.y)
  h.send(moved, second)
  h.send(moved, { ...second, pinched: true })
  assert.equal(h.state().status, 'holding')
  h.send(moved, second)
  h.send(moved, { ...second, fist: true })
  h.send(moved, { ...second, x: moved.x + 2 * (second.x - moved.x), fist: true })
  assert.equal(h.state().status, 'both')
  assert.equal(h.state().target, 'view')
  assert.ok(Math.abs(h.camera.zoom - 2) < 1e-8)
  h.send()
  assert.equal(h.state().status, 'lost')
  assert.deepEqual(h.pose(), h.initial)
  assert.deepEqual(h.edits, [])
})

test('switching between pinch and fist requires neutral and an ambiguous grip stops movement', t => {
  const h = harness(t)
  const open = hand('first', 0.3, 0.4)
  const pinch = { ...open, pinched: true }
  const fist = { ...open, fist: true }
  h.send(open)
  h.send(pinch)
  h.send({ ...pinch, x: 0.5 })
  const moved = h.pose()
  h.send(fist)
  h.send({ ...fist, x: 0.6 })
  assert.equal(h.state().target, null)
  assert.deepEqual(h.pose(), moved)
  assertVector(h.camera.orbit, new Vector3())

  h.send(open)
  h.send(fist)
  h.send({ ...fist, x: 0.5 })
  assert.equal(h.state().target, 'view')
  h.send({ ...fist, pinched: true })
  assert.equal(h.state().status, 'lost')
  h.send({ ...fist, x: 0.6 })
  assert.equal(h.state().target, null)
  assertVector(h.camera.orbit, new Vector3(0.2, 0, 0))

  h.send(open)
  h.send(fist)
  h.send(pinch)
  assert.equal(h.state().target, null)
  h.send(open)
  h.send(pinch)
  assert.equal(h.state().target, 'part')
})

test('a stalled stream releases the grip and requires a fresh pinch when it resumes', async t => {
  const h = harness(t)
  const first = hand('first', 0.3, 0.4)
  h.send(first)
  h.send({ ...first, pinched: true })
  const moved = { ...first, x: 0.5, pinched: true }
  h.send(moved)
  const lastPose = h.pose()
  await delay(FRAME_MAX_AGE_MS + 25)
  assert.equal(h.state().status, 'lost')
  assert.equal(h.edits.length, 1)
  h.send({ ...moved, x: 0.7 })
  assert.equal(h.state().status, 'lost')
  assert.deepEqual(h.pose(), lastPose)
  h.send({ ...moved, pinched: false })
  h.send(moved)
  assert.equal(h.state().held, 'wheel')
})
