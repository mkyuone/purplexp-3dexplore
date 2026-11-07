import assert from 'node:assert/strict'
import { test } from 'node:test'
import { setTimeout as delay } from 'node:timers/promises'
import { Quaternion, Vector3 } from 'three'
import { FRAME_MAX_AGE_MS, GestureController, isInsideJoin, LIVE_PINCH_CONTINUITY_MS, RESET_HOLD_MS } from '../src/gestures.ts'
import type { GestureState, HandPose, ObjectPose } from '../src/gestures.ts'
import { CALIBRATION_INSET_X, CALIBRATION_INSET_Y } from '../src/reach.ts'

function hand(id: string, x: number, y: number, pinched = false, depth = 0): HandPose {
  return { id, x, y, pinched, depth, depthValid: true }
}

function assertVector(actual: Vector3, expected: Vector3) {
  assert.ok(actual.distanceTo(expected) < 1e-8, `${actual.toArray()} differs from ${expected.toArray()}`)
}

function harness(t: { after: (cleanup: () => void) => void }, options: {
  joinAnywhere?: boolean
  liveTracking?: boolean
  pick?: (x: number, y: number) => string | null
  onReset?: () => void
} = {}) {
  const initial = { position: new Vector3(1, 2, 3), rotation: new Quaternion(), scale: 1 }
  let pose = initial
  let state: GestureState
  let sequence = 0
  let selected = 'assembly'
  let hover: string[] = []
  const edits: { id: string; before: ObjectPose }[] = []
  const camera = { orbit: new Vector3(), pan: new Vector3(), zoom: 1 }
  const controller = new GestureController({
    pick: options.pick ?? ((x) => x < 0.85 ? 'wheel' : null),
    select: (id) => { selected = id ?? '' },
    hover: (ids) => { hover = ids },
    getPose: () => pose,
    setPose: (_id, value) => { pose = value },
    screenPoint: (x, y, reference, depth = 0) => new Vector3(x, -y, depth).add(reference),
    orbit: (dx, dy) => camera.orbit.add(new Vector3(dx, dy, 0)),
    pan: (dx, dy) => camera.pan.add(new Vector3(dx, dy, 0)),
    zoom: (factor) => { camera.zoom *= factor },
    commitGrip: (id, before) => { edits.push({ id, before }) },
  }, value => { state = value }, options)
  t.after(() => controller.dispose())
  return {
    controller, initial, edits, camera,
    pose: () => pose, state: () => state, hover: () => hover, selected: () => selected,
    send: (...hands: HandPose[]) => controller.receive({ sequence: ++sequence, capturedAt: performance.now(), hands }),
    sendAged: (age: number, ...hands: HandPose[]) => controller.receive({
      sequence: ++sequence, capturedAt: performance.now() - age, hands,
    }),
    sendLost: (age = 0) => controller.receive({
      sequence: ++sequence, capturedAt: performance.now() - age, hands: [], tracking: 'lost',
    }),
    resetSequence: () => { sequence = 0 },
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

test('joinAnywhere lets a second pinch join off-ring without jumping the held part', t => {
  const h = harness(t, { joinAnywhere: true })
  const open = hand('first', 0.3, 0.4)
  h.send(open)
  h.send({ ...open, pinched: true })
  const held = { ...open, x: 0.4, y: 0.5, depth: 0.1, pinched: true }
  h.send(held)
  const beforeJoin = h.pose()
  const second = hand('second', 0.08, 0.16, false, -0.1)
  assert.equal(isInsideJoin(h.state().join!, second.x, second.y), false)
  h.send(held, second)
  h.send(held, { ...second, pinched: true })
  assert.equal(h.state().status, 'both')
  assert.equal(h.state().held, 'wheel')
  assert.deepEqual(h.pose(), beforeJoin)

  const move = (point: HandPose): HandPose => ({ ...point, x: point.x + 0.05, y: point.y + 0.02, depth: point.depth + 0.04, pinched: true })
  const movedFirst = move(held)
  const movedSecond = move(second)
  h.send(movedFirst, movedSecond)
  assertVector(h.pose().position, beforeJoin.position.clone().add(new Vector3(0.05, -0.02, 0.04)))
  assert.ok(h.pose().rotation.angleTo(beforeJoin.rotation) < 1e-8)
  assert.ok(Math.abs(h.pose().scale - beforeJoin.scale) < 1e-8)

  const beforeRelease = h.pose()
  h.send({ ...movedFirst, pinched: false }, movedSecond)
  assert.deepEqual(h.pose(), beforeRelease)
  h.send({ ...movedSecond, pinched: false })
  assert.equal(h.edits.length, 1)
})

test('joinAnywhere still requires neutral, matching pinch and usable hand separation for a part', t => {
  const h = harness(t, { joinAnywhere: true })
  const open = hand('first', 0.3, 0.4)
  const held = { ...open, pinched: true }
  const second = hand('second', 0.08, 0.16)
  h.send(open)
  h.send(held)
  h.send(held, { ...second, pinched: true })
  assert.equal(h.state().status, 'holding')
  h.send(held, second)
  h.send(held, { ...second, fist: true })
  assert.equal(h.state().status, 'holding')
  h.send(held, { ...second, pinched: true })
  assert.equal(h.state().status, 'holding')

  const near = hand('second', 0.31, 0.41)
  h.send(held, near)
  h.send(held, { ...near, pinched: true })
  assert.equal(h.state().status, 'holding')
  h.send(held, second)
  h.send(held, { ...second, pinched: true })
  assert.equal(h.state().status, 'both')
  assert.equal(h.state().target, 'part')
  assert.equal(h.camera.zoom, 1)
  assertVector(h.camera.orbit, new Vector3())
  assert.deepEqual(h.pose(), h.initial)
})

test('freezes when the gripping hand is missing and ignores held newcomers, stale frames, and sequence replays', t => {
  let now = 1000
  t.mock.method(performance, 'now', () => now)
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
  h.send(stranger)
  assert.equal(h.state().status, 'holding')
  h.send({ ...stranger, x: 0.6 })
  assert.equal(h.state().held, 'wheel')
  now += 151
  h.send(stranger)
  assert.equal(h.state().status, 'lost')
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

test('camera latency leaves a closing pinch time to acquire the recent neutral target', t => {
  let now = 1000
  t.mock.method(performance, 'now', () => now)
  const h = harness(t)
  const open = hand('first', 0.849, 0.4)
  h.sendAged(170, open)
  const pinch = { ...open, x: 0.86, pinched: true }
  now += 60
  h.sendAged(170, pinch)
  assert.equal(h.state().held, 'wheel')
  assert.deepEqual(h.pose(), h.initial)
  h.sendAged(170, { ...pinch, x: 0.88 })
  assertVector(h.pose().position, h.initial.position.clone().add(new Vector3(0.02, 0, 0)))
})

test('an exact pinch hit wins over a recent target from the same hand', t => {
  const h = harness(t, { pick: x => x < 0.85 ? 'wheel' : 'axle' })
  h.send(hand('first', 0.849, 0.4))
  h.send(hand('first', 0.86, 0.4, true))
  assert.equal(h.state().held, 'axle')
  assert.equal(h.selected(), 'axle')
})

test('recent target acquisition is limited by capture age and circular viewport pixels', async t => {
  let now = 1000
  t.mock.method(performance, 'now', () => now)
  const cases = [
    { name: '27 horizontal pixels on a wide viewport', width: 1600, height: 900, dx: 27, dy: 0, age: 199, held: 'wheel' },
    { name: '27 vertical pixels on a wide viewport', width: 1600, height: 900, dx: 0, dy: 27, age: 199, held: 'wheel' },
    { name: '27 vertical pixels on a tall viewport', width: 900, height: 1600, dx: 0, dy: 27, age: 199, held: 'wheel' },
    { name: '29 horizontal pixels', width: 1600, height: 900, dx: 29, dy: 0, age: 199, held: null },
    { name: '29 vertical pixels', width: 1600, height: 900, dx: 0, dy: 29, age: 199, held: null },
    { name: 'diagonal travel exceeds the radius', width: 1600, height: 900, dx: 20, dy: 20, age: 199, held: null },
    { name: 'the target is older than 200 ms', width: 1600, height: 900, dx: 2, dy: 0, age: 201, held: null },
  ]
  for (const sample of cases) {
    await t.test(sample.name, subtest => {
      const h = harness(subtest, { pick: (x, y) => x === 0.5 && y === 0.5 ? 'wheel' : null })
      h.controller.setViewport(sample.width, sample.height)
      h.send(hand('first', 0.5, 0.5))
      now += sample.age
      h.send(hand('first', 0.5 + sample.dx / sample.width, 0.5 + sample.dy / sample.height, true))
      assert.equal(h.state().held, sample.held)
    })
  }
})

test('recent target acquisition clears on cancellation, replacement, loss and leaving the viewport', async t => {
  const open = hand('first', 0.849, 0.4)
  const miss = { ...open, x: 0.86 }
  const interruptions: { name: string; interrupt: (h: ReturnType<typeof harness>) => void }[] = [
    { name: 'cancellation', interrupt: h => h.controller.cancel() },
    { name: 'stream replacement', interrupt: h => h.controller.resetStream() },
    { name: 'viewport resize', interrupt: h => h.controller.setViewport(1600, 900) },
    { name: 'missing hand', interrupt: h => h.send() },
    { name: 'malformed depth', interrupt: h => h.send({ ...open, depth: NaN, depthValid: false }) },
    { name: 'outside viewport', interrupt: h => h.send({ ...open, x: 1.01 }) },
    { name: 'new hand ID', interrupt: h => h.send({ ...miss, id: 'other' }) },
    { name: 'travel beyond the tolerance', interrupt: h => h.send({ ...open, x: 0.9 }) },
  ]
  for (const sample of interruptions) {
    await t.test(sample.name, subtest => {
      const h = harness(subtest)
      h.send(open)
      sample.interrupt(h)
      h.send(miss)
      h.send({ ...miss, pinched: true })
      assert.equal(h.state().held, null)
    })
  }
})

test('one fist orbits and zooms while an unarmed second fist leaves ownership alone', t => {
  const h = harness(t)
  const first = hand('first', 0.3, 0.4)
  h.send(first)
  const fist = { ...first, fist: true }
  h.send(fist)
  assert.equal(h.state().status, 'holding')
  assert.equal(h.state().target, 'view')
  assert.equal(h.state().held, null)
  assert.equal(h.state().join, null)

  h.send({ ...fist, depth: 0.003 })
  h.send({ ...fist, depth: -0.003 })
  assert.equal(h.camera.zoom, 1)

  const second = { ...hand('second', 0.8, 0.4), fist: true }
  const pulled = { ...fist, x: 0.5, y: 0.5, depth: 0.106 }
  h.send(pulled, second)
  const expectedOrbit = new Vector3(0.2, 0.1, 0)
  assertVector(h.camera.orbit, expectedOrbit.setLength(expectedOrbit.length() - 0.005))
  assert.ok(Math.abs(h.camera.zoom - Math.exp(0.2)) < 1e-8)
  assert.equal(h.state().status, 'holding')
  h.send(pulled, { ...second, x: 0.95, depth: 0.2, depthValid: false })
  assert.ok(Math.abs(h.camera.zoom - Math.exp(0.2)) < 1e-8)

  h.send({ ...pulled, depth: -0.006 }, second)
  assert.ok(Math.abs(h.camera.zoom - 1) < 1e-8)
  h.send(first, second)
  assert.equal(h.state().target, null)
  h.send(second)
  assert.equal(h.state().target, null)
  assert.deepEqual(h.pose(), h.initial)
})

test('two fists pan by their midpoint and one fist resumes orbit and zoom without jumping', t => {
  const h = harness(t)
  const first = hand('first', 0.25, 0.4)
  const second = hand('second', 0.65, 0.4)
  h.send(first, second)
  const left = { ...first, fist: true }
  const right = { ...second, fist: true }
  h.send(left, right)
  assert.equal(h.state().status, 'both')
  assert.equal(h.state().target, 'view')
  assert.deepEqual(h.state().activeHands, ['first', 'second'])

  h.send({ ...left, x: 0.2, depth: 0.4 }, { ...right, x: 0.7, depth: -0.4 })
  assertVector(h.camera.pan, new Vector3())
  h.send({ ...left, x: 0.3, y: 0.32, depth: 0.4 }, { ...right, x: 0.8, y: 0.32, depth: 0.3 })
  assertVector(h.camera.pan, new Vector3(0.1, -0.08, 0))
  assertVector(h.camera.orbit, new Vector3())
  assert.equal(h.camera.zoom, 1)

  const remaining = { ...left, x: 0.5, y: 0.32, depth: 0.2 }
  h.send(remaining, { ...right, fist: false, pinched: true })
  assert.equal(h.state().status, 'holding')
  assert.equal(h.state().target, 'view')
  assert.deepEqual(h.state().activeHands, ['first'])
  assertVector(h.camera.orbit, new Vector3())
  assert.equal(h.camera.zoom, 1)
  h.send({ ...remaining, x: 0.58, depth: 0.306 })
  assertVector(h.camera.orbit, new Vector3(0.075, 0, 0))
  assert.ok(Math.abs(h.camera.zoom - Math.exp(0.2)) < 1e-8)
  assertVector(h.camera.pan, new Vector3(0.1, -0.08, 0))
  h.send({ ...remaining, fist: false, pinched: true })
  assert.equal(h.state().target, null)
  assert.deepEqual(h.pose(), h.initial)
})

test('a live second fist confirms, drops out, and rejoins an RGB-only first fist without a camera jump', t => {
  let now = 1000
  t.mock.method(performance, 'now', () => now)
  const h = harness(t, { liveTracking: true })
  const open = { ...hand('first', 0.3, 0.4), depthValid: false }
  const first = { ...open, fist: true }
  const second = { ...hand('second', 0.65, 0.4, false, 0.4), fist: true }
  h.send(open)
  h.send(first)
  h.send(first, second)
  now += 99
  h.send(first, second)
  assert.equal(h.state().status, 'holding')
  now += 1
  h.send(first, second)
  assert.equal(h.state().status, 'both')
  h.send({ ...first, x: 0.35 }, { ...second, x: 0.7, depth: -0.4 })
  assertVector(h.camera.pan, new Vector3(0.05, 0, 0))
  assert.equal(h.camera.zoom, 1)

  const survivor = { ...first, x: 0.45 }
  h.send(survivor)
  assert.deepEqual(h.state().activeHands, ['first'])
  assertVector(h.camera.orbit, new Vector3())
  const moved = { ...survivor, x: 0.5 }
  h.send(moved)
  assertVector(h.camera.orbit, new Vector3(0.045, 0, 0))
  now += 50
  h.send(moved, second)
  assert.deepEqual(h.state().activeHands, ['first', 'second'])
  assertVector(h.camera.pan, new Vector3(0.05, 0, 0))
  assertVector(h.camera.orbit, new Vector3(0.045, 0, 0))
  h.send({ ...moved, y: 0.5 }, { ...second, y: 0.5 })
  assertVector(h.camera.pan, new Vector3(0.05, 0.1, 0))
  assert.equal(h.camera.zoom, 1)
})

test('two already-closed fists confirm together when live tracking starts', t => {
  let now = 1000
  t.mock.method(performance, 'now', () => now)
  const h = harness(t, { liveTracking: true })
  const first = { ...hand('first', 0.3, 0.4), fist: true }
  const second = { ...hand('second', 0.7, 0.4), fist: true }
  h.send(second, first)
  now += 99
  h.send(first, second)
  assert.equal(h.state().target, null)
  now += 1
  h.send(second, first)
  assert.equal(h.state().status, 'both')
  assert.equal(h.state().target, 'view')
  assert.deepEqual(h.state().activeHands, ['second', 'first'])
  assertVector(h.camera.pan, new Vector3())
  assertVector(h.camera.orbit, new Vector3())
})

test('a delayed camera loss report leaves time to rebase at the frozen view', t => {
  let now = 1000
  t.mock.method(performance, 'now', () => now)
  const h = harness(t)
  const first = hand('first', 0.3, 0.4)
  h.sendAged(170, first)
  const fist = { ...first, fist: true }
  h.sendAged(170, fist)
  h.sendAged(170, { ...fist, x: 0.4, depth: 0.106 })
  const frozenOrbit = h.camera.orbit.clone()
  const frozenZoom = h.camera.zoom

  now += 35
  h.sendLost(170)
  assert.equal(h.state().status, 'holding')
  assert.deepEqual(h.state().hands, [])
  now += 100
  const returned = { ...fist, x: 0.7, y: 0.6, depth: -0.2 }
  h.sendAged(170, returned)
  assertVector(h.camera.orbit, frozenOrbit)
  assert.equal(h.camera.zoom, frozenZoom)

  h.sendAged(170, { ...returned, x: 0.75, depth: -0.094 })
  assertVector(h.camera.orbit, frozenOrbit.clone().add(new Vector3(0.045, 0, 0)))
  assert.ok(Math.abs(h.camera.zoom - frozenZoom * Math.exp(0.2)) < 1e-8)
})

test('a held fist filters position jitter and still responds to slow deliberate orbit', t => {
  const h = harness(t)
  const open = hand('first', 0.3, 0.4)
  const fist = { ...open, fist: true }
  h.send(open)
  h.send(fist)
  h.send({ ...fist, x: 0.303, y: 0.402 })
  h.send({ ...fist, x: 0.297, y: 0.398 })
  assertVector(h.camera.orbit, new Vector3())
  for (const x of [0.302, 0.304, 0.306, 0.308, 0.310]) h.send({ ...fist, x })
  assertVector(h.camera.orbit, new Vector3(0.005, 0, 0))
})

test('a part stays frozen through loss, rebases on return, and releases on an open hand', t => {
  const h = harness(t)
  const open = hand('first', 0.3, 0.4)
  h.send(open)
  const pinch = { ...open, pinched: true }
  h.send(pinch)
  h.send({ ...pinch, x: 0.4 })
  const frozen = h.pose()

  h.sendLost()
  assert.equal(h.state().held, 'wheel')
  assert.equal(h.edits.length, 0)
  const returned = { ...pinch, x: 0.7, y: 0.5, depth: 0.2 }
  h.send(returned)
  assert.deepEqual(h.pose(), frozen)

  h.send({ ...returned, x: 0.75 })
  assertVector(h.pose().position, frozen.position.clone().add(new Vector3(0.05, 0, 0)))
  h.sendLost()
  h.send({ ...returned, pinched: false })
  assert.equal(h.state().held, null)
  assert.equal(h.edits.length, 1)
})

test('an RGB-only pinch moves in XY, rebases through depth changes and releases after tracking loss', t => {
  const h = harness(t, { liveTracking: true })
  const open = { ...hand('first', 0.3, 0.4), depthValid: false }
  const pinch = { ...open, pinched: true }
  const expected = (x: number, y: number, z: number) => h.initial.position.clone().add(new Vector3(x, y, z))
  h.send(open)
  assert.equal(h.state().hovered, 'wheel')
  h.send(pinch)
  h.send({ ...pinch, x: 0.4, y: 0.5 })
  assertVector(h.pose().position, expected(0.1, -0.1, 0))

  const measured = { ...pinch, x: 0.5, y: 0.5, depthValid: true, depth: 0.45 }
  h.send(measured)
  assertVector(h.pose().position, expected(0.2, -0.1, 0))
  h.send({ ...measured, depth: 0.6 })
  assertVector(h.pose().position, expected(0.2, -0.1, 0.15))

  const rgbAgain = { ...pinch, x: 0.6, y: 0.5 }
  h.send(rgbAgain)
  assertVector(h.pose().position, expected(0.3, -0.1, 0.15))
  h.sendLost()
  assert.deepEqual(h.state().activeHands, [])
  h.send({ ...rgbAgain, x: 0.7 })
  assert.deepEqual(h.state().activeHands, ['first'])
  assertVector(h.pose().position, expected(0.3, -0.1, 0.15))
  h.send({ ...rgbAgain, x: 0.75 })
  assertVector(h.pose().position, expected(0.35, -0.1, 0.15))

  h.sendLost()
  h.send({ ...open, x: 0.75 })
  assert.equal(h.state().held, null)
  assert.equal(h.edits.length, 1)
  h.send({ ...pinch, x: 0.75 })
  assert.equal(h.state().held, 'wheel')
  h.send({ ...open, x: 0.75, depthValid: true, depth: -0.2 })
  assert.equal(h.state().held, null)
  assert.equal(h.edits.length, 2)
  assertVector(h.pose().position, expected(0.35, -0.1, 0.15))
})

test('an RGB-only fist keeps orbiting while depth zoom pauses and rebases', t => {
  const h = harness(t, { liveTracking: true })
  const open = { ...hand('first', 0.3, 0.4), depthValid: false }
  const fist = { ...open, fist: true }
  h.send(open)
  h.send(fist)
  h.send({ ...fist, x: 0.5 })
  assert.equal(h.state().target, 'view')
  assertVector(h.camera.orbit, new Vector3(0.195, 0, 0))
  assert.equal(h.camera.zoom, 1)

  const measured = { ...fist, x: 0.6, depthValid: true, depth: 0.4 }
  h.send(measured)
  assertVector(h.camera.orbit, new Vector3(0.295, 0, 0))
  assert.equal(h.camera.zoom, 1)
  h.send({ ...measured, depth: 0.506 })
  assert.ok(Math.abs(h.camera.zoom - Math.exp(0.2)) < 1e-8)

  h.send({ ...fist, x: 0.7 })
  assertVector(h.camera.orbit, new Vector3(0.395, 0, 0))
  assert.ok(Math.abs(h.camera.zoom - Math.exp(0.2)) < 1e-8)
  h.sendLost()
  const returned = { ...measured, x: 0.8, depth: -0.3 }
  h.send(returned)
  assertVector(h.camera.orbit, new Vector3(0.395, 0, 0))
  assert.ok(Math.abs(h.camera.zoom - Math.exp(0.2)) < 1e-8)
  h.send({ ...returned, x: 0.9, depth: -0.194 })
  assertVector(h.camera.orbit, new Vector3(0.49, 0, 0))
  assert.ok(Math.abs(h.camera.zoom - Math.exp(0.4)) < 1e-8)
})

test('a mixed-depth pair scales and rotates in XY and restores spatial depth without a jump', t => {
  const h = harness(t, { joinAnywhere: true })
  const open = { ...hand('first', 0.3, 0.4), depthValid: false }
  const pinch = { ...open, pinched: true }
  const secondOpen = hand('second', 0.6, 0.4, false, 0.45)
  h.send(open, secondOpen)
  h.send(pinch, secondOpen)
  h.send(pinch, { ...secondOpen, pinched: true })
  assert.equal(h.state().status, 'both')

  const left = { ...pinch, x: 0.25, y: 0.5 }
  const right = { ...secondOpen, x: 0.65, y: 0.5, pinched: true, depth: 0.8 }
  h.send(left, right)
  const flat = h.pose()
  assertVector(flat.position, h.initial.position.clone().add(new Vector3(0, -0.1, 0)))
  assert.ok(Math.abs(flat.scale - 4 / 3) < 1e-8)

  const measured = { ...left, depthValid: true, depth: -0.15 }
  h.send(measured, right)
  assert.deepEqual(h.pose(), flat)
  h.send({ ...measured, depth: -0.05 }, { ...right, depth: 0.9 })
  const translated = h.pose()
  assertVector(translated.position, flat.position.clone().add(new Vector3(0, 0, 0.1)))
  assert.ok(Math.abs(translated.scale - flat.scale) < 1e-8)

  const top = { ...measured, x: 0.45, y: 0.3, depth: 0.3 }
  const bottom = { ...right, x: 0.45, y: 0.7, depthValid: false, depth: 0 }
  h.send(top, bottom)
  const rotated = h.pose()
  assertVector(rotated.position, translated.position)
  assertVector(new Vector3(1, 0, 0).applyQuaternion(rotated.rotation), new Vector3(0, -1, 0))
  assert.ok(Math.abs(rotated.scale - translated.scale) < 1e-8)
  h.send({ ...top, depthValid: false, depth: 0 }, { ...bottom, depthValid: true, depth: 0.8 })
  assert.deepEqual(h.pose(), rotated)
  h.send(top, { ...bottom, depthValid: true, depth: 0.8 })
  assert.deepEqual(h.pose(), rotated)
})

test('a pair separated only in depth preserves scale when it switches to XY', t => {
  const h = harness(t, { joinAnywhere: true })
  const first = hand('first', 0.4, 0.4, false, -0.15)
  const second = hand('second', 0.4, 0.4, false, 0.15)
  h.send(first, second)
  h.send({ ...first, pinched: true }, second)
  h.send({ ...first, pinched: true }, { ...second, pinched: true })
  assert.equal(h.state().status, 'both')
  const rgb = { ...first, pinched: true, depthValid: false, depth: 0 }
  const moved = { ...second, pinched: true, x: 0.6 }
  h.send(rgb, moved)
  assertVector(h.pose().position, h.initial.position.clone().add(new Vector3(0.1, 0, 0)))
  assert.equal(h.pose().scale, 1)
  h.send(rgb, { ...moved, x: 0.8 })
  assert.ok(Math.abs(h.pose().scale - 2) < 1e-8)
})

test('a live hand returning with the same ID can resume past the simulator grace', t => {
  let now = 1000
  t.mock.method(performance, 'now', () => now)
  const h = harness(t, { liveTracking: true })
  const open = hand('first', 0.3, 0.4)
  const pinch = { ...open, pinched: true }
  h.send(open)
  h.send(pinch)
  h.send({ ...pinch, x: 0.4 })
  const frozen = h.pose()
  h.sendLost()

  now += 190
  const returned = { ...pinch, x: 0.7, depth: 0.2 }
  h.send(returned)
  assert.equal(h.state().held, 'wheel')
  assert.deepEqual(h.pose(), frozen)
  assert.equal(h.edits.length, 0)
  h.send({ ...returned, x: 0.75 })
  assertVector(h.pose().position, frozen.position.clone().add(new Vector3(0.05, 0, 0)))
})

test('a nearby renamed pinch survives a quiet 260 ms gap and keeps the part outside its hit target', t => {
  let now = 1000
  t.mock.method(performance, 'now', () => now)
  t.mock.timers.enable({ apis: ['setTimeout'] })
  const h = harness(t, { liveTracking: true })
  const open = hand('first', 0.8, 0.4)
  const pinch = { ...open, pinched: true }
  h.send(open)
  h.send(pinch)
  h.send({ ...pinch, x: 0.84 })
  const frozen = h.pose()

  now += 260
  t.mock.timers.tick(260)
  assert.equal(h.state().held, 'wheel')
  assert.equal(h.edits.length, 0)
  const renamed = { ...pinch, id: 'new', x: 0.88, depthValid: false }
  h.send(renamed)
  assert.deepEqual(h.state().activeHands, ['new'])
  assert.deepEqual(h.pose(), frozen)
  h.send({ ...renamed, x: 0.93 })
  assertVector(h.pose().position, frozen.position.clone().add(new Vector3(0.05, 0, 0)))
  h.send({ ...renamed, pinched: false })
  assert.equal(h.state().held, null)
  assert.equal(h.edits.length, 1)
})

test('a late loss report cannot extend the live pinch beyond its last reliable capture', t => {
  let now = 1000
  t.mock.method(performance, 'now', () => now)
  const h = harness(t, { liveTracking: true })
  const open = hand('first', 0.8, 0.4)
  const pinch = { ...open, pinched: true }
  h.sendAged(170, open)
  h.sendAged(170, pinch)
  now += 100
  h.sendLost()
  assert.equal(h.state().held, 'wheel')
  now += LIVE_PINCH_CONTINUITY_MS - 270
  h.send({ ...pinch, x: 0.86 })
  assert.equal(h.state().held, null)
  assert.equal(h.edits.length, 1)
  assert.deepEqual(h.pose(), h.initial)
})

test('a known nearby hand or two competing new IDs cannot take over a lost pinch', t => {
  let now = 1000
  t.mock.method(performance, 'now', () => now)
  const h = harness(t, { liveTracking: true, joinAnywhere: true })
  const open = hand('first', 0.4, 0.4)
  const other = hand('other', 0.45, 0.4)
  const pinch = { ...open, pinched: true }
  h.send(open, other)
  h.send(pinch, other)
  now += 40
  h.sendLost()
  h.send({ ...other, pinched: true })
  assert.deepEqual(h.state().activeHands, [])
  now += 40
  h.send({ ...pinch, id: 'new-one', x: 0.37 }, { ...pinch, id: 'new-two', x: 0.44 })
  assert.deepEqual(h.state().activeHands, [])
  assert.equal(h.state().held, 'wheel')
  h.send(open)
  assert.equal(h.state().held, null)
  assert.equal(h.edits.length, 1)
})

test('a fresh live connection confirms a closed exact target and a reconnect rebases with a new ID', t => {
  let now = 1000
  t.mock.method(performance, 'now', () => now)
  const h = harness(t, { liveTracking: true })
  const missed = hand('first', 0.9, 0.4, true)
  h.send(missed)
  now += 120
  h.send(missed)
  assert.equal(h.state().held, null)

  const aimed = { ...missed, x: 0.4 }
  h.send(aimed)
  now += 99
  h.send(aimed)
  assert.equal(h.state().held, null)
  now += 1
  h.send(aimed)
  assert.equal(h.state().held, 'wheel')
  assert.deepEqual(h.state().activeHands, ['first'])
  h.send({ ...aimed, x: 0.5 })
  const frozen = h.pose()

  h.controller.resetStream()
  const returned = { ...aimed, id: 'new-id', x: 0.7, depth: 0.2 }
  h.resetSequence()
  h.send(returned)
  assert.equal(h.state().held, null)
  now += 100
  h.send(returned)
  assert.equal(h.state().held, 'wheel')
  assert.deepEqual(h.state().activeHands, ['new-id'])
  assert.deepEqual(h.pose(), frozen)
  assert.equal(h.edits.length, 1)
})

test('an explicit live cancel or nonrecovering reset still requires a neutral hand', t => {
  let now = 1000
  t.mock.method(performance, 'now', () => now)
  const h = harness(t, { liveTracking: true })
  const open = hand('first', 0.3, 0.4)
  const closed = { ...open, pinched: true }
  for (const cancel of [() => h.controller.cancel(), () => h.controller.resetStream(false)]) {
    h.send(open)
    h.send(closed)
    cancel()
    h.send(closed)
    now += 120
    h.send(closed)
    assert.equal(h.state().held, null)
    h.send(open)
    h.send(closed)
    assert.equal(h.state().held, 'wheel')
  }
})

test('a fresh live fist confirms steady aim before it owns the view', t => {
  let now = 1000
  t.mock.method(performance, 'now', () => now)
  const h = harness(t, { liveTracking: true })
  const fist = { ...hand('first', 0.3, 0.4), fist: true }
  h.send(fist)
  now += 80
  const moved = { ...fist, x: 0.4 }
  h.send(moved)
  now += 99
  h.send(moved)
  assert.equal(h.state().target, null)
  now += 1
  h.send(moved)
  assert.equal(h.state().target, 'view')
  assert.deepEqual(h.state().activeHands, ['first'])
  assertVector(h.camera.orbit, new Vector3())
  assert.equal(h.camera.zoom, 1)
})

test('a closed hand after a live tracking loss has to confirm a fresh exact target', t => {
  let now = 1000
  t.mock.method(performance, 'now', () => now)
  const h = harness(t, { liveTracking: true, pick: x => x < 0.45 ? 'wheel' : x < 0.8 ? 'axle' : null })
  const closed = hand('first', 0.3, 0.4, true)
  h.send({ ...closed, pinched: false })
  h.send(closed)
  h.send({ ...closed, x: 0.4 })
  const frozen = h.pose()
  h.sendLost()
  now += LIVE_PINCH_CONTINUITY_MS + 1
  const returned = hand('new-id', 0.9, 0.4, true)
  const otherOpen = hand('other', 0.92, 0.4)
  h.send(returned, otherOpen)
  now += 120
  h.send(returned, otherOpen)
  assert.equal(h.state().held, null)
  assert.deepEqual(h.pose(), frozen)

  const aimed = { ...returned, x: 0.6 }
  h.send(aimed, otherOpen)
  now += 99
  h.send(aimed, otherOpen)
  assert.equal(h.state().held, null)
  assert.deepEqual(h.state().activeHands, [])
  now += 1
  h.send(aimed, { ...otherOpen, pinched: true })
  assert.equal(h.state().held, 'axle')
  assert.deepEqual(h.state().activeHands, ['new-id'])
  assert.equal(h.edits.length, 1)
  assert.deepEqual(h.pose(), frozen)
})

test('a neutral survivor does not prevent a privately confirming pinch from returning', t => {
  let now = 1000
  t.mock.method(performance, 'now', () => now)
  const h = harness(t, { liveTracking: true })
  const open = hand('old', 0.3, 0.4)
  h.send(open)
  h.send({ ...open, pinched: true })
  h.sendLost()
  now += LIVE_PINCH_CONTINUITY_MS + 1
  const survivor = hand('other', 0.95, 0.4)
  h.send(survivor)
  const returned = hand('new-id', 0.6, 0.4, true)
  h.send(survivor, returned)
  now += 100
  h.send(returned, survivor)
  assert.equal(h.state().held, 'wheel')
  assert.deepEqual(h.state().activeHands, ['new-id'])
  assert.deepEqual(h.pose(), h.initial)
  assert.equal(h.edits.length, 1)
})

test('two closed live pinches on the same part acquire a stable primary before the other joins', t => {
  let now = 1000
  t.mock.method(performance, 'now', () => now)
  const h = harness(t, { liveTracking: true, joinAnywhere: true })
  const first = hand('alpha', 0.3, 0.4, true)
  const second = hand('beta', 0.6, 0.4, true)
  h.send(second, first)
  now += 100
  h.send(first, second)
  assert.equal(h.state().status, 'holding')
  assert.deepEqual(h.state().activeHands, ['alpha'])
  assert.deepEqual(h.pose(), h.initial)
  h.send(second, first)
  now += 100
  h.send(first, second)
  assert.equal(h.state().status, 'both')
  assert.deepEqual(h.state().activeHands, ['alpha', 'beta'])
  assert.deepEqual(h.pose(), h.initial)
})

test('closed pinches on different parts remain ambiguous unless an established candidate stays aimed', t => {
  let now = 1000
  t.mock.method(performance, 'now', () => now)
  const h = harness(t, { liveTracking: true, pick: x => x < 0.45 ? 'wheel' : 'axle' })
  const first = hand('alpha', 0.3, 0.4, true)
  const second = hand('beta', 0.6, 0.4, true)
  h.send(first, second)
  now += 120
  h.send(second, first)
  assert.equal(h.state().held, null)
  h.send(first)
  now += 50
  h.send(first, second)
  assert.equal(h.state().held, null)
  now += 50
  h.send(second, first)
  assert.equal(h.state().held, 'wheel')
  assert.deepEqual(h.state().activeHands, ['alpha'])
  assert.deepEqual(h.pose(), h.initial)
})

test('a live pinch that initially missed can retry with the same hand', t => {
  let now = 1000
  t.mock.method(performance, 'now', () => now)
  const h = harness(t, { liveTracking: true })
  const open = hand('first', 0.9, 0.4)
  const closed = { ...open, pinched: true }
  const aimed = { ...closed, x: 0.6 }
  h.send(open)
  h.send(closed)
  h.send({ ...aimed, id: 'stranger' })
  now += 100
  h.send({ ...aimed, id: 'stranger' })
  assert.equal(h.state().held, null)

  h.send(open)
  h.send(closed)
  h.send(aimed)
  assert.equal(h.state().held, null)
  now += 100
  h.send(aimed)
  assert.equal(h.state().held, 'wheel')
  assert.deepEqual(h.state().activeHands, ['first'])
  assert.deepEqual(h.pose(), h.initial)
})

test('live secondary pinches can join after separating and recover without restarting the primary', t => {
  let now = 1000
  t.mock.method(performance, 'now', () => now)
  const h = harness(t, { liveTracking: true, joinAnywhere: true })
  const open = hand('first', 0.3, 0.4)
  const first = { ...open, pinched: true }
  const closeSecond = hand('second', 0.32, 0.4, true)
  const second = { ...closeSecond, x: 0.65 }
  h.send(open)
  h.send(first, closeSecond)
  h.send(first, closeSecond)
  now += 100
  h.send(first, closeSecond)
  assert.equal(h.state().status, 'holding')
  h.send(first, second)
  assert.equal(h.state().status, 'both')
  assert.deepEqual(h.state().activeHands, ['first', 'second'])
  assert.deepEqual(h.pose(), h.initial)

  const moved = { ...first, x: 0.4 }
  h.send(moved)
  assert.deepEqual(h.state().activeHands, ['first'])
  assert.deepEqual(h.pose(), h.initial)
  now += 25
  const stillMoving = { ...moved, x: 0.45 }
  h.send(stillMoving)
  const translated = h.pose()
  assertVector(translated.position, h.initial.position.clone().add(new Vector3(0.05, 0, 0)))
  now += 50
  h.send(stillMoving, second)
  assert.equal(h.state().status, 'both')
  assert.deepEqual(h.pose(), translated)

  h.send(stillMoving)
  now += 100
  h.send(stillMoving)
  assert.equal(h.state().status, 'holding')
  assert.deepEqual(h.state().activeHands, ['first'])
  assert.deepEqual(h.pose(), translated)
  h.send({ ...stillMoving, x: 0.5 })
  assertVector(h.pose().position, h.initial.position.clone().add(new Vector3(0.1, 0, 0)))
  assert.equal(h.edits.length, 0)
})

test('a live second pinch retains confirmation across a missing frame without counting that gap', t => {
  let now = 1000
  t.mock.method(performance, 'now', () => now)
  const h = harness(t, { liveTracking: true, joinAnywhere: true })
  const open = hand('first', 0.3, 0.4)
  const first = { ...open, pinched: true }
  const second = { ...hand('second', 0.65, 0.4, true), depthValid: false }
  h.send(open)
  h.send(first, second)
  now += 60
  h.send(first, second)
  now += 40
  h.send(first)
  now += 100
  h.send(first, second)
  assert.equal(h.state().status, 'holding')
  now += 39
  h.send(first, second)
  assert.equal(h.state().status, 'holding')
  now += 1
  h.send(first, second)
  assert.equal(h.state().status, 'both')
  assert.deepEqual(h.pose(), h.initial)

  h.send(first)
  h.send({ ...second, pinched: false })
  h.send(first, second)
  assert.deepEqual(h.state().activeHands, ['first'])
  assert.equal(h.edits.length, 0)
})

test('two fresh live pinches join on the same frame', t => {
  const h = harness(t, { liveTracking: true, joinAnywhere: true })
  const first = hand('first', 0.3, 0.4)
  const second = hand('second', 0.65, 0.4)
  h.send(first, second)
  h.send({ ...first, pinched: true }, { ...second, pinched: true })
  assert.deepEqual(h.state().activeHands, ['first', 'second'])
  assert.equal(h.state().status, 'both')
  assert.deepEqual(h.pose(), h.initial)
})

test('reports arriving close together cannot extend a long interruption in camera time', t => {
  let now = 1000
  t.mock.method(performance, 'now', () => now)
  const h = harness(t)
  const open = hand('first', 0.3, 0.4)
  const pinch = { ...open, pinched: true }
  h.sendAged(170, open)
  h.sendAged(170, pinch)
  h.sendAged(170, { ...pinch, x: 0.4 })
  const frozen = h.pose()
  h.sendLost(170)
  assert.equal(h.state().held, 'wheel')

  now += 20
  h.send({ ...pinch, x: 0.7 })
  assert.equal(h.state().status, 'lost')
  assert.equal(h.edits.length, 1)
  assert.deepEqual(h.pose(), frozen)
})

test('the interruption deadline releases a grip even if the camera stops sending frames', t => {
  let now = 1000
  t.mock.method(performance, 'now', () => now)
  t.mock.timers.enable({ apis: ['setTimeout'] })
  const h = harness(t)
  const open = hand('first', 0.3, 0.4)
  h.send(open)
  const pinch = { ...open, pinched: true }
  h.send(pinch)
  h.send({ ...pinch, x: 0.4 })
  h.sendLost()

  now += 149
  t.mock.timers.tick(149)
  assert.equal(h.state().status, 'holding')
  now += 2
  t.mock.timers.tick(2)
  assert.equal(h.state().status, 'lost')
  assert.equal(h.edits.length, 1)
  h.send(pinch)
  assert.equal(h.state().held, null)
})

test('steady camera latency leaves the stream watchdog time for another frame', t => {
  let now = 1000
  t.mock.method(performance, 'now', () => now)
  t.mock.timers.enable({ apis: ['setTimeout'] })
  const h = harness(t)
  const open = hand('first', 0.3, 0.4)
  h.sendAged(215, open)
  now += 35
  t.mock.timers.tick(35)
  h.sendAged(215, { ...open, pinched: true })
  assert.equal(h.state().held, 'wheel')

  now += FRAME_MAX_AGE_MS - 1
  t.mock.timers.tick(FRAME_MAX_AGE_MS - 1)
  assert.equal(h.state().status, 'holding')
  now += 2
  t.mock.timers.tick(2)
  assert.equal(h.state().status, 'lost')
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
  assert.equal(h.state().status, 'holding')
  h.controller.resetStream()
  h.send({ ...fist, x: 0.6 })
  assert.equal(h.state().target, null)
  assertVector(h.camera.orbit, new Vector3(0.195, 0, 0))

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

test('the join ring stays continuous at the edge and uses a circular pixel hit area', t => {
  const h = harness(t)
  const width = 1600
  const height = 900
  h.controller.setViewport(width, height)
  const open = hand('first', 0.6, 0.4)
  const held = { ...open, pinched: true }
  h.send(open)
  h.send(held)
  const radius = h.state().join!.radiusPx
  const edge = 1 - CALIBRATION_INSET_X - radius / width - 0.19
  h.send({ ...held, x: edge - 0.001 })
  const before = h.state().join!
  h.send({ ...held, x: edge + 0.001 })
  const after = h.state().join!
  assert.ok(Math.abs((after.x - before.x) * width) < 3.3)
  assert.ok(Math.abs((after.y - before.y) * height) < 3.3)
  assert.equal(after.radiusX * width, radius)
  assert.equal(after.radiusY * height, radius)
  assert.ok(isInsideJoin(after, after.x, after.y + (radius - 1) / height))
  assert.equal(isInsideJoin(after, after.x, after.y + (radius + 1) / height), false)
  assert.equal(isInsideJoin(after, after.x - (radius + 1) / width, after.y), false)

  const outsider = hand('second', after.x - (radius + 1) / width, after.y)
  const first = { ...held, x: edge + 0.001 }
  h.send(first, outsider)
  h.send(first, { ...outsider, pinched: true })
  assert.equal(h.state().status, 'holding')
  const insider = hand('second', after.x - (radius - 1) / width, after.y)
  h.send(first, insider)
  h.send(first, { ...insider, pinched: true })
  assert.equal(h.state().status, 'both')
})

test('the join ring fits and can be held at each calibrated corner', t => {
  const h = harness(t)
  for (const [width, height] of [[1600, 900], [40, 30]]) {
    h.controller.setViewport(width, height)
    for (const x of [CALIBRATION_INSET_X, 1 - CALIBRATION_INSET_X]) {
      for (const y of [CALIBRATION_INSET_Y, 1 - CALIBRATION_INSET_Y]) {
        h.controller.cancel()
        const start = hand('first', x < 0.5 ? 0.4 : 0.6, y < 0.5 ? 0.6 : 0.4)
        h.send(start)
        h.send({ ...start, pinched: true })
        const atCorner = { ...start, x, y, pinched: true }
        h.send(atCorner)
        const join = h.state().join!
        const tolerance = 1e-12
        assert.ok(join.x - join.radiusX >= CALIBRATION_INSET_X - tolerance)
        assert.ok(join.x + join.radiusX <= 1 - CALIBRATION_INSET_X + tolerance)
        assert.ok(join.y - join.radiusY >= CALIBRATION_INSET_Y - tolerance)
        assert.ok(join.y + join.radiusY <= 1 - CALIBRATION_INSET_Y + tolerance)
        const second = hand('second', join.x, join.y)
        h.send(atCorner, second)
        h.send(atCorner, { ...second, pinched: true })
        assert.equal(h.state().status, 'both')
      }
    }
  }
})

test('a two-hand depth rotation makes a full turn through the opposite direction without snapping', t => {
  const h = harness(t, { liveTracking: true, joinAnywhere: true })
  const turnAxis = new Vector3(0, 1, 1).normalize()
  const first = hand('first', 0.32, 0.5)
  const second = hand('second', 0.68, 0.5)
  h.send(first, second)
  h.send({ ...first, pinched: true }, { ...second, pinched: true })
  assert.equal(h.state().status, 'both')

  const step = Math.PI / 12
  for (let index = 1; index <= 24; index++) {
    const before = h.pose().rotation
    const expected = new Quaternion().setFromAxisAngle(turnAxis, index * step)
    const span = new Vector3(0.18, 0, 0).applyQuaternion(expected)
    h.send(
      hand('first', 0.5 - span.x, 0.5 + span.y, true, -span.z),
      hand('second', 0.5 + span.x, 0.5 - span.y, true, span.z),
    )
    const pose = h.pose()
    assert.ok(pose.rotation.angleTo(expected) < 1e-6, `incorrect orientation at step ${index}`)
    assert.ok(Math.abs(pose.rotation.angleTo(before) - step) < 1e-6, `rotation jumped at step ${index}`)
    assertVector(pose.position, h.initial.position)
    assert.ok(Math.abs(pose.scale - 1) < 1e-8)
  }
})

test('two-hand rotation rebases after close or crossed positions without snapping', async t => {
  for (const passNearCenter of [true, false]) {
    await t.test(passNearCenter ? 'sampled inside the dead zone' : 'crossed between frames', subtest => {
      const h = harness(subtest)
      const first = hand('first', 0.4, 0.5)
      const held = { ...first, pinched: true }
      h.send(first)
      h.send(held)
      const join = h.state().join!
      const right = hand('second', join.x, join.y)
      h.send(held, right)
      h.send(held, { ...right, pinched: true })

      if (passNearCenter) {
        h.send({ ...held, x: 0.49 }, { ...right, x: 0.51, pinched: true })
      }

      h.send({ ...held, x: 0.535 }, { ...right, x: 0.465, pinched: true })
      assert.equal(h.pose().scale, 1)
      assert.ok(h.pose().rotation.angleTo(h.initial.rotation) < 1e-8)

      h.send({ ...held, x: 0.57 }, { ...right, x: 0.43, pinched: true })
      assert.equal(h.pose().scale, 1)
      assert.ok(h.pose().rotation.angleTo(h.initial.rotation) < 1e-8)
      h.send({ ...held, x: 0.64 }, { ...right, x: 0.36, pinched: true })
      assert.ok(Math.abs(h.pose().scale - 2) < 1e-8)
      assert.ok(h.pose().rotation.angleTo(h.initial.rotation) < 1e-8)
    })
  }
})

test('a viewport resize ends a grip once and ignores an unchanged or invalid viewport', t => {
  const h = harness(t)
  const open = hand('first', 0.3, 0.4)
  const held = { ...open, pinched: true }
  h.send(open)
  h.send(held)
  h.send({ ...held, x: 0.4 })
  const position = h.pose().position.clone()
  h.controller.setViewport(1000, 1000)
  h.controller.setViewport(0, 900)
  assert.equal(h.state().target, 'part')
  h.controller.setViewport(1600, 900)
  assert.equal(h.state().target, null)
  assert.equal(h.edits.length, 1)
  h.send({ ...held, x: 0.6 })
  assertVector(h.pose().position, position)
  h.send(open)
  h.send(held)
  assert.equal(h.state().target, 'part')
})

test('reconnecting accepts a restarted sequence after a neutral sample', t => {
  const h = harness(t)
  const open = hand('first', 0.3, 0.4)
  const held = { ...open, pinched: true }
  h.send(open)
  h.send(held)
  h.send({ ...held, x: 0.4 })
  h.controller.resetStream()
  h.resetSequence()
  assert.equal(h.state().status, 'lost')
  assert.equal(h.edits.length, 1)
  h.send(held)
  assert.equal(h.state().target, null)
  h.send(open)
  h.send(held)
  assert.equal(h.state().target, 'part')
})

test('a 1.5-second thumbs-up freezes a grip and resets once until its owner releases', t => {
  let now = 1000
  let resets = 0
  t.mock.method(performance, 'now', () => now)
  const h = harness(t, { onReset: () => { resets++; h.controller.cancel() } })
  const first = hand('first', 0.3, 0.4)
  const pinch = { ...first, pinched: true }
  let thumb = { ...hand('second', 0.7, 0.4), thumbsUp: true }
  h.send(first)
  h.send(pinch)
  h.send({ ...pinch, x: 0.4 })
  const frozen = h.pose()
  h.send({ ...pinch, x: 0.8 }, thumb)
  assert.deepEqual(h.pose(), frozen)
  assert.deepEqual(h.state().activeHands, [])
  assert.equal(h.state().target, null)
  assert.equal(h.edits.length, 1)
  for (let index = 0; index < 14; index++) {
    now += 100
    h.send(pinch, thumb)
  }
  now += 99
  h.send(pinch, thumb)
  assert.equal(resets, 0)
  assert.equal(h.state().resetProgress, 1499 / 1500)
  now += 1
  h.send(pinch, thumb)
  assert.equal(resets, 1)
  assert.equal(h.state().resetProgress, 0)
  for (let index = 0; index < 32; index++) {
    now += 100
    h.send({ ...first, thumbsUp: true }, { ...thumb, thumbsUp: false })
  }
  assert.equal(resets, 1)
  h.controller.resetStream()
  h.resetSequence()
  thumb = { ...thumb, id: 'renamed' }
  h.send(first, thumb)
  h.send(first, { ...thumb, thumbsUp: false, depthValid: false })
  for (let index = 0; index < 32; index++) {
    now += 100
    h.send(first, thumb)
  }
  assert.equal(resets, 1)
  h.send(first, { ...thumb, thumbsUp: false })
  h.send(first, thumb)
  for (let index = 0; index < RESET_HOLD_MS / 100; index++) {
    now += 100
    h.send(first, thumb)
  }
  assert.equal(resets, 2)
  assert.deepEqual(h.pose(), frozen)
})

test('a thumbs-up without measured depth cannot reset or start orbiting', t => {
  let now = 1000
  let resets = 0
  t.mock.method(performance, 'now', () => now)
  const h = harness(t, { liveTracking: true, onReset: () => { resets++ } })
  const open = { ...hand('first', 0.3, 0.4), depthValid: false }
  const thumb = { ...open, thumbsUp: true }
  h.send(open)
  h.send(thumb)
  assert.equal(h.state().target, null)
  for (let index = 0; index < RESET_HOLD_MS / 100; index++) {
    now += 100
    h.send(thumb)
  }
  assert.equal(h.state().resetProgress, 0)
  assert.equal(resets, 0)
})

test('two thumbs-up choose a stable deterministic owner and changing owner restarts the countdown', t => {
  let now = 1000
  t.mock.method(performance, 'now', () => now)
  const h = harness(t)
  const alpha = { ...hand('alpha', 0.3, 0.4), thumbsUp: true }
  const beta = { ...hand('beta', 0.7, 0.4), thumbsUp: true }
  h.send(beta, alpha)
  now += 100
  h.send(alpha, beta)
  assert.equal(h.state().resetProgress, 100 / RESET_HOLD_MS)
  now += 100
  h.send(beta)
  assert.equal(h.state().resetProgress, 0)
  now += 100
  h.send(alpha, beta)
  assert.equal(h.state().resetProgress, 100 / RESET_HOLD_MS)
  now += 100
  h.send({ ...beta, id: 'new-person' })
  assert.equal(h.state().resetProgress, 0)
})

test('thumbs-up countdown uses fresh captures and cancels on loss, stale frames and setup cancellation', t => {
  let now = 1000
  let resets = 0
  t.mock.method(performance, 'now', () => now)
  const h = harness(t, { onReset: () => { resets++ } })
  const thumb = { ...hand('first', 0.3, 0.4), thumbsUp: true }
  h.send(thumb)
  now += 100
  h.replay([thumb], 0, 0)
  h.sendAged(100, thumb)
  assert.equal(h.state().resetProgress, 0)
  for (let index = 0; index < 10; index++) {
    now += 100
    h.send(thumb)
  }
  assert.ok(h.state().resetProgress > 0)
  h.sendLost()
  assert.equal(h.state().resetProgress, 0)
  h.send(thumb)
  now += 100
  h.send(thumb)
  h.controller.cancel()
  assert.equal(h.state().resetProgress, 0)
  h.send(thumb)
  now += 100
  h.send(thumb)
  now += 500
  h.sendAged(FRAME_MAX_AGE_MS, thumb)
  assert.equal(h.state().resetProgress, 0)
  h.send(thumb)
  assert.equal(h.state().resetProgress, 0)
  now += 500
  h.send(thumb)
  assert.equal(h.state().resetProgress, 0)
  assert.equal(resets, 0)
})

test('a stopped thumbs-up stream cancels its progress without firing the reset timer', t => {
  let now = 1000
  let resets = 0
  t.mock.method(performance, 'now', () => now)
  t.mock.timers.enable({ apis: ['setTimeout'] })
  const h = harness(t, { onReset: () => { resets++ } })
  const thumb = { ...hand('first', 0.3, 0.4), thumbsUp: true }
  h.send(thumb)
  now += 100
  h.send(thumb)
  assert.ok(h.state().resetProgress > 0)
  now += RESET_HOLD_MS
  t.mock.timers.tick(RESET_HOLD_MS)
  assert.equal(h.state().resetProgress, 0)
  assert.equal(resets, 0)
})
