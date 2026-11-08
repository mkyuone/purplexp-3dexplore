import assert from 'node:assert/strict'
import { test, type TestContext } from 'node:test'
import { applyHandCameraSettings, connectLiveHands, fetchHandCameraSettings } from '../src/live-hands.ts'
import type { PoseFrame } from '../src/gestures.ts'

const settings = {
  fps: 60, applying: false,
  frameRates: [30, 60], diagnostics: {
    poseFps: 28.5, processingMs: 22.4, frameAgeMs: 8,
    estimatedHands: 2, depthHands: 1, trackedHands: 1,
    queueMs: 2, inferenceMs: 19, alignmentMs: 2, fusionMs: 1,
    processingP95Ms: 24, captureDroppedFrames: 2, depthFrameAvailable: true, frameAgeSource: 'hostArrival',
  },
}

function withWindow(t: TestContext) {
  const previous = Object.getOwnPropertyDescriptor(globalThis, 'window')
  Object.defineProperty(globalThis, 'window', { configurable: true, value: {
    location: { href: 'http://localhost:4180/?operator=1' }, innerWidth: 1280, innerHeight: 720,
    screen: { width: 1920, height: 1080 }, devicePixelRatio: 1, screenX: 0, screenY: 0,
  } })
  t.after(() => {
    if (previous) Object.defineProperty(globalThis, 'window', previous)
    else Reflect.deleteProperty(globalThis, 'window')
  })
}

test('camera settings retain capture rates and diagnostics', async t => {
  withWindow(t)
  let requestUrl: string | undefined
  const fetch = t.mock.method(globalThis, 'fetch', async (url: URL, init: RequestInit) => {
    requestUrl = url.href
    assert.equal(init.method, 'GET')
    assert.equal(init.cache, 'no-store')
    return Response.json(settings)
  })
  const result = await fetchHandCameraSettings(new AbortController().signal)
  assert.equal(requestUrl, 'http://127.0.0.1:4182/settings')
  assert.deepEqual(result.frameRates, [30, 60])
  assert.deepEqual(result.diagnostics, settings.diagnostics)
  assert.equal(result.fps, 60)
  assert.equal(fetch.mock.callCount(), 1)
})

test('applying a capture rate uses RGB MediaPipe and shares cancellation', async t => {
  withWindow(t)
  const controller = new AbortController()
  let requestSignal: AbortSignal | undefined
  t.mock.method(globalThis, 'fetch', async (_url: URL, init: RequestInit) => {
    assert.equal(init.method, 'POST')
    assert.deepEqual(JSON.parse(String(init.body)), { source: 'rgb', tracker: 'mediapipe', fps: 60 })
    requestSignal = init.signal as AbortSignal
    return Response.json({ ...settings, applying: true })
  })
  const result = await applyHandCameraSettings(60, controller.signal)
  assert.equal(result.applying, true)
  controller.abort()
  assert.equal(requestSignal?.aborted, true)
})

test('camera settings reject server failures and unsupported capture rates', async t => {
  withWindow(t)
  const fetch = t.mock.method(globalThis, 'fetch', async () => Response.json({ error: 'Capture rate unavailable' }, { status: 400 }))
  await assert.rejects(applyHandCameraSettings(60, new AbortController().signal), /Capture rate unavailable/)
  fetch.mock.mockImplementation(async () => Response.json({ ...settings, fps: 120, frameRates: [30, 60, 120] }))
  await assert.rejects(fetchHandCameraSettings(new AbortController().signal), /invalid settings/)
  fetch.mock.mockImplementation(async () => Response.json({ ...settings, frameRates: [30] }))
  await assert.rejects(fetchHandCameraSettings(new AbortController().signal), /invalid settings/)
})

test('camera hand counts accept zero and omit invalid or missing counts', async t => {
  withWindow(t)
  t.mock.method(globalThis, 'fetch', async () => Response.json({
    ...settings, diagnostics: { estimatedHands: 1, depthHands: 0, trackedHands: -1 },
  }))
  const result = await fetchHandCameraSettings(new AbortController().signal)
  assert.equal(result.diagnostics?.estimatedHands, 1)
  assert.equal(result.diagnostics?.depthHands, 0)
  assert.equal(result.diagnostics?.trackedHands, undefined)
})

test('a missed pose request and timestamp jitter preserve the active camera stream', { timeout: 1500 }, async t => {
  withWindow(t)
  let now = 100
  let requests = 0
  let resets = 0
  let initialResets = 0
  const frames: PoseFrame[] = []
  const states: string[] = []
  const received = Promise.withResolvers<void>()
  t.mock.method(performance, 'now', () => now)
  t.mock.method(globalThis, 'fetch', async () => {
    requests++
    now += 15
    if (requests === 2) return new Response(null, { status: 503 })
    return Response.json({
      status: 'ready', session: 'same-camera', sequence: requests, ageMs: requests === 1 ? 5 : 60,
      tracking: 'tracking', hands: [{ id: 'pinching', x: .5, y: .5, depth: 0, depthValid: false, pinched: true }],
    })
  })
  t.after(connectLiveHands({
    navigation: 'fuzzy', reset: () => { resets++ }, status: value => states.push(value.phase),
    receive: frame => {
      frames.push(frame)
      if (frames.length === 1) initialResets = resets
      else received.resolve()
    },
  }))
  await received.promise
  assert.deepEqual(frames.map(frame => frame.sequence), [1, 3])
  assert.equal(frames[1].capturedAt, frames[0].capturedAt)
  assert.equal(resets, initialResets)
  assert.deepEqual(states, ['ready'])
})

test('a sustained request failure marks a connected camera offline and resets its stream', { timeout: 1500 }, async t => {
  withWindow(t)
  let now = 100
  let requests = 0
  let resets = 0
  let initialResets = 0
  const offline = Promise.withResolvers<void>()
  t.mock.method(performance, 'now', () => now)
  t.mock.method(globalThis, 'fetch', async () => {
    if (requests++ === 0) return Response.json({
      status: 'ready', session: 'same-camera', sequence: 1, ageMs: 0, tracking: 'lost', hands: [],
    })
    now += 600
    throw new Error('Camera unavailable')
  })
  t.after(connectLiveHands({
    navigation: 'fuzzy', reset: () => { resets++ }, receive: () => { initialResets = resets },
    status: value => { if (value.phase === 'offline') offline.resolve() },
  }))
  await offline.promise
  assert.equal(resets, initialResets + 1)
})

test('long polling requests the next sequence and excludes the bridge wait from pose age', { timeout: 1000 }, async t => {
  withWindow(t)
  const received = Promise.withResolvers<void>()
  const requests: URL[] = []
  const frames: PoseFrame[] = []
  const session = '0123456789abcdef0123456789abcdef:fuzzy:1280x720@1:1920x1080:0,0'
  let now = 100
  t.mock.method(performance, 'now', () => now)
  t.mock.method(globalThis, 'fetch', async (url: URL) => {
    requests.push(url)
    const bridgeMs = requests.length === 1 ? 0 : 32
    now += bridgeMs + 2
    return Response.json({ status: 'ready', waitForPose: true, bridgeMs, session,
      sequence: requests.length, ageMs: 8, tracking: 'lost', hands: [] })
  })
  const disconnect = connectLiveHands({
    navigation: 'fuzzy', reset: () => {}, status: () => {},
    receive: frame => {
      frames.push(frame)
      if (frames.length === 2) received.resolve()
    },
  })
  t.after(disconnect)
  await received.promise
  assert.equal(requests[0].searchParams.has('after'), false)
  assert.equal(requests[1].searchParams.get('after'), '1')
  assert.equal(requests[1].searchParams.get('session'), session)
  assert.deepEqual(frames.map(frame => frame.capturedAt), [93, 127])
})

test('a new depth range drops the old grip before delivering depth in the new coordinates', { timeout: 1000 }, async t => {
  withWindow(t)
  const received = Promise.withResolvers<void>()
  const events: string[] = []
  const frames = [
    { sequence: 1, depthRangeRevision: 0 },
    { sequence: 1, depthRangeRevision: 1 },
    { sequence: 2, depthRangeRevision: 1 },
  ]
  t.mock.method(globalThis, 'fetch', async () => Response.json({
    status: 'ready', session: 'same-camera', ageMs: 0,
    tracking: 'tracking', hands: [{ id: 'held', x: .5, y: .5, depth: .2, depthValid: true, pinched: true }],
    ...frames.shift(),
  }))
  const disconnect = connectLiveHands({
    navigation: 'fuzzy', status: () => {},
    reset: reason => { if (reason) events.push(reason) },
    receive: frame => {
      events.push(`pose ${frame.sequence}`)
      if (frame.sequence === 2) received.resolve()
    },
  })
  t.after(disconnect)
  await received.promise
  assert.deepEqual(events, ['pose 1', 'depth-range', 'pose 2'])
})
