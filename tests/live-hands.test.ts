import assert from 'node:assert/strict'
import { test, type TestContext } from 'node:test'
import { applyHandCameraSettings, fetchHandCameraSettings } from '../src/live-hands.ts'

const settings = {
  fps: 60, applying: false,
  frameRates: [30, 60], diagnostics: {
    poseFps: 28.5, processingMs: 22.4, frameAgeMs: 8,
    estimatedHands: 2, depthHands: 1, trackedHands: 1,
  },
}

function withWindow(t: TestContext) {
  const previous = Object.getOwnPropertyDescriptor(globalThis, 'window')
  Object.defineProperty(globalThis, 'window', { configurable: true, value: { location: { href: 'http://localhost:4180/?operator=1' } } })
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
