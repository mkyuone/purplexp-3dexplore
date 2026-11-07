import assert from 'node:assert/strict';
import { test } from 'node:test';
import { HeadPoseTracker, type HeadFace } from '../src/head-pose.ts';

const face = (x = 0.5, y = 0.5, width = 0.23, height = 0.29): HeadFace => ({ x, y, width, height });

function center(tracker: HeadPoseTracker, at = 0, viewer = face()) {
  for (let offset = 0; offset <= 250; offset += 25) tracker.receive([viewer], at + offset);
}

function near(actual: number, expected: number) {
  assert.ok(Math.abs(actual - expected) < 1e-8, `${actual} differs from ${expected}`);
}

test('centers after a continuous quarter-second with one steady viewer', () => {
  const tracker = new HeadPoseTracker();
  for (let at = 0; at < 250; at += 25) tracker.receive([face()], at);
  assert.equal(tracker.sample(249).phase, 'centering');
  tracker.receive([face()], 250);
  assert.deepEqual(tracker.sample(250), { x: 0, y: 0, phase: 'tracking', animating: false, nextSampleAt: 400 });

  tracker.reset();
  tracker.receive([face()], 0);
  tracker.receive([face()], 250);
  assert.equal(tracker.sample(250).phase, 'centering');
  for (let at = 275; at <= 700; at += 25) tracker.receive([face(at % 50 === 0 ? 0.45 : 0.55)], at);
  assert.equal(tracker.sample(700).phase, 'centering');
  for (let at = 725; at <= 1025; at += 25) tracker.receive([face(0.32), face(0.7)], at);
  assert.equal(tracker.sample(1025).phase, 'centering');
  center(tracker, 1050, face(0.7));
  assert.equal(tracker.sample(1300).phase, 'tracking');
});

test('mirrors horizontal movement, maps up and smooths and clamps both axes', () => {
  const tracker = new HeadPoseTracker();
  center(tracker);
  tracker.receive([face(0.45, 0.47)], 283);
  const moving = tracker.sample(283);
  assert.ok(moving.x > 0.035 && moving.x < 0.047);
  assert.ok(moving.y > 0.02 && moving.y < 0.028);

  for (let i = 1; i <= 8; i += 1) tracker.receive([face(0.45 - i * 0.022, 0.47 - i * 0.023)], 283 + i * 33);
  for (let at = 580; at <= 910; at += 33) tracker.receive([face(0.274, 0.286)], at);
  near(tracker.sample(910).x, 0.18);
  near(tracker.sample(910).y, 0.12);

  tracker.recenter();
  near(tracker.sample(910).x, 0);
  near(tracker.sample(910).y, 0);
  tracker.receive([face(0.2748, 0.2869)], 943);
  near(tracker.sample(943).x, 0);
  near(tracker.sample(943).y, 0);
  tracker.receive([face(0.31, 0.32)], 976);
  assert.ok(tracker.sample(976).x < 0 && tracker.sample(976).y < 0);
});

test('holds the last pose for 150 ms and eases to center for the next 250 ms even if frames stall', () => {
  const tracker = new HeadPoseTracker();
  center(tracker);
  tracker.receive([face(0.46)], 283);
  const before = tracker.sample(283).x;
  assert.ok(before > 0);
  assert.equal(tracker.sample(432).nextSampleAt, 433);
  near(tracker.sample(432).x, before);
  assert.equal(tracker.sample(432).phase, 'tracking');

  const midpoint = tracker.sample(558);
  assert.equal(midpoint.phase, 'lost');
  assert.equal(midpoint.animating, true);
  assert.equal(midpoint.nextSampleAt, null);
  near(midpoint.x, before / 2);
  assert.deepEqual(tracker.sample(683), { x: 0, y: 0, phase: 'lost', animating: false, nextSampleAt: null });
});

test('recovers from a brief occlusion without changing the neutral point or jumping on reacquisition', () => {
  const tracker = new HeadPoseTracker();
  center(tracker);
  tracker.receive([face(0.46)], 283);
  const held = tracker.sample(283).x;
  tracker.receive([], 316);
  assert.equal(tracker.sample(316).phase, 'lost');
  near(tracker.sample(316).x, held);
  tracker.receive([face(0.46)], 382);
  assert.equal(tracker.sample(382).phase, 'tracking');
  assert.ok(tracker.sample(382).x >= held);

  const fading = tracker.sample(600).x;
  tracker.receive([face(0.46)], 600);
  assert.equal(tracker.sample(600).phase, 'tracking');
  near(tracker.sample(600).x, fading);
  tracker.receive([face(0.46)], 633);
  assert.ok(tracker.sample(633).x > fading);
  tracker.receive([face(0.49)], 666);
  assert.ok(tracker.sample(666).x > 0);
});

test('keeps its viewer when a bystander appears and waits for a stable new viewer after loss', () => {
  const tracker = new HeadPoseTracker();
  center(tracker);
  const bystander = face(0.8, 0.54, 0.35, 0.37);
  tracker.receive([bystander, face(0.48)], 283);
  assert.equal(tracker.sample(283).phase, 'tracking');
  const held = tracker.sample(283).x;
  assert.ok(held > 0);
  tracker.receive([face(0.47), face(0.49)], 316);
  assert.equal(tracker.sample(316).phase, 'lost');
  near(tracker.sample(316).x, held);
  for (let at = 349; at < 800; at += 33) tracker.receive([bystander], at);
  tracker.receive([bystander], 800);
  near(tracker.sample(800).x, 0);
  assert.equal(tracker.sample(800).phase, 'centering');

  tracker.receive([bystander, face(0.5)], 830);
  center(tracker, 860, bystander);
  assert.equal(tracker.sample(1110).phase, 'tracking');
  near(tracker.sample(1110).x, 0);
  tracker.receive([face(0.77, 0.54, 0.35, 0.37)], 1143);
  assert.ok(tracker.sample(1143).x > 0);
});

test('ignores malformed and out-of-order samples and starts fresh after a capture reset', () => {
  const tracker = new HeadPoseTracker();
  center(tracker);
  tracker.receive([face(NaN)], 283);
  assert.equal(tracker.sample(283).phase, 'lost');
  tracker.receive([face(0.45)], 270);
  assert.equal(tracker.sample(283).phase, 'lost');
  tracker.recenter();
  assert.equal(tracker.sample(283).phase, 'centering');
  tracker.reset();
  center(tracker, 0, face(0.3, 0.3));
  assert.equal(tracker.sample(250).phase, 'tracking');
  near(tracker.sample(250).x, 0);
});

test('recentering while the camera has stalled waits for a new stable face', () => {
  const tracker = new HeadPoseTracker();
  center(tracker);
  tracker.recenter(450);
  assert.equal(tracker.sample(450).phase, 'centering');
  center(tracker, 500, face(0.63));
  near(tracker.sample(750).x, 0);
  assert.equal(tracker.sample(750).phase, 'tracking');
});
