import assert from 'node:assert/strict';
import { test } from 'node:test';
import { PerspectiveCamera, Raycaster, Vector2, Vector3 } from 'three';
import { applyHeadProjection } from '../src/head-projection.ts';

const SCREEN_DISTANCE = 6.5;

function neutralCamera(aspect: number) {
  const camera = new PerspectiveCamera(34, aspect, 0.01, 1000);
  camera.position.set(6, 4, 9);
  camera.lookAt(1, -2, 0.5);
  camera.updateMatrixWorld();
  return camera;
}

function near(actual: number, expected: number) {
  assert.ok(Math.abs(actual - expected) < 1e-8, `${actual} differs from ${expected}`);
}

test('a centered eye exactly preserves the neutral view', () => {
  const neutral = neutralCamera(16 / 9);
  const viewer = neutralCamera(1);
  applyHeadProjection(viewer, neutral, SCREEN_DISTANCE, { x: 0.13, y: 0.07 });
  applyHeadProjection(viewer, neutral, SCREEN_DISTANCE, { x: 0, y: 0 });

  assert.deepEqual(viewer.matrixWorld.elements, neutral.matrixWorld.elements);
  assert.deepEqual(viewer.projectionMatrix.elements, neutral.projectionMatrix.elements);
  assert.deepEqual(viewer.projectionMatrixInverse.elements, neutral.projectionMatrixInverse.elements);
});

test('the virtual screen corners stay fixed through head movement and viewport resize', () => {
  const neutral = neutralCamera(16 / 9);
  const viewer = neutralCamera(1);
  for (const aspect of [16 / 9, 4 / 3, 3 / 4]) {
    neutral.aspect = aspect;
    neutral.updateProjectionMatrix();
    const size = neutral.getViewSize(SCREEN_DISTANCE, new Vector2());
    for (const offset of [{ x: 0.18, y: -0.12 }, { x: -0.15, y: 0.10 }]) {
      applyHeadProjection(viewer, neutral, SCREEN_DISTANCE, offset);
      for (const x of [-1, 1]) {
        for (const y of [-1, 1]) {
          const corner = new Vector3(x * size.x / 2, y * size.y / 2, -SCREEN_DISTANCE);
          const projected = neutral.localToWorld(corner).project(viewer);
          near(projected.x, x);
          near(projected.y, y);
        }
      }
    }
  }
});

test('points at distinct depths produce distinct parallax with unchanged view scale', () => {
  const neutral = neutralCamera(16 / 9);
  const viewer = neutralCamera(1);
  applyHeadProjection(viewer, neutral, SCREEN_DISTANCE, { x: 0.12, y: -0.08 });
  const depths = [0.5, 1, 2, 4].map((factor) => factor * SCREEN_DISTANCE);
  const points = depths.map((depth) => neutral.localToWorld(new Vector3(0, 0, -depth)).project(viewer));
  [-0.24, 0, 0.12, 0.18].forEach((x, i) => near(points[i].x, x));
  [0.16, 0, -0.08, -0.12].forEach((y, i) => near(points[i].y, y));

  const neutralSize = neutral.getViewSize(depths[2], new Vector2());
  const viewerSize = viewer.getViewSize(depths[2], new Vector2());
  near(viewerSize.x, neutralSize.x);
  near(viewerSize.y, neutralSize.y);
});

test('picking rays follow the applied projection after the navigation camera moves', () => {
  const neutral = neutralCamera(16 / 9);
  const viewer = neutralCamera(1);
  const raycaster = new Raycaster();
  const point = new Vector3(2, 1.5, -4);
  for (const position of [new Vector3(6, 4, 9), new Vector3(-5, 6, 6), new Vector3(4, 0, 2)]) {
    neutral.position.copy(position);
    neutral.lookAt(1, -2, 0.5);
    neutral.updateMatrixWorld();
    applyHeadProjection(viewer, neutral, SCREEN_DISTANCE, { x: -0.11, y: 0.09 });
    const projected = point.clone().project(viewer);
    raycaster.setFromCamera(new Vector2(projected.x, projected.y), viewer);
    near(raycaster.ray.distanceToPoint(point), 0);
  }
});
