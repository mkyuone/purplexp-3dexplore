import type { PerspectiveCamera } from 'three';

export interface ViewerOffset { x: number; y: number }

export function applyHeadProjection(
  camera: PerspectiveCamera,
  neutral: PerspectiveCamera,
  screenDistance: number,
  offset: ViewerOffset,
) {
  const { near, far, projectionMatrix } = neutral;
  const halfWidth = screenDistance / projectionMatrix.elements[0];
  const halfHeight = screenDistance / projectionMatrix.elements[5];
  const x = offset.x * 2 * halfWidth;
  const y = offset.y * 2 * halfHeight;
  camera.position.set(x, y, 0).applyQuaternion(neutral.quaternion).add(neutral.position);
  camera.quaternion.copy(neutral.quaternion);
  camera.updateMatrixWorld();

  if (x === 0 && y === 0) {
    camera.projectionMatrix.copy(projectionMatrix);
    camera.projectionMatrixInverse.copy(neutral.projectionMatrixInverse);
    return;
  }

  // Keep the screen fixed in the neutral camera's frame. Only the eye
  // translates, so the same screen corners define a different near frustum.
  const scale = near / screenDistance;
  camera.projectionMatrix.makePerspective(
    (-halfWidth - x) * scale, (halfWidth - x) * scale,
    (halfHeight - y) * scale, (-halfHeight - y) * scale,
    near, far, neutral.coordinateSystem, neutral.reversedDepth,
  );
  camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert();
}
