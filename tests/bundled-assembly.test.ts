import assert from "node:assert/strict";
import { test } from "node:test";
import { decodeAssembly, encodeAssembly, sha256 } from "../src/assembly-binary.ts";
import type { CadAssembly, CadOccurrence } from "../src/cad.ts";

const placement = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

function fixture(): CadAssembly {
  const part: CadOccurrence = {
    id: "root/a",
    name: "Left bracket",
    matrix: placement,
    mesh: 0,
    color: [0, 0, 1],
    colors: [[1, 0, 0], undefined],
    children: [],
  };

  return {
    roots: [{
      id: "root", name: "Brackets", matrix: placement,
      children: [part, { ...part, id: "root/b", name: "Right bracket", colors: [[0, 1, 0], undefined] }],
    }],
    meshes: [{
      positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
      normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
      indices: new Uint32Array([0, 1, 2, 0, 2, 1]),
      groups: [{ start: 0, count: 3 }, { start: 3, count: 3 }],
    }],
  };
}

test("bundled assembly preserves shared geometry, face groups and occurrence colors", async () => {
  const input = fixture();
  const sourceHash = await sha256(new TextEncoder().encode("source STEP"));
  const bytes = await encodeAssembly(input, sourceHash);
  assert.deepEqual(bytes, await encodeAssembly(input, sourceHash));

  const { assembly, sourceSha256 } = await decodeAssembly(bytes);
  assert.deepEqual(sourceSha256, sourceHash);
  const [left, right] = assembly.roots[0].children;
  assert.equal(left.mesh, right.mesh);
  assert.deepEqual(left.colors, input.roots[0].children[0].colors);
  assert.deepEqual(right.colors, input.roots[0].children[1].colors);
  const [mesh] = assembly.meshes;
  assert.deepEqual(mesh.groups, input.meshes[0].groups);
  assert.deepEqual(mesh.positions, input.meshes[0].positions);
  assert.deepEqual(mesh.indices, input.meshes[0].indices);
  assert.equal(mesh.positions.buffer, bytes);
});

test("bundled assembly detects corrupted payloads and unsupported versions", async () => {
  const bytes = await encodeAssembly(fixture(), new Uint8Array(32));
  const corrupted = bytes.slice(0);
  const payload = new Uint8Array(corrupted);
  payload[payload.length - 1] ^= 1;
  await assert.rejects(decodeAssembly(corrupted), /checksum/);

  const unsupported = bytes.slice(0);
  new DataView(unsupported).setUint32(8, 99, true);
  await assert.rejects(decodeAssembly(unsupported), /format version/);
});
