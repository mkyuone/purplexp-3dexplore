import type { CadAssembly, CadColor, CadMesh, CadOccurrence } from "./cad.ts";

const magic = new TextEncoder().encode("SCADMESH");
const formatVersion = 1;
const headerBytes = 80;
const sha256Bytes = 32;
const sourceHashOffset = 16;
const payloadHashOffset = sourceHashOffset + sha256Bytes;
const littleEndian = new Uint8Array(new Uint16Array([1]).buffer)[0] === 1;

type StoredOccurrence = Omit<CadOccurrence, "children" | "colors"> & {
  children: StoredOccurrence[];
  colors?: (CadColor | null)[];
};
type StoredMesh = Pick<CadMesh, "color" | "groups"> & {
  positions: number;
  normals: number;
  indices: number;
};
type Manifest = { roots: StoredOccurrence[]; meshes: StoredMesh[] };

const align4 = (size: number) => Math.ceil(size / 4) * 4;
const equalBytes = (first: Uint8Array, second: Uint8Array) =>
  first.length === second.length && first.every((byte, index) => byte === second[index]);

export async function sha256(data: BufferSource) {
  return new Uint8Array(await crypto.subtle.digest("SHA-256", data));
}

export async function encodeAssembly(assembly: CadAssembly, sourceSha256: Uint8Array) {
  if (!littleEndian || sourceSha256.length !== sha256Bytes) {
    throw new Error("A little-endian system and a SHA-256 source hash are required to build the bundled assembly.");
  }
  const manifest = new TextEncoder().encode(JSON.stringify({
    roots: assembly.roots,
    meshes: assembly.meshes.map(({ positions, normals, indices, color, groups }) => ({
      positions: positions.length, normals: normals.length, indices: indices.length, color, groups,
    })),
  }));
  const meshBytes = assembly.meshes.reduce(
    (size, mesh) => size + mesh.positions.byteLength + mesh.normals.byteLength + mesh.indices.byteLength, 0,
  );
  const geometryOffset = align4(headerBytes + manifest.length);
  const buffer = new ArrayBuffer(geometryOffset + meshBytes);
  const bytes = new Uint8Array(buffer);
  const header = new DataView(buffer);
  bytes.set(magic);
  header.setUint32(8, formatVersion, true);
  header.setUint32(12, manifest.length, true);
  bytes.set(sourceSha256, sourceHashOffset);
  bytes.set(manifest, headerBytes);

  let offset = geometryOffset;
  for (const mesh of assembly.meshes) {
    for (const data of [mesh.positions, mesh.normals, mesh.indices]) {
      bytes.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength), offset);
      offset += data.byteLength;
    }
  }

  bytes.set(await sha256(bytes.subarray(headerBytes)), payloadHashOffset);
  return buffer;
}

function validColor(value: unknown): value is CadColor {
  return Array.isArray(value) && value.length === 3
    && value.every((channel) => Number.isFinite(channel) && channel >= 0 && channel <= 1);
}

function readOccurrences(items: StoredOccurrence[], meshes: CadMesh[], ids: Set<string>): CadOccurrence[] {
  if (!Array.isArray(items)) throw new Error("The bundled assembly tree is invalid.");

  return items.map((item) => {
    if (!item || typeof item.id !== "string" || ids.has(item.id) || typeof item.name !== "string"
      || !Array.isArray(item.matrix) || item.matrix.length !== 16 || !item.matrix.every(Number.isFinite)
      || (item.mesh !== undefined && (!Number.isInteger(item.mesh) || !meshes[item.mesh]))
      || (item.color !== undefined && !validColor(item.color))
      || (item.colors !== undefined && (!Array.isArray(item.colors)
        || item.mesh === undefined || item.colors.length !== meshes[item.mesh].groups?.length
        || item.colors.some((color) => color !== null && !validColor(color))))) {
      throw new Error("The bundled assembly contains an invalid occurrence.");
    }
    ids.add(item.id);
    return {
      id: item.id,
      name: item.name,
      matrix: item.matrix,
      mesh: item.mesh,
      color: item.color,
      colors: item.colors?.map((color) => color ?? undefined),
      children: readOccurrences(item.children, meshes, ids),
    };
  });
}

export async function decodeAssembly(buffer: ArrayBuffer) {
  const bytes = new Uint8Array(buffer);
  if (!littleEndian || bytes.length < headerBytes || !equalBytes(bytes.subarray(0, magic.length), magic)) {
    throw new Error("The bundled assembly header is invalid.");
  }
  const header = new DataView(buffer);
  if (header.getUint32(8, true) !== formatVersion) {
    throw new Error("The bundled assembly format version is unsupported.");
  }

  const manifestEnd = headerBytes + header.getUint32(12, true);
  let offset = align4(manifestEnd);
  if (manifestEnd === headerBytes || offset > bytes.length) {
    throw new Error("The bundled assembly is truncated.");
  }
  if (globalThis.crypto?.subtle && !equalBytes(
    await sha256(bytes.subarray(headerBytes)), bytes.subarray(payloadHashOffset, headerBytes),
  )) {
    throw new Error("The bundled assembly checksum does not match.");
  }

  const manifest = JSON.parse(new TextDecoder().decode(bytes.subarray(headerBytes, manifestEnd))) as Manifest | null;
  if (!manifest || !Array.isArray(manifest.meshes) || manifest.meshes.length === 0) {
    throw new Error("The bundled assembly manifest is invalid.");
  }

  const meshes = manifest.meshes.map((mesh): CadMesh => {
    if (!mesh || ![mesh.positions, mesh.normals, mesh.indices].every((count) => Number.isSafeInteger(count) && count >= 0)
      || mesh.positions === 0 || mesh.positions % 3 !== 0 || mesh.indices === 0 || mesh.indices % 3 !== 0
      || offset + (mesh.positions + mesh.normals + mesh.indices) * 4 > buffer.byteLength
      || (mesh.color !== undefined && !validColor(mesh.color))) {
      throw new Error("The bundled assembly mesh data is invalid.");
    }
    if (mesh.groups !== undefined) {
      let end = 0;
      if (!Array.isArray(mesh.groups) || !mesh.groups.every((group) => {
        if (!group || group.start !== end || !Number.isSafeInteger(group.count) || group.count <= 0 || group.count % 3 !== 0) return false;
        end += group.count;
        return true;
      }) || end !== mesh.indices) {
        throw new Error("The bundled assembly material groups are invalid.");
      }
    }

    const positions = new Float32Array(buffer, offset, mesh.positions);
    offset += positions.byteLength;
    const normals = new Float32Array(buffer, offset, mesh.normals);
    offset += normals.byteLength;
    const indices = new Uint32Array(buffer, offset, mesh.indices);
    offset += indices.byteLength;
    return { positions, normals, indices, color: mesh.color, groups: mesh.groups };
  });

  if (offset !== buffer.byteLength) throw new Error("The bundled assembly contains unexpected data.");
  const roots = readOccurrences(manifest.roots, meshes, new Set());
  if (!roots.length) throw new Error("The bundled assembly contains no parts.");
  const sourceSha256 = bytes.slice(sourceHashOffset, payloadHashOffset);
  return { assembly: { roots, meshes }, sourceSha256 };
}
