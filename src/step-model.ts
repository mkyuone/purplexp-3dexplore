import type { LabelTag, OcctKernel } from "occt-wasm";
import type { CadAssembly, CadColor, CadMesh, CadOccurrence } from "./cad.ts";

type StepDocument = ReturnType<OcctKernel["importXCAFFromSTEP"]>;

interface MaterialNode {
  name?: string;
  mesh?: number;
  children?: number[];
  matrix?: number[];
  rotation?: number[];
  translation?: number[];
  scale?: number[];
}

interface MaterialPrimitive {
  material?: number;
  indices?: number;
  attributes: { POSITION: number };
}

interface MaterialScene {
  scenes: { nodes: number[] }[];
  scene?: number;
  nodes: MaterialNode[];
  meshes: { primitives: MaterialPrimitive[] }[];
  accessors: { count: number; min?: number[]; max?: number[] }[];
  materials?: { pbrMetallicRoughness?: { baseColorFactor?: number[] } }[];
}

type SurfaceColor = CadColor | undefined;
type FaceStyle = { occurrence: CadOccurrence; colors: SurfaceColor[] };
type Bounds = { min: number[]; max: number[] };
type FaceCandidate = { bounds: Bounds; color: string };
type CoordinateIndex = { coordinate: number; candidates: FaceCandidate[]; largest: number };

const toSrgb = (value: number) =>
  value <= 0.0031308 ? 12.92 * value : 1.055 * value ** (1 / 2.4) - 0.055;

const colorKey = (color: SurfaceColor) => color?.join(",") ?? "";
const positionTolerance = (magnitude: number) => 0.0002 + magnitude * 2e-7;
const nearPosition = (a: number, b: number) => Math.abs(a - b) <= positionTolerance(Math.max(Math.abs(a), Math.abs(b)));

function readMaterialScene(doc: StepDocument): MaterialScene | undefined {
  try {
    // OCCT exposes imported surface styles through its glTF writer. Reading
    // its JSON leaves the actual geometry and placements in the XCAF tree.
    const glb = doc.exportGLTF();
    const header = new DataView(glb.buffer, glb.byteOffset, glb.byteLength);
    const jsonEnd = 20 + header.getUint32(12, true);
    return JSON.parse(new TextDecoder().decode(glb.subarray(20, jsonEnd)));
  } catch {
    // Surface styling is optional, so a writer failure leaves the STEP usable.
    return undefined;
  }
}

function matchesPlacement(occurrence: CadOccurrence, node: MaterialNode) {
  const [x, y, z, w] = node.rotation ?? [0, 0, 0, 1];
  const [sx, sy, sz] = node.scale ?? [1, 1, 1];
  const [tx, ty, tz] = node.translation ?? [0, 0, 0];
  const matrix = node.matrix ?? [
    (1 - 2 * (y * y + z * z)) * sx, 2 * (x * y + w * z) * sx, 2 * (x * z - w * y) * sx, 0,
    2 * (x * y - w * z) * sy, (1 - 2 * (x * x + z * z)) * sy, 2 * (y * z + w * x) * sy, 0,
    2 * (x * z + w * y) * sz, 2 * (y * z - w * x) * sz, (1 - 2 * (x * x + y * y)) * sz, 0,
    tx, ty, tz, 1,
  ];
  return occurrence.matrix.every((value, index) => index >= 12 && index < 15
    ? nearPosition(value, matrix[index] * 1000)
    : Math.abs(value - matrix[index]) <= 1e-7);
}

function faceBounds(mesh: CadMesh, start: number, count: number): Bounds {
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  for (let i = start; i < start + count; i++) {
    const vertex = mesh.indices[i] * 3;
    for (let axis = 0; axis < 3; axis++) {
      const value = mesh.positions[vertex + axis];
      min[axis] = Math.min(min[axis], value);
      max[axis] = Math.max(max[axis], value);
    }
  }
  return { min, max };
}

function sameBounds(a: Bounds, b: Bounds) {
  return a.min.every((value, axis) => nearPosition(value, b.min[axis]))
    && a.max.every((value, axis) => nearPosition(value, b.max[axis]));
}

const coordinateOf = (bounds: Bounds, coordinate: number) =>
  coordinate < 3 ? bounds.min[coordinate] : bounds.max[coordinate - 3];

function indexCandidates(candidates: FaceCandidate[]): CoordinateIndex[] {
  if (new Set(candidates.map((candidate) => candidate.color)).size < 2) return [];
  return Array.from({ length: 6 }, (_, coordinate) => {
    const value = (candidate: FaceCandidate) => coordinateOf(candidate.bounds, coordinate);
    const sorted = candidates.slice().sort((a, b) => value(a) - value(b));
    const largest = Math.max(Math.abs(value(sorted[0])), Math.abs(value(sorted[sorted.length - 1])));
    return { coordinate, candidates: sorted, largest };
  });
}

function boundary(index: CoordinateIndex, limit: number, includeEqual: boolean) {
  let low = 0;
  let high = index.candidates.length;
  while (low < high) {
    const middle = (low + high) >>> 1;
    const value = coordinateOf(index.candidates[middle].bounds, index.coordinate);
    if (value < limit || (includeEqual && value === limit)) low = middle + 1;
    else high = middle;
  }
  return low;
}

function hasConflictingColor(indexes: CoordinateIndex[], bounds: Bounds, color: string) {
  let shortest: { candidates: FaceCandidate[]; start: number; end: number } | undefined;
  for (const index of indexes) {
    const value = coordinateOf(bounds, index.coordinate);
    // Use a conservative interval and keep sameBounds as the final check.
    const radius = 2 * positionTolerance(Math.max(Math.abs(value), index.largest));
    const start = boundary(index, value - radius, false);
    const end = boundary(index, value + radius, true);
    if (end - start <= 1) return false; // Only the already-verified face can be present.
    if (!shortest || end - start < shortest.end - shortest.start) {
      shortest = { candidates: index.candidates, start, end };
    }
  }
  if (!shortest) return false;
  for (let i = shortest.start; i < shortest.end; i++) {
    const candidate = shortest.candidates[i];
    if (candidate.color !== color && sameBounds(bounds, candidate.bounds)) return true;
  }
  return false;
}

function matchesFaces(
  scene: MaterialScene, primitives: MaterialPrimitive[], colors: SurfaceColor[], mesh: CadMesh, faces: Int32Array,
) {
  if (faces.length !== primitives.length * 3) return false;
  const candidatesByCount = new Map<number, FaceCandidate[]>();
  const boundsInOrder: Bounds[] = [];
  for (const [face, primitive] of primitives.entries()) {
    const accessor = scene.accessors[primitive.attributes.POSITION];
    const count = faces[face * 3 + 1];
    if (primitive.indices === undefined || scene.accessors[primitive.indices]?.count !== count
      || accessor?.min?.length !== 3 || accessor.max?.length !== 3) return false;
    const bounds = { min: accessor.min.map((value) => value * 1000), max: accessor.max.map((value) => value * 1000) };
    boundsInOrder.push(bounds);
    const candidates = candidatesByCount.get(count) ?? [];
    candidates.push({ bounds, color: colorKey(colors[face]) });
    candidatesByCount.set(count, candidates);
  }
  const indexByCount = new Map(Array.from(candidatesByCount, ([count, candidates]) => [count, indexCandidates(candidates)]));

  // OCCT 5.8.1 reports index offsets/counts here, including three indices per triangle.
  // Use exporter order only after verifying complete coverage and an unambiguous color match.
  let end = 0;
  for (let face = 0; face < primitives.length; face++) {
    const start = faces[face * 3];
    const count = faces[face * 3 + 1];
    if (start !== end || count <= 0 || count % 3 !== 0 || start + count > mesh.indices.length) return false;
    end = start + count;
    const bounds = faceBounds(mesh, start, count);
    if (!sameBounds(bounds, boundsInOrder[face])) return false;
    const color = colorKey(colors[face]);
    const index = indexByCount.get(count);
    if (index && hasConflictingColor(index, bounds, color)) return false;
  }
  return end === mesh.indices.length;
}

function groupStyledFaces(mesh: CadMesh, faces: Int32Array, styles: FaceStyle[]) {
  const partitions = new Map<string, { faces: number[]; colors: SurfaceColor[] }>();
  for (let face = 0; face < faces.length / 3; face++) {
    const colors = styles.map((style) => style.colors[face]);
    const key = colors.map(colorKey).join("|");
    const partition = partitions.get(key) ?? { faces: [], colors };
    partition.faces.push(face * 3);
    partitions.set(key, partition);
  }

  const indices = new Uint32Array(mesh.indices.length);
  const groups: NonNullable<CadMesh["groups"]> = [];
  const palettes = styles.map((): SurfaceColor[] => []);
  let offset = 0;
  for (const partition of partitions.values()) {
    const start = offset;
    for (const face of partition.faces) {
      const faceStart = faces[face];
      const faceEnd = faceStart + faces[face + 1];
      indices.set(mesh.indices.subarray(faceStart, faceEnd), offset);
      offset += faceEnd - faceStart;
    }
    groups.push({ start, count: offset - start });
    partition.colors.forEach((color, index) => palettes[index].push(color));
  }
  mesh.indices = indices;
  mesh.groups = groups;
  styles.forEach(({ occurrence }, index) => { occurrence.colors = palettes[index]; });
}

function applyMaterialColors(
  scene: MaterialScene, roots: CadOccurrence[], meshes: CadMesh[], faceGroups: (Int32Array | undefined)[],
  sourceNames: Map<string, string>,
) {
  const colors = scene.materials?.map((material): SurfaceColor => {
    const factor = material.pbrMetallicRoughness?.baseColorFactor;
    if (factor) return [toSrgb(factor[0]), toSrgb(factor[1]), toSrgb(factor[2])];
  });
  const stylesByMesh = new Map<number, FaceStyle[]>();

  const apply = (occurrences: CadOccurrence[], nodeIds: number[]) => {
    if (occurrences.length !== nodeIds.length) return;
    occurrences.forEach((occurrence, index) => {
      const node = scene.nodes[nodeIds[index]];
      const sourceName = sourceNames.get(occurrence.id);
      if (!node || (node.name !== occurrence.name && (!sourceName || node.name !== sourceName))
        || !matchesPlacement(occurrence, node)) return;
      if (occurrence.mesh !== undefined && node.mesh !== undefined) {
        const primitives = scene.meshes[node.mesh]?.primitives ?? [];
        const faceColors: SurfaceColor[] = [];
        const weights = new Map<string, { color: CadColor; count: number }>();
        for (const primitive of primitives) {
          const color = primitive.material === undefined ? undefined : colors?.[primitive.material];
          faceColors.push(color ?? occurrence.color);
          if (!color) continue;
          const key = colorKey(color);
          const weight = weights.get(key) ?? { color, count: 0 };
          weight.count += scene.accessors[primitive.indices ?? primitive.attributes.POSITION]?.count ?? 0;
          weights.set(key, weight);
        }
        const faces = faceGroups[occurrence.mesh];
        if (faces && new Set(faceColors.map(colorKey)).size > 1
          && matchesFaces(scene, primitives, faceColors, meshes[occurrence.mesh], faces)) {
          const styles = stylesByMesh.get(occurrence.mesh) ?? [];
          styles.push({ occurrence, colors: faceColors });
          stylesByMesh.set(occurrence.mesh, styles);
        }
        const dominant = Array.from(weights.values()).sort((a, b) => b.count - a.count)[0];
        occurrence.color = dominant?.color ?? occurrence.color;
      }
      apply(occurrence.children, node.children ?? []);
    });
  };
  apply(roots, scene.scenes[scene.scene ?? 0]?.nodes ?? []);
  for (const [mesh, styles] of stylesByMesh) groupStyledFaces(meshes[mesh], faceGroups[mesh]!, styles);
}

export function readStep(kernel: OcctKernel, data: ArrayBuffer): CadAssembly {
  const text = new TextDecoder().decode(data);
  if (!/^\s*ISO-10303-21\s*;/i.test(text)) {
    throw new Error("This file is not a STEP model. Choose a .step or .stp export.");
  }

  const doc = kernel.importXCAFFromSTEP(text);
  const hasStyles = /\bSTYLED_ITEM\s*\(/i.test(text);
  const meshes: CadMesh[] = [];
  const faceGroups: (Int32Array | undefined)[] = [];
  const sourceNames = new Map<string, string>();
  const prototypes = new Map<LabelTag, {
    name: string;
    color?: CadColor;
    children: LabelTag[];
    mesh?: number;
  }>();

  const getPrototype = (label: LabelTag) => {
    const cached = prototypes.get(label);
    if (cached) return cached;
    const info = doc.getLabelInfo(label);
    const children = doc.getChildren(label);
    const { shapeHandle } = info;
    let mesh: number | undefined;
    try {
      if (shapeHandle !== null && children.length === 0) {
        const tessellation = hasStyles ? kernel.meshShape(shapeHandle) : kernel.tessellate(shapeHandle);
        if (tessellation.indices.length > 0) {
          const { positions, normals, indices } = tessellation;
          mesh = meshes.push({ positions, normals, indices }) - 1;
          faceGroups[mesh] = tessellation.faceGroups;
        }
      }
    } finally {
      if (shapeHandle !== null) kernel.release(shapeHandle);
    }
    const prototype = {
      name: info.name.trim(),
      color: info.hasColor ? info.color.map(toSrgb) as CadColor : undefined,
      children,
      mesh,
    };
    prototypes.set(label, prototype);
    return prototype;
  };

  const ancestors = new Set<LabelTag>();
  const visit = (label: LabelTag, prefix: string): CadOccurrence => {
    const referred = doc.getReferredLabel(label);
    const prototypeLabel = referred ?? label;
    if (ancestors.has(prototypeLabel)) throw new Error("The STEP assembly contains a circular reference.");
    const prototype = getPrototype(prototypeLabel);

    const id = prefix + label;
    let name = prototype.name;
    let color = prototype.color;
    if (referred !== null) {
      const occurrence = doc.getLabelInfo(label);
      if (occurrence.shapeHandle !== null) kernel.release(occurrence.shapeHandle);
      const occurrenceName = occurrence.name.trim();
      sourceNames.set(id, occurrenceName);
      if (occurrenceName && !occurrenceName.startsWith("=>[")) name = occurrenceName;
      if (occurrence.hasColor) color = occurrence.color.map(toSrgb) as CadColor;
    }

    const [xx, xy, xz, tx, yx, yy, yz, ty, zx, zy, zz, tz] = doc.getLocation(label);
    ancestors.add(prototypeLabel);
    const children = prototype.children.map((child) => visit(child, id + "/"));
    ancestors.delete(prototypeLabel);

    return {
      id,
      name: name || (children.length ? "Assembly" : "Part") + " " + label,
      matrix: [xx, yx, zx, 0, xy, yy, zy, 0, xz, yz, zz, 0, tx, ty, tz, 1],
      mesh: prototype.mesh,
      color,
      children,
    };
  };

  try {
    const roots = doc.getRoots().map((label) => visit(label, ""));
    if (meshes.length === 0) throw new Error("The STEP file contains no visible surfaces.");

    if (hasStyles) {
      const materialScene = readMaterialScene(doc);
      if (materialScene) applyMaterialColors(materialScene, roots, meshes, faceGroups, sourceNames);
    }
    return { roots, meshes };
  } finally {
    doc.close();
  }
}
