import type { LabelTag, OcctKernel } from "occt-wasm";
import type { CadAssembly, CadColor, CadMesh, CadOccurrence } from "./cad.ts";

type StepDocument = ReturnType<OcctKernel["importXCAFFromSTEP"]>;

interface MaterialScene {
  scenes: { nodes: number[] }[];
  scene?: number;
  nodes: { name?: string; mesh?: number; children?: number[] }[];
  meshes: {
    primitives: { material?: number; indices?: number; attributes: { POSITION: number } }[];
  }[];
  accessors: { count: number }[];
  materials?: { pbrMetallicRoughness?: { baseColorFactor?: number[] } }[];
}

const toSrgb = (value: number) =>
  value <= 0.0031308 ? 12.92 * value : 1.055 * value ** (1 / 2.4) - 0.055;

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

function materialColor(scene: MaterialScene, meshIndex: number): CadColor | undefined {
  const weights = new Map<number, number>();
  for (const primitive of scene.meshes[meshIndex].primitives) {
    if (primitive.material === undefined) continue;
    const accessor = scene.accessors[primitive.indices ?? primitive.attributes.POSITION];
    weights.set(primitive.material, (weights.get(primitive.material) ?? 0) + accessor.count);
  }
  const dominant = Array.from(weights).sort((a, b) => b[1] - a[1])[0]?.[0];
  if (dominant === undefined) return;
  const color = scene.materials?.[dominant]?.pbrMetallicRoughness?.baseColorFactor;
  if (color) return [toSrgb(color[0]), toSrgb(color[1]), toSrgb(color[2])];
}

function applyMaterialColors(scene: MaterialScene, roots: CadOccurrence[]) {
  const apply = (occurrences: CadOccurrence[], nodeIds: number[]) => {
    if (occurrences.length !== nodeIds.length) return;
    occurrences.forEach((occurrence, index) => {
      const node = scene.nodes[nodeIds[index]];
      if (!node || node.name !== occurrence.name) return;
      if (occurrence.mesh !== undefined && node.mesh !== undefined) {
        occurrence.color = materialColor(scene, node.mesh) ?? occurrence.color;
      }
      apply(occurrence.children, node.children ?? []);
    });
  };
  apply(roots, scene.scenes[scene.scene ?? 0]?.nodes ?? []);
}

export function readStep(kernel: OcctKernel, data: ArrayBuffer): CadAssembly {
  const text = new TextDecoder().decode(data);
  if (!/^\s*ISO-10303-21\s*;/i.test(text)) {
    throw new Error("This file is not a STEP model. Choose a .step or .stp export.");
  }

  const doc = kernel.importXCAFFromSTEP(text);
  const meshes: CadMesh[] = [];
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
        const tessellation = kernel.tessellate(shapeHandle);
        if (tessellation.indices.length > 0) {
          const { positions, normals, indices } = tessellation;
          mesh = meshes.push({ positions, normals, indices }) - 1;
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

    let name = prototype.name;
    let color = prototype.color;
    if (referred !== null) {
      const occurrence = doc.getLabelInfo(label);
      if (occurrence.shapeHandle !== null) kernel.release(occurrence.shapeHandle);
      const occurrenceName = occurrence.name.trim();
      if (occurrenceName && !occurrenceName.startsWith("=>[")) name = occurrenceName;
      if (occurrence.hasColor) color = occurrence.color.map(toSrgb) as CadColor;
    }

    const id = prefix + label;
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

    if (/\bSTYLED_ITEM\s*\(/i.test(text)) {
      const materialScene = readMaterialScene(doc);
      if (materialScene) applyMaterialColors(materialScene, roots);
    }
    return { roots, meshes };
  } finally {
    doc.close();
  }
}
