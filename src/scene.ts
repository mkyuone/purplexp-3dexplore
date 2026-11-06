import {
  ACESFilmicToneMapping, Box3, BufferAttribute, BufferGeometry, Color, DirectionalLight,
  EdgesGeometry, Group, HemisphereLight, LineBasicMaterial, LineSegments,
  MathUtils, Matrix4, Mesh, MeshStandardMaterial, PCFShadowMap, PerspectiveCamera, Plane,
  PlaneGeometry, PMREMGenerator, Quaternion, Raycaster, Scene, ShaderMaterial, ShadowMaterial,
  Spherical, SRGBColorSpace, Vector2, Vector3, WebGLRenderer, type WebGLRenderTarget,
} from 'three';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import type { CadAssembly, CadMesh, CadOccurrence } from './cad';
import type { ObjectPose } from './gestures';

type OccurrenceMesh = Mesh<BufferGeometry, MeshStandardMaterial | MeshStandardMaterial[]>;

interface SceneOccurrence {
  id: string;
  parent: SceneOccurrence | null;
  pivot: Vector3;
  placement: Group;
  editable: Group;
  mesh?: OccurrenceMesh;
  sourceColors?: Color[];
  outline?: LineSegments;
}

export type CameraView = 'front' | 'side' | 'isometric';

const cameraDirections: Record<CameraView, Vector3> = {
  front: new Vector3(0, 0, 1),
  side: new Vector3(1, 0, 0),
  isometric: new Vector3(0.82, 0.48, 1.25).normalize(),
};

function makeGeometry(mesh: CadMesh) {
  const geometry = new BufferGeometry();
  geometry.setAttribute('position', new BufferAttribute(mesh.positions, 3));
  geometry.setIndex(new BufferAttribute(mesh.indices, 1));
  mesh.groups?.forEach(({ start, count }, index) => geometry.addGroup(start, count, index));
  if (mesh.normals.length === mesh.positions.length) {
    geometry.setAttribute('normal', new BufferAttribute(mesh.normals, 3));
  } else {
    geometry.computeVertexNormals();
  }
  geometry.computeBoundingBox();
  return geometry;
}

function materialsOf(mesh?: OccurrenceMesh) {
  if (!mesh) return [];
  return Array.isArray(mesh.material) ? mesh.material : [mesh.material];
}

function localPose(node: Group): ObjectPose {
  return {
    position: node.position.clone(),
    rotation: node.quaternion.clone(),
    scale: node.scale.x,
  };
}

function applyPose(node: Group, pose: ObjectPose) {
  node.position.copy(pose.position);
  node.quaternion.copy(pose.rotation);
  node.scale.setScalar(pose.scale);
}

function samePose(a: ObjectPose, b: ObjectPose) {
  return a.position.distanceToSquared(b.position) < 1e-10
    && a.rotation.angleTo(b.rotation) < 1e-5
    && Math.abs(a.scale - b.scale) < 1e-5;
}

function makeStudioMap(renderer: WebGLRenderer) {
  if (!renderer.extensions.has('EXT_color_buffer_float')) return null;
  const studio = new RoomEnvironment();
  const pmrem = new PMREMGenerator(renderer);
  try {
    return pmrem.fromScene(studio, 0.05, 0.1, 60, { size: 128 });
  } finally {
    studio.dispose();
    pmrem.dispose();
  }
}

export class CadScene {
  private readonly world = new Scene();
  private readonly camera = new PerspectiveCamera(34, 1, 0.01, 1000);
  private readonly renderer: WebGLRenderer;
  private readonly studioMap: WebGLRenderTarget | null;
  private readonly observer: ResizeObserver;
  private readonly assemblyRoot = new Group();
  private readonly target = new Vector3();
  private readonly raycaster = new Raycaster();
  private readonly ndc = new Vector2();
  private readonly picks: Array<{ x: number; y: number; id: string | null }> = [];
  private readonly keyLight = new DirectionalLight(0xfffaf4, 2.35);
  private readonly ground = new Mesh(
    new PlaneGeometry(45, 45),
    new ShadowMaterial({ color: 0x090712, opacity: 0.37, depthWrite: false }),
  );
  private readonly floorGlow = new Mesh(
    new PlaneGeometry(17, 12),
    new ShaderMaterial({
      transparent: true,
      depthWrite: false,
      uniforms: { glow: { value: new Color('#71588f') } },
      vertexShader: `
        varying vec2 floorPosition;
        void main() {
          floorPosition = uv * 2.0 - 1.0;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: `
        uniform vec3 glow;
        varying vec2 floorPosition;
        void main() {
          float falloff = 1.0 - smoothstep(0.0, 1.0, length(floorPosition));
          gl_FragColor = vec4(glow, falloff * falloff * 0.38);
          #include <colorspace_fragment>
        }
      `,
    }),
  );
  private readonly selectedLine = new LineBasicMaterial({
    color: 0xb49cff, depthWrite: false, toneMapped: false,
  });
  private readonly hoverLine = new LineBasicMaterial({
    color: 0x62d9d1, depthWrite: false, toneMapped: false,
  });
  private readonly edges = new Map<BufferGeometry, EdgesGeometry>();
  private readonly occurrences = new Map<string, SceneOccurrence>();
  private readonly meshes: OccurrenceMesh[] = [];
  private readonly geometries: BufferGeometry[] = [];
  private readonly undoStack: Array<() => void> = [];
  private selection: string | null = null;
  private hovered: string[] = [];
  private frame: number | null = null;
  private modelScale = 1;
  private fitOnResize = false;

  constructor(private readonly canvas: HTMLCanvasElement) {
    this.renderer = new WebGLRenderer({ canvas, antialias: true, alpha: true });
    this.renderer.toneMapping = ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.04;
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = PCFShadowMap;
    this.studioMap = makeStudioMap(this.renderer);
    this.world.environment = this.studioMap?.texture ?? null;
    this.world.environmentIntensity = 0.68;

    const key = this.keyLight;
    key.position.set(-3.5, 8, 5.5);
    key.castShadow = true;
    key.shadow.autoUpdate = false;
    key.shadow.needsUpdate = true;
    key.shadow.mapSize.set(1536, 1536);
    Object.assign(key.shadow.camera, { left: -7, right: 7, top: 7, bottom: -7, far: 28 });
    key.shadow.camera.updateProjectionMatrix();
    key.shadow.normalBias = 0.012;
    key.shadow.bias = -0.00015;
    key.shadow.radius = 5;
    const fill = new DirectionalLight(0xcddbf4, 0.55);
    fill.position.set(6, 3, 2);
    const rim = new DirectionalLight(0xd0b9f4, 1.22);
    rim.position.set(-3, 4, -6);

    this.ground.rotation.x = -Math.PI / 2;
    this.ground.receiveShadow = true;
    this.floorGlow.rotation.x = -Math.PI / 2;
    this.floorGlow.renderOrder = 1;
    this.ground.renderOrder = 2;
    this.world.add(new HemisphereLight(0xf2f3ff, 0x918a9f, this.studioMap ? 0.55 : 1.15), key, fill, rim,
      this.floorGlow, this.ground, this.assemblyRoot);

    this.camera.position.copy(cameraDirections.isometric).multiplyScalar(12);
    this.aimCamera();
    this.observer = new ResizeObserver(() => this.resize());
    this.observer.observe(canvas);
    this.resize();
  }

  load(assembly: CadAssembly) {
    this.clearAssembly();
    this.keyLight.shadow.needsUpdate = true;
    this.assemblyRoot.rotation.set(-Math.PI / 2, 0, 0);
    this.assemblyRoot.position.set(0, 0, 0);
    this.assemblyRoot.scale.setScalar(1);
    for (const mesh of assembly.meshes) this.geometries.push(makeGeometry(mesh));
    for (const root of assembly.roots) {
      this.assemblyRoot.add(this.buildOccurrence(root, null, assembly.meshes).placement);
    }

    const bounds = new Box3().setFromObject(this.assemblyRoot);
    if (bounds.isEmpty()) {
      this.requestRender(true);
      return;
    }

    const center = bounds.getCenter(new Vector3());
    this.modelScale = 5.6 / Math.max(bounds.getSize(new Vector3()).length(), 1e-9);
    this.assemblyRoot.scale.setScalar(this.modelScale);
    this.assemblyRoot.position.copy(center).multiplyScalar(-this.modelScale);
    this.assemblyRoot.updateMatrixWorld(true);
    this.ground.position.y = (bounds.min.y - center.y) * this.modelScale - 0.015;
    this.floorGlow.position.y = this.ground.position.y - 0.008;

    this.fit('isometric');
  }

  private buildOccurrence(node: CadOccurrence, parent: SceneOccurrence | null, meshes: CadMesh[]) {
    const placement = new Group();
    const matrix = new Matrix4().fromArray(node.matrix);
    // Keep the complete imported affine placement on its own node.
    placement.matrixAutoUpdate = false;
    placement.matrix.copy(matrix);

    const editable = new Group();
    const contents = new Group();
    placement.add(editable);
    editable.add(contents);
    const occurrence: SceneOccurrence = {
      id: node.id, parent, pivot: new Vector3(), placement, editable,
    };
    this.occurrences.set(node.id, occurrence);

    const bounds = new Box3();
    if (node.mesh !== undefined) {
      const geometry = this.geometries[node.mesh];
      const fallback = meshes[node.mesh].color;
      const palette = node.colors?.length === geometry.groups.length ? node.colors : [node.color ?? fallback];
      const sourceColors = palette.map((entry) => {
        const values = entry ?? fallback;
        return values ? new Color().setRGB(...values, SRGBColorSpace) : new Color('#a3adbc');
      });
      const materials = sourceColors.map((color) => new MeshStandardMaterial({ color, roughness: 0.47 }));
      const mesh = new Mesh(geometry, materials.length > 1 ? materials : materials[0]);
      mesh.userData.occurrenceId = node.id;
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      contents.add(mesh);
      this.meshes.push(mesh);
      occurrence.mesh = mesh;
      occurrence.sourceColors = sourceColors;
      if (geometry.boundingBox) bounds.copy(geometry.boundingBox);
    }

    for (const child of node.children) {
      const built = this.buildOccurrence(child, occurrence, meshes);
      contents.add(built.placement);
      bounds.union(built.bounds.applyMatrix4(built.placement.matrix));
    }
    bounds.getCenter(occurrence.pivot);
    editable.position.copy(occurrence.pivot);
    contents.position.copy(occurrence.pivot).negate();
    return { placement, bounds };
  }

  resize() {
    const width = this.canvas.clientWidth;
    const height = this.canvas.clientHeight;
    if (!width || !height) return;
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setSize(width, height, false);
    const aspect = width / height;
    const aspectChanged = this.camera.aspect !== aspect;
    this.camera.aspect = aspect;
    this.camera.updateProjectionMatrix();
    if (aspectChanged && this.fitOnResize && this.meshes.length) {
      this.fit();
      return;
    }
    this.requestRender(true);
  }

  fit(view?: CameraView) {
    const bounds = new Box3().setFromObject(this.assemblyRoot);
    if (bounds.isEmpty()) return;
    const direction = view
      ? cameraDirections[view]
      : this.camera.position.clone().sub(this.target).normalize();
    bounds.getCenter(this.target);
    const right = new Vector3(0, 1, 0).cross(direction).normalize();
    const up = direction.clone().cross(right);
    const vertical = Math.tan(MathUtils.degToRad(this.camera.fov / 2));
    const horizontal = vertical * this.camera.aspect;
    let distance = 1;
    const corner = new Vector3();
    for (const mesh of this.meshes) {
      const box = mesh.geometry.boundingBox;
      if (!box || box.isEmpty()) continue;
      for (const x of [box.min.x, box.max.x]) {
        for (const y of [box.min.y, box.max.y]) {
          for (const z of [box.min.z, box.max.z]) {
            corner.set(x, y, z).applyMatrix4(mesh.matrixWorld).sub(this.target);
            const clearance = Math.max(
              Math.abs(corner.dot(right)) / (horizontal * 0.86),
              Math.abs(corner.dot(up)) / (vertical * 0.72),
            );
            distance = Math.max(distance, corner.dot(direction) + clearance);
          }
        }
      }
    }
    this.camera.position.copy(this.target).addScaledVector(direction, distance);
    this.fitOnResize = true;
    this.aimCamera();
    this.requestRender(true);
  }

  orbit(dx: number, dy: number) {
    const angles = new Spherical().setFromVector3(this.camera.position.clone().sub(this.target));
    angles.theta -= dx * Math.PI * 1.7;
    angles.phi = MathUtils.clamp(angles.phi - dy * Math.PI * 1.4, 0.05, Math.PI - 0.05);
    this.camera.position.setFromSpherical(angles).add(this.target);
    this.fitOnResize = false;
    this.aimCamera();
    this.requestRender(true);
  }

  zoom(factor: number) {
    const offset = this.camera.position.clone().sub(this.target);
    offset.setLength(MathUtils.clamp(offset.length() / factor, 1.4, 300));
    this.camera.position.copy(this.target).add(offset);
    this.fitOnResize = false;
    this.aimCamera();
    this.requestRender(true);
  }

  private aimCamera() {
    this.camera.lookAt(this.target);
    this.camera.updateMatrixWorld();
  }

  private castRay(x: number, y: number) {
    this.raycaster.setFromCamera(this.ndc.set(x * 2 - 1, 1 - y * 2), this.camera);
  }

  screenPoint(x: number, y: number, reference: Vector3, depth = 0) {
    const towardViewer = new Vector3(0, 0, 1).applyQuaternion(this.camera.quaternion);
    this.castRay(x, y);
    const plane = new Plane().setFromNormalAndCoplanarPoint(towardViewer, reference);
    const point = this.raycaster.ray.intersectPlane(plane, new Vector3()) ?? reference.clone();
    const referenceDistance = Math.abs(this.camera.position.clone().sub(reference).dot(towardViewer));
    const width = 2 * referenceDistance * Math.tan(MathUtils.degToRad(this.camera.fov / 2))
      * this.camera.aspect;
    return point.addScaledVector(towardViewer, depth * width);
  }

  pick(x: number, y: number) {
    if (x < 0 || x > 1 || y < 0 || y > 1) return null;
    const cached = this.picks.find((pick) => pick.x === x && pick.y === y);
    if (cached) return cached.id;
    this.assemblyRoot.updateMatrixWorld(true);
    this.castRay(x, y);
    const hit = this.raycaster.intersectObjects(this.meshes, false)[0];
    const id = hit ? hit.object.userData.occurrenceId as string : null;
    this.picks.unshift({ x, y, id });
    if (this.picks.length > 2) this.picks.pop();
    return id;
  }

  isWithin(id: string, group: string) {
    let occurrence = this.occurrences.get(id);
    while (occurrence) {
      if (occurrence.id === group) return true;
      occurrence = occurrence.parent ?? undefined;
    }
    return false;
  }

  getSelected() {
    return this.selection;
  }

  select(id: string | null) {
    if (this.selection === id) return;
    this.selection = id;
    this.updateHighlights();
  }

  hover(ids: string[]) {
    if (ids.length === this.hovered.length && ids.every((id, index) => id === this.hovered[index])) return;
    this.hovered = ids;
    this.updateHighlights();
  }

  private updateHighlights() {
    for (const occurrence of this.occurrences.values()) {
      const mesh = occurrence.mesh;
      if (!mesh) continue;
      const selected = this.selection !== null && this.isWithin(occurrence.id, this.selection);
      const hovered = this.hovered.includes(occurrence.id);
      if (!selected && !hovered) {
        if (occurrence.outline) occurrence.outline.visible = false;
        for (const material of materialsOf(mesh)) material.emissive.setHex(0);
        continue;
      }

      if (!occurrence.outline) {
        let edges = this.edges.get(mesh.geometry);
        if (!edges) {
          edges = new EdgesGeometry(mesh.geometry, 24);
          this.edges.set(mesh.geometry, edges);
        }
        occurrence.outline = new LineSegments(edges, this.hoverLine);
        occurrence.outline.renderOrder = 1;
        mesh.add(occurrence.outline);
      }
      occurrence.outline.visible = true;
      occurrence.outline.material = selected ? this.selectedLine : this.hoverLine;
      for (const material of materialsOf(mesh)) {
        material.emissive.setHex(selected ? 0x554098 : 0x19767f);
        material.emissiveIntensity = 0.24;
      }
    }
    this.requestRender();
  }

  getPose(id: string): ObjectPose | null {
    const occurrence = this.occurrences.get(id);
    if (!occurrence) return null;
    const editable = occurrence.editable;
    const position = new Vector3();
    const rotation = new Quaternion();
    const scale = new Vector3();
    editable.updateWorldMatrix(true, false);
    editable.matrixWorld.decompose(position, rotation, scale);
    return { position, rotation, scale: Math.abs(scale.x) / this.modelScale };
  }

  private toLocalPose(occurrence: SceneOccurrence, pose: ObjectPose): ObjectPose {
    occurrence.placement.updateWorldMatrix(true, false);
    const parent = occurrence.placement.matrixWorld;
    const matrix = new Matrix4().compose(
      pose.position, pose.rotation, new Vector3().setScalar(pose.scale * this.modelScale),
    );
    matrix.premultiply(parent.clone().invert());
    const position = new Vector3();
    const rotation = new Quaternion();
    const scale = new Vector3();
    matrix.decompose(position, rotation, scale);
    return { position, rotation, scale: Math.abs(scale.x) };
  }

  setPose(id: string, pose: ObjectPose) {
    const occurrence = this.occurrences.get(id);
    if (!occurrence) return;
    applyPose(occurrence.editable, this.toLocalPose(occurrence, pose));
    this.keyLight.shadow.needsUpdate = true;
    this.requestRender(true);
  }

  commitGrip(id: string, startingPose: ObjectPose) {
    const occurrence = this.occurrences.get(id);
    if (!occurrence) return;
    const before = this.toLocalPose(occurrence, startingPose);
    if (!samePose(before, localPose(occurrence.editable))) {
      this.pushUndo(() => applyPose(occurrence.editable, before));
    }
  }

  describe(id: string) {
    const occurrence = this.occurrences.get(id);
    if (!occurrence) return null;
    const offset = occurrence.editable.position.clone().sub(occurrence.pivot).toArray();
    const materials = materialsOf(occurrence.mesh);
    const color = materials[0]?.color;
    const uniform = color && materials.every((material) => material.color.equals(color));
    return { color: uniform ? `#${color.getHexString()}` : null, offset, scale: occurrence.editable.scale.x };
  }

  color(id: string, cssHex: string) {
    const next = new Color(cssHex);
    const restore: Array<() => void> = [];
    for (const occurrence of this.occurrences.values()) {
      if (!this.isWithin(occurrence.id, id)) continue;
      for (const { color } of materialsOf(occurrence.mesh)) {
        if (color.equals(next)) continue;
        const before = color.clone();
        restore.push(() => color.copy(before));
        color.copy(next);
      }
    }
    if (restore.length) this.pushUndo(() => restore.forEach((undo) => undo()));
    this.requestRender();
  }

  reset(id?: string) {
    const restore: Array<() => void> = [];
    for (const occurrence of this.occurrences.values()) {
      if (id !== undefined && !this.isWithin(occurrence.id, id)) continue;
      const initial = { position: occurrence.pivot, rotation: new Quaternion(), scale: 1 };
      const before = localPose(occurrence.editable);
      if (!samePose(before, initial)) {
        applyPose(occurrence.editable, initial);
        restore.push(() => applyPose(occurrence.editable, before));
      }
      for (const [index, { color }] of materialsOf(occurrence.mesh).entries()) {
        const source = occurrence.sourceColors?.[index];
        if (!source || color.equals(source)) continue;
        const previous = color.clone();
        color.copy(source);
        restore.push(() => color.copy(previous));
      }
    }
    if (restore.length) this.pushUndo(() => restore.forEach((undo) => undo()));
    this.keyLight.shadow.needsUpdate = true;
    this.requestRender(true);
  }

  canUndo() {
    return this.undoStack.length > 0;
  }

  undo() {
    this.undoStack.pop()?.();
    this.keyLight.shadow.needsUpdate = true;
    this.requestRender(true);
  }

  private pushUndo(restore: () => void) {
    this.undoStack.push(restore);
    if (this.undoStack.length > 100) this.undoStack.shift();
  }

  private requestRender(picksChanged = false) {
    if (picksChanged) this.picks.length = 0;
    if (this.frame !== null) return;
    this.frame = requestAnimationFrame(() => {
      this.frame = null;
      this.renderer.render(this.world, this.camera);
    });
  }

  private clearAssembly() {
    this.assemblyRoot.clear();
    for (const mesh of this.meshes) {
      for (const material of materialsOf(mesh)) material.dispose();
    }
    this.edges.forEach((edge) => edge.dispose());
    this.geometries.forEach((geometry) => geometry.dispose());
    this.edges.clear();
    this.occurrences.clear();
    this.meshes.length = 0;
    this.geometries.length = 0;
    this.undoStack.length = 0;
    this.selection = null;
    this.hovered = [];
  }

  dispose() {
    this.observer.disconnect();
    if (this.frame !== null) cancelAnimationFrame(this.frame);
    this.clearAssembly();
    this.floorGlow.geometry.dispose();
    this.floorGlow.material.dispose();
    this.ground.geometry.dispose();
    this.ground.material.dispose();
    this.keyLight.shadow.dispose();
    this.studioMap?.dispose();
    this.selectedLine.dispose();
    this.hoverLine.dispose();
    this.renderer.dispose();
  }
}
