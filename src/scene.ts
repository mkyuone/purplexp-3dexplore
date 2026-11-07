import {
  Box3, BufferAttribute, BufferGeometry, Color, DirectionalLight,
  Group, HemisphereLight,
  MathUtils, Matrix4, Mesh, MeshPhysicalMaterial, MeshStandardMaterial, NeutralToneMapping, PCFShadowMap, PerspectiveCamera, Plane,
  PlaneGeometry, PMREMGenerator, Quaternion, Raycaster, Scene, ShaderMaterial, ShadowMaterial,
  Sphere, Spherical, SRGBColorSpace, Vector2, Vector3, WebGLRenderer, type MeshPhysicalMaterialParameters, type WebGLRenderTarget,
} from 'three';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import type { CadAssembly, CadMesh, CadOccurrence } from './cad';
import type { ObjectPose } from './gestures';
import { CALIBRATION_INSET_X, CALIBRATION_INSET_Y } from './reach';

type OccurrenceMesh = Mesh<BufferGeometry, MeshStandardMaterial | MeshStandardMaterial[]>;

interface SceneOccurrence {
  id: string;
  parent: SceneOccurrence | null;
  children: SceneOccurrence[];
  pivot: Vector3;
  placement: Group;
  editable: Group;
  mesh?: OccurrenceMesh;
  sourceColors?: Color[];
}

interface SceneModel {
  root: Group;
  occurrences: Map<string, SceneOccurrence>;
  meshes: OccurrenceMesh[];
  geometries: BufferGeometry[];
  scale: number;
}

interface CameraFrame { position: Vector3; target: Vector3 }

export type CameraView = 'front' | 'side' | 'isometric';
export type SceneContextStatus = 'lost' | 'restored';
type SurfaceFinish = Pick<MeshPhysicalMaterialParameters, 'roughness' | 'metalness' | 'clearcoat' | 'clearcoatRoughness'>;
export type SurfaceFinishes = Readonly<Partial<Record<string, SurfaceFinish>>>;

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
  geometry.boundingSphere = geometry.boundingBox!.getBoundingSphere(new Sphere());
  return geometry;
}

function materialsOf(mesh?: OccurrenceMesh) {
  if (!mesh) return [];
  return Array.isArray(mesh.material) ? mesh.material : [mesh.material];
}

function makeMaterial(color: Color, finishes: SurfaceFinishes) {
  const finish = finishes[color.getHexString()];
  const parameters = { color, roughness: 0.47, ...finish };
  return finish?.clearcoat ? new MeshPhysicalMaterial(parameters) : new MeshStandardMaterial(parameters);
}

function makeModel(): SceneModel {
  return { root: new Group(), occurrences: new Map(), meshes: [], geometries: [], scale: 1 };
}

function disposeModel(model: SceneModel) {
  model.root.clear();
  for (const mesh of model.meshes) {
    for (const material of materialsOf(mesh)) material.dispose();
  }
  model.geometries.forEach((geometry) => geometry.dispose());
  model.meshes.length = 0;
  model.geometries.length = 0;
  model.occurrences.clear();
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
  } catch {
    return null;
  } finally {
    studio.dispose();
    pmrem.dispose();
  }
}

export class CadScene {
  private readonly world = new Scene();
  private readonly camera = new PerspectiveCamera(34, 1, 0.01, 1000);
  private readonly renderer: WebGLRenderer;
  private studioMap: WebGLRenderTarget | null = null;
  private readonly observer: ResizeObserver;
  private model = makeModel();
  private readonly target = new Vector3();
  private readonly raycaster = new Raycaster();
  private readonly ndc = new Vector2();
  private readonly picks: Array<{ x: number; y: number; id: string | null }> = [];
  private readonly keyLight = new DirectionalLight(0xfffaf4, 2.1);
  private readonly ambientLight = new HemisphereLight(0xf2f3ff, 0x918a9f);
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
  private readonly undoStack: Array<() => void> = [];
  private selection: string | null = null;
  private hovered: string[] = [];
  private frame: number | null = null;
  private modelMatricesDirty = false;
  private fitOnResize = false;
  private contextLost = false;
  private width = 0;
  private height = 0;
  private pixelRatio = 0;

  constructor(
    private readonly canvas: HTMLCanvasElement,
    private readonly onContextChange?: (status: SceneContextStatus) => void,
  ) {
    this.renderer = new WebGLRenderer({ canvas, antialias: true, alpha: true });
    this.renderer.toneMapping = NeutralToneMapping;
    this.renderer.toneMappingExposure = 0.96;
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = PCFShadowMap;
    this.world.environmentIntensity = 0.72;
    this.rebuildStudioMap();

    const key = this.keyLight;
    key.position.set(-3.5, 10.5, 4.1);
    key.castShadow = true;
    key.shadow.autoUpdate = false;
    key.shadow.needsUpdate = true;
    key.shadow.mapSize.set(2048, 2048);
    Object.assign(key.shadow.camera, { left: -7, right: 7, top: 7, bottom: -7, far: 28 });
    key.shadow.camera.updateProjectionMatrix();
    key.shadow.normalBias = 0.008;
    key.shadow.bias = -0.00015;
    key.shadow.radius = 9;
    const fill = new DirectionalLight(0xe2eaff, 0.68);
    fill.position.set(6, 3, 2);
    const rim = new DirectionalLight(0xe0cbff, 1.35);
    rim.position.set(-3, 3.5, -6);

    this.ground.rotation.x = -Math.PI / 2;
    this.ground.receiveShadow = true;
    this.floorGlow.rotation.x = -Math.PI / 2;
    this.floorGlow.renderOrder = 1;
    this.ground.renderOrder = 2;
    this.world.add(this.ambientLight, key, fill, rim, this.floorGlow, this.ground, this.model.root);

    this.camera.position.copy(cameraDirections.isometric).multiplyScalar(12);
    this.aimCamera();
    canvas.addEventListener('webglcontextlost', this.handleContextLost);
    canvas.addEventListener('webglcontextrestored', this.handleContextRestored);
    this.observer = new ResizeObserver(() => this.resize());
    this.observer.observe(canvas);
    this.resize();
  }

  load(assembly: CadAssembly, finishes: SurfaceFinishes = {}) {
    const draft = makeModel();
    let floorHeight: number;
    let framing: CameraFrame;
    try {
      draft.root.rotation.x = -Math.PI / 2;
      for (const mesh of assembly.meshes) draft.geometries.push(makeGeometry(mesh));
      for (const root of assembly.roots) {
        draft.root.add(this.buildOccurrence(root, null, draft, assembly.meshes, finishes).placement);
      }

      const bounds = new Box3().setFromObject(draft.root);
      const size = bounds.getSize(new Vector3()).length();
      if (!Number.isFinite(size) || size <= 0) throw new Error('The STEP assembly has no usable bounds.');
      const center = bounds.getCenter(new Vector3());
      draft.scale = 5.6 / Math.max(size, 1e-9);
      draft.root.scale.setScalar(draft.scale);
      draft.root.position.copy(center).multiplyScalar(-draft.scale);
      floorHeight = (bounds.min.y - center.y) * draft.scale - 0.008;
      const fitted = this.frameModel(draft, cameraDirections.isometric);
      if (!fitted) throw new Error('The STEP assembly could not be framed.');
      framing = fitted;
    } catch (error) {
      disposeModel(draft);
      throw error;
    }

    const previous = this.model;
    this.world.remove(previous.root);
    this.world.add(draft.root);
    this.model = draft;
    this.selection = null;
    this.hovered = [];
    this.undoStack.length = 0;
    this.ground.position.y = floorHeight;
    this.floorGlow.position.y = floorHeight - 0.008;
    this.applyFraming(framing);
    this.geometryChanged();
    disposeModel(previous);
  }

  private buildOccurrence(node: CadOccurrence, parent: SceneOccurrence | null, model: SceneModel, meshes: CadMesh[], finishes: SurfaceFinishes) {
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
      id: node.id, parent, children: [], pivot: new Vector3(), placement, editable,
    };
    model.occurrences.set(node.id, occurrence);
    parent?.children.push(occurrence);

    const bounds = new Box3();
    if (node.mesh !== undefined) {
      const geometry = model.geometries[node.mesh];
      if (!geometry) throw new Error('A STEP part references missing geometry.');
      const fallback = meshes[node.mesh].color;
      const palette = node.colors?.length === geometry.groups.length ? node.colors : [node.color ?? fallback];
      const sourceColors = palette.map((entry) => {
        const values = entry ?? fallback;
        return values ? new Color().setRGB(...values, SRGBColorSpace) : new Color('#a3adbc');
      });
      const materials = sourceColors.map((color) => makeMaterial(color, finishes));
      const mesh = new Mesh(geometry, materials.length > 1 ? materials : materials[0]);
      mesh.userData.occurrenceId = node.id;
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      contents.add(mesh);
      model.meshes.push(mesh);
      occurrence.mesh = mesh;
      occurrence.sourceColors = sourceColors;
      if (geometry.boundingBox) bounds.copy(geometry.boundingBox);
    }

    for (const child of node.children) {
      const built = this.buildOccurrence(child, occurrence, model, meshes, finishes);
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
    const pixelRatio = Math.min(window.devicePixelRatio, 2);
    if (!width || !height || (width === this.width && height === this.height && pixelRatio === this.pixelRatio)) return;
    this.width = width;
    this.height = height;
    this.pixelRatio = pixelRatio;
    this.renderer.setDrawingBufferSize(width, height, pixelRatio);
    const aspect = width / height;
    const aspectChanged = this.camera.aspect !== aspect;
    this.camera.aspect = aspect;
    this.camera.updateProjectionMatrix();
    if (aspectChanged && this.fitOnResize && this.model.meshes.length) {
      this.fit();
      return;
    }
    this.requestRender(true);
  }

  fit(view?: CameraView) {
    const direction = view
      ? cameraDirections[view]
      : this.camera.position.clone().sub(this.target).normalize();
    const framing = this.frameModel(this.model, direction);
    if (!framing) return;
    this.applyFraming(framing);
    this.requestRender(true);
  }

  private frameModel(model: SceneModel, direction: Vector3): CameraFrame | null {
    const bounds = new Box3().setFromObject(model.root);
    if (bounds.isEmpty()) return null;
    const target = bounds.getCenter(new Vector3());
    const right = new Vector3(0, 1, 0).cross(direction).normalize();
    const up = direction.clone().cross(right);
    const halfVerticalFov = Math.tan(MathUtils.degToRad(this.camera.fov / 2));
    const reachX = halfVerticalFov * this.camera.aspect * (1 - 2 * CALIBRATION_INSET_X);
    const reachY = halfVerticalFov * (1 - 2 * CALIBRATION_INSET_Y);
    let distance = 1;
    const corner = new Vector3();
    for (const mesh of model.meshes) {
      const box = mesh.geometry.boundingBox;
      if (!box || box.isEmpty()) continue;
      for (const x of [box.min.x, box.max.x]) {
        for (const y of [box.min.y, box.max.y]) {
          for (const z of [box.min.z, box.max.z]) {
            corner.set(x, y, z).applyMatrix4(mesh.matrixWorld).sub(target);
            const clearance = Math.max(
              Math.abs(corner.dot(right)) / reachX,
              Math.abs(corner.dot(up)) / reachY,
            );
            distance = Math.max(distance, corner.dot(direction) + clearance);
          }
        }
      }
    }
    return { position: target.clone().addScaledVector(direction, distance), target };
  }

  private applyFraming({ position, target }: CameraFrame) {
    this.target.copy(target);
    this.camera.position.copy(position);
    this.fitOnResize = true;
    this.aimCamera();
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

  pan(dx: number, dy: number) {
    if (!Number.isFinite(dx) || !Number.isFinite(dy) || (dx === 0 && dy === 0)) return;
    const distance = this.camera.position.distanceTo(this.target);
    const size = this.camera.getViewSize(distance, new Vector2());
    // Move the camera opposite the drag in screen coordinates, where y points down.
    const offset = new Vector3(-dx * size.x, dy * size.y, 0).applyQuaternion(this.camera.quaternion);
    this.camera.position.add(offset);
    this.target.add(offset);
    this.fitOnResize = false;
    this.aimCamera();
    this.requestRender(true);
  }

  zoom(factor: number) {
    if (!(factor > 0) || !Number.isFinite(factor)) return;
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
    if (this.modelMatricesDirty) {
      this.model.root.updateMatrixWorld(true);
      this.modelMatricesDirty = false;
    }
    this.castRay(x, y);
    const hit = this.raycaster.intersectObjects(this.model.meshes, false)[0];
    const id = hit ? hit.object.userData.occurrenceId as string : null;
    this.picks.unshift({ x, y, id });
    if (this.picks.length > 2) this.picks.pop();
    return id;
  }

  isWithin(id: string, group: string) {
    let occurrence = this.model.occurrences.get(id);
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
    const previous = this.selection;
    this.selection = id;
    if (previous !== null) this.updateHighlight(previous, true);
    if (id !== null) this.updateHighlight(id, true);
    this.requestRender();
  }

  hover(ids: string[]) {
    if (ids.length === this.hovered.length && ids.every((id, index) => id === this.hovered[index])) return;
    const changed = new Set([...this.hovered, ...ids]);
    this.hovered = ids;
    for (const id of changed) this.updateHighlight(id);
    this.requestRender();
  }

  private updateHighlight(id: string, includeChildren = false) {
    const occurrence = this.model.occurrences.get(id);
    if (!occurrence) return;
    const selected = this.selection !== null && this.isWithin(id, this.selection);
    const tint = selected ? 0x554098 : this.hovered.includes(id) ? 0x19767f : 0;
    for (const material of materialsOf(occurrence.mesh)) {
      material.emissive.setHex(tint);
      material.emissiveIntensity = 0.38;
    }
    if (includeChildren) {
      for (const child of occurrence.children) this.updateHighlight(child.id, true);
    }
  }

  getPose(id: string): ObjectPose | null {
    const occurrence = this.model.occurrences.get(id);
    if (!occurrence) return null;
    const editable = occurrence.editable;
    const position = new Vector3();
    const rotation = new Quaternion();
    const scale = new Vector3();
    editable.updateWorldMatrix(true, false);
    editable.matrixWorld.decompose(position, rotation, scale);
    return { position, rotation, scale: Math.abs(scale.x) / this.model.scale };
  }

  private toLocalPose(occurrence: SceneOccurrence, pose: ObjectPose): ObjectPose {
    occurrence.placement.updateWorldMatrix(true, false);
    const parent = occurrence.placement.matrixWorld;
    const matrix = new Matrix4().compose(
      pose.position, pose.rotation, new Vector3().setScalar(pose.scale * this.model.scale),
    );
    matrix.premultiply(parent.clone().invert());
    const position = new Vector3();
    const rotation = new Quaternion();
    const scale = new Vector3();
    matrix.decompose(position, rotation, scale);
    return { position, rotation, scale: Math.abs(scale.x) };
  }

  setPose(id: string, pose: ObjectPose) {
    const occurrence = this.model.occurrences.get(id);
    if (!occurrence) return;
    applyPose(occurrence.editable, this.toLocalPose(occurrence, pose));
    this.geometryChanged();
  }

  commitGrip(id: string, startingPose: ObjectPose) {
    const occurrence = this.model.occurrences.get(id);
    if (!occurrence) return;
    const before = this.toLocalPose(occurrence, startingPose);
    if (!samePose(before, localPose(occurrence.editable))) {
      this.pushUndo(() => applyPose(occurrence.editable, before));
    }
  }

  describe(id: string) {
    const occurrence = this.model.occurrences.get(id);
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
    for (const occurrence of this.model.occurrences.values()) {
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
    for (const occurrence of this.model.occurrences.values()) {
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
    this.geometryChanged();
  }

  canUndo() {
    return this.undoStack.length > 0;
  }

  undo() {
    const restore = this.undoStack.pop();
    if (!restore) return;
    restore();
    this.geometryChanged();
  }

  private pushUndo(restore: () => void) {
    this.undoStack.push(restore);
    if (this.undoStack.length > 100) this.undoStack.shift();
  }

  private geometryChanged() {
    this.modelMatricesDirty = true;
    this.keyLight.shadow.needsUpdate = true;
    this.requestRender(true);
  }

  private requestRender(picksChanged = false) {
    if (picksChanged) this.picks.length = 0;
    if (this.contextLost || this.frame !== null) return;
    this.frame = requestAnimationFrame(() => {
      this.frame = null;
      this.renderer.render(this.world, this.camera);
      this.modelMatricesDirty = false;
    });
  }

  private rebuildStudioMap() {
    this.studioMap?.dispose();
    this.studioMap = makeStudioMap(this.renderer);
    this.world.environment = this.studioMap?.texture ?? null;
    this.ambientLight.intensity = this.studioMap ? 0.55 : 1.15;
  }

  private handleContextLost = (event: Event) => {
    event.preventDefault();
    this.contextLost = true;
    if (this.frame !== null) cancelAnimationFrame(this.frame);
    this.frame = null;
    this.onContextChange?.('lost');
  };

  private handleContextRestored = () => {
    this.contextLost = false;
    this.rebuildStudioMap();
    const shadow = this.keyLight.shadow;
    shadow.dispose();
    shadow.map = null;
    shadow.mapPass = null;
    this.renderer.setViewport(0, 0, this.width, this.height);
    this.geometryChanged();
    this.onContextChange?.('restored');
  };

  dispose() {
    this.observer.disconnect();
    this.canvas.removeEventListener('webglcontextlost', this.handleContextLost);
    this.canvas.removeEventListener('webglcontextrestored', this.handleContextRestored);
    if (this.frame !== null) cancelAnimationFrame(this.frame);
    this.undoStack.length = 0;
    disposeModel(this.model);
    this.floorGlow.geometry.dispose();
    this.floorGlow.material.dispose();
    this.ground.geometry.dispose();
    this.ground.material.dispose();
    this.keyLight.shadow.dispose();
    this.studioMap?.dispose();
    this.renderer.dispose();
  }
}
