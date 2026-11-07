import { decodeAssembly } from "./assembly-binary.ts";
import bundledAssemblyUrl from "./models/ultraviolet-full-car.cadmesh?url";
import type { CadAssembly } from "./cad.ts";
import type { SurfaceFinishes } from "./scene.ts";

// Match the finishes to the authored sRGB swatches of the bundled car.
export const ultravioletFinishes: SurfaceFinishes = {
  '38147a': { roughness: 0.33, clearcoat: 0.44, clearcoatRoughness: 0.26 }, // Purple bodywork
  '3f96c6': { roughness: 0.36, clearcoat: 0.34, clearcoatRoughness: 0.29 }, // Blue wings and halo
  '3c127e': { roughness: 0.2, clearcoat: 0.65, clearcoatRoughness: 0.16 }, // Helmet visor
  'a3a9af': { roughness: 0.32 }, // Helmet shell
  '101115': { roughness: 0.76 }, // Rubber
  '55565a': { roughness: 0.4, metalness: 0.25 }, // Wheel caps
};

export async function loadBundledAssembly(signal?: AbortSignal): Promise<CadAssembly> {
  const response = await fetch(bundledAssemblyUrl, { signal });
  if (!response.ok) throw new Error("The starting model could not be opened.");
  const bytes = await response.arrayBuffer();
  signal?.throwIfAborted();
  const { assembly } = await decodeAssembly(bytes);
  signal?.throwIfAborted();
  return assembly;
}
