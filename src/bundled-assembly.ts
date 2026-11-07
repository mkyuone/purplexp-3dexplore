import { decodeAssembly } from "./assembly-binary.ts";
import bundledAssemblyUrl from "./models/ultraviolet-full-car.cadmesh?url";
import type { CadAssembly } from "./cad.ts";

export async function loadBundledAssembly(signal?: AbortSignal): Promise<CadAssembly> {
  const response = await fetch(bundledAssemblyUrl, { signal });
  if (!response.ok) throw new Error("The starting model could not be opened.");
  const bytes = await response.arrayBuffer();
  signal?.throwIfAborted();
  const { assembly } = await decodeAssembly(bytes);
  signal?.throwIfAborted();
  return assembly;
}
