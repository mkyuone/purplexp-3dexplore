export type CadColor = [number, number, number];

export interface CadMesh {
  positions: Float32Array;
  normals: Float32Array;
  indices: Uint32Array;
  color?: CadColor;
  /** Contiguous index ranges for the optional occurrence palette. */
  groups?: { start: number; count: number }[];
}

export interface CadOccurrence {
  id: string;
  name: string;
  /** A column-major local transform in millimeters, with Z up. */
  matrix: number[];
  mesh?: number;
  /** sRGB surface color, shared by default and overridable per occurrence. */
  color?: CadColor;
  /** sRGB face colors in the shared mesh's group order. */
  colors?: (CadColor | undefined)[];
  children: CadOccurrence[];
}

export interface CadAssembly {
  roots: CadOccurrence[];
  meshes: CadMesh[];
}

export type StepResponse = { assembly: CadAssembly } | { error: string };

export async function importStep(data: ArrayBuffer, signal?: AbortSignal): Promise<CadAssembly> {
  signal?.throwIfAborted();
  const worker = new Worker(new URL("./step.worker.ts", import.meta.url), { type: "module" });

  return new Promise((resolve, reject) => {
    const cleanup = () => {
      worker.terminate();
      signal?.removeEventListener("abort", abort);
    };

    const abort = () => {
      cleanup();
      reject(signal?.reason ?? new DOMException("The import was canceled.", "AbortError"));
    };

    const fail = (message: string) => {
      cleanup();
      reject(new Error(message));
    };

    signal?.addEventListener("abort", abort, { once: true });
    worker.onmessage = ({ data: result }: MessageEvent<StepResponse>) => {
      if ("error" in result) {
        fail(result.error);
        return;
      }
      cleanup();
      resolve(result.assembly);
    };
    worker.onerror = () => fail("The STEP importer could not start.");
    worker.onmessageerror = () => fail("The imported geometry could not be received.");

    try {
      worker.postMessage(data, [data]);
    } catch (error) {
      cleanup();
      reject(error);
    }
  });
}
