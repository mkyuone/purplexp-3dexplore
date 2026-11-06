/// <reference lib="webworker" />
/// <reference lib="esnext.disposable" />

import { OcctKernel } from "occt-wasm";
import wasmUrl from "occt-wasm/dist/occt-wasm.wasm?url";
import type { StepResponse } from "./cad.ts";
import { readStep } from "./step-model.ts";

const scope = self as unknown as DedicatedWorkerGlobalScope;

scope.onmessage = async ({ data }: MessageEvent<ArrayBuffer>) => {
  try {
    const kernel = await OcctKernel.init({ wasm: wasmUrl });
    try {
      const assembly = readStep(kernel, data);
      const transfer = assembly.meshes.flatMap(({ positions, normals, indices }) => [
        positions.buffer as ArrayBuffer,
        normals.buffer as ArrayBuffer,
        indices.buffer as ArrayBuffer,
      ]);
      scope.postMessage({ assembly } satisfies StepResponse, transfer);
    } finally {
      kernel[Symbol.dispose]();
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    scope.postMessage({ error: message } satisfies StepResponse);
  }
};
