import assert from "node:assert/strict";
import { readFile, rename, rm, writeFile } from "node:fs/promises";
import { OcctKernel } from "occt-wasm";
import { decodeAssembly, encodeAssembly, sha256 } from "../src/assembly-binary.ts";
import { collectPartNameRows, importPartNames } from "../src/part-names.ts";
import { readStep } from "../src/step-model.ts";

const sourceUrl = new URL("../src/models/ultraviolet-full-car.step", import.meta.url);
const bundleUrl = new URL("../src/models/ultraviolet-full-car.cadmesh", import.meta.url);
const displayNamesUrl = new URL("../src/models/ultraviolet-part-display-names.csv", import.meta.url);
const source = Uint8Array.from(await readFile(sourceUrl));
const sourceHash = await sha256(source);
const command = process.argv[2];

if (command === "--check") {
  const output = Uint8Array.from(await readFile(bundleUrl));
  const decoded = await decodeAssembly(output.buffer);
  assert.deepEqual(decoded.sourceSha256, sourceHash, "The STEP changed. Regenerate the bundled assembly.");
  const displayNames = await readFile(displayNamesUrl, "utf8");
  importPartNames(displayNames, collectPartNameRows(decoded.assembly.roots, {}));
  console.log("Bundled assembly, STEP source hash, and display names match.");
} else if (command === undefined) {
  const kernel = await OcctKernel.init();
  try {
    const assembly = readStep(kernel, source.buffer);
    const output = await encodeAssembly(assembly, sourceHash);
    const decoded = await decodeAssembly(output);
    assert.deepEqual(decoded.sourceSha256, sourceHash);
    const temporaryUrl = new URL(`${bundleUrl.href}.tmp-${process.pid}`);
    try {
      await writeFile(temporaryUrl, new Uint8Array(output));
      await rename(temporaryUrl, bundleUrl);
    } finally {
      await rm(temporaryUrl, { force: true });
    }
    console.log(`Wrote ${output.byteLength.toLocaleString("en-US")} bytes for ${assembly.meshes.length} shared meshes.`);
  } finally {
    kernel[Symbol.dispose]();
  }
} else {
  throw new Error("Usage: node --experimental-strip-types scripts/bundle-starting-model.ts [--check]");
}
