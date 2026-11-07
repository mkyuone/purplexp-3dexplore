import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { OcctKernel } from "occt-wasm";
import { readStepFile } from "../src/cad.ts";
import { readStep } from "../src/step-model.ts";

const kernel = await OcctKernel.init();

test("STEP preserves nested, repeated and rotated occurrences and display colors", () => {
  const sample = readFileSync(new URL("../public/samples/suspension.step", import.meta.url));
  const assembly = readStep(kernel, Uint8Array.from(sample).buffer);
  assert.equal(assembly.roots.length, 1);
  assert.equal(assembly.roots[0].name, "Suspension rig");

  const wheels = assembly.roots[0].children.filter(({ name }) => name.endsWith("wheel"));
  assert.equal(wheels.length, 4);
  assert.deepEqual(wheels.map(({ matrix }) => matrix.slice(12, 15)), [
    [84, 66, 31], [84, -66, 31], [-83, 66, 31], [-83, -66, 31],
  ]);
  assert.equal(wheels[0].matrix[0], 1);
  assert.equal(wheels[1].matrix[0], -1);
  assert.equal(wheels[1].matrix[5], -1);
  assert.equal(wheels[1].matrix[10], 1);

  const tyres = wheels.map(({ children }) => {
    assert.deepEqual(children.map(({ name }) => name), ["Tyre", "Alloy rim"]);
    return children[0];
  });
  assert.equal(new Set(tyres.map(({ id }) => id)).size, 4);
  assert.equal(new Set(tyres.map(({ mesh }) => mesh)).size, 1);
  assert.ok(tyres[0].mesh !== undefined && assembly.meshes[tyres[0].mesh].indices.length > 0);
  assert.deepEqual(tyres[0].color?.map((channel) => Math.round(channel * 255)), [38, 43, 56]);
  assert.deepEqual(assembly.roots[0].children[0].color?.map((channel) => Math.round(channel * 255)), [105, 81, 207]);
});

test("an inch STEP export returns millimeter geometry", () => {
  const box = kernel.makeBox(25.4, 50.8, 76.2);
  try {
    const input = new TextEncoder().encode(kernel.exportStep(box, { unit: "inch" }));
    const { meshes: [mesh] } = readStep(kernel, input.buffer);
    const extent = [0, 0, 0];
    mesh.positions.forEach((coordinate, index) => {
      extent[index % 3] = Math.max(extent[index % 3], coordinate);
    });
    assert.deepEqual(extent.map((coordinate) => Math.round(coordinate * 10) / 10), [25.4, 50.8, 76.2]);
  } finally {
    kernel.release(box);
  }
});

test("STEP keeps an individually styled face on its source triangles", () => {
  const shape = kernel.makeBox(12, 18, 24);
  const faces = kernel.getSubShapes(shape, "face");
  const doc = kernel.createXCAFDocument();
  try {
    const label = doc.addShape(shape, { name: "Two-tone box", color: [1, 0, 0] });
    const face = faces[0];
    const { xmin, ymin, zmin, xmax, ymax, zmax } = kernel.getBoundingBox(face);
    doc.addSubShape(label, face, { color: [0, 0, 1] });

    const data = new TextEncoder().encode(doc.exportSTEP());
    const { roots: [part], meshes: [mesh] } = readStep(kernel, data.buffer);
    const palette = part.colors?.map((color) => color?.map((value) => Math.round(value * 255)));
    const blue = palette?.findIndex((color) => color?.join(",") === "0,0,255") ?? -1;
    assert.ok(blue >= 0);
    assert.deepEqual(palette?.[1 - blue], [255, 0, 0]);
    assert.equal(mesh.groups?.length, 2);
    const { start, count } = mesh.groups![blue];
    assert.equal(count, 6);
    const vertices = Array.from(mesh.indices.subarray(start, start + count), (index) =>
      Array.from(mesh.positions.subarray(index * 3, index * 3 + 3)));
    assert.deepEqual([0, 1, 2].map((axis) => Math.min(...vertices.map((vertex) => vertex[axis]))), [xmin, ymin, zmin]);
    assert.deepEqual([0, 1, 2].map((axis) => Math.max(...vertices.map((vertex) => vertex[axis]))), [xmax, ymax, zmax]);
  } finally {
    doc.close();
    for (const face of faces) kernel.release(face);
    kernel.release(shape);
  }
});

test("invalid STEP input is reported and a subsequent import still works", () => {
  assert.throws(
    () => readStep(kernel, new TextEncoder().encode("not a model").buffer),
    /not a STEP model/,
  );
  const box = kernel.makeBox(2, 4, 6);
  try {
    const input = new TextEncoder().encode(kernel.exportStep(box));
    assert.ok(readStep(kernel, input.buffer).meshes[0].indices.length > 0);
  } finally {
    kernel.release(box);
  }
});

test("a superseding import aborts its pending local file read", async () => {
  const original = Object.getOwnPropertyDescriptor(globalThis, "FileReader");
  let started = false;
  let aborted = false;
  const PendingReader = class {
    readAsArrayBuffer() { started = true; }
    abort() { aborted = true; }
  };
  Object.defineProperty(globalThis, "FileReader", { configurable: true, value: PendingReader });
  try {
    const controller = new AbortController();
    const pending = readStepFile(new File(["pending read"], "pending.step"), controller.signal);
    assert.equal(started, true);
    controller.abort();
    await assert.rejects(pending, { name: "AbortError" });
    assert.equal(aborted, true);
  } finally {
    if (original) Object.defineProperty(globalThis, "FileReader", original);
    else Reflect.deleteProperty(globalThis, "FileReader");
  }
});
