import { Color, Mesh, PlaneGeometry, ShaderMaterial } from 'three';

export function makeHeadFloorGrid() {
  const grid = new Mesh(
    new PlaneGeometry(30, 30),
    new ShaderMaterial({
      transparent: true,
      depthWrite: false,
      uniforms: {
        tileColor: { value: new Color(0x352846) },
        seamColor: { value: new Color(0x594369) },
      },
      vertexShader: `
        varying vec2 surfacePosition;
        void main() {
          surfacePosition = uv * 2.0 - 1.0;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: `
        varying vec2 surfacePosition;
        uniform vec3 tileColor, seamColor;
        void main() {
          vec2 tiles = surfacePosition * 7.5 + 0.5;
          vec2 distanceToSeam = min(fract(tiles), 1.0 - fract(tiles));
          vec2 sampleWidth = max(fwidth(tiles), vec2(0.001));
          vec2 coverage = 1.0 - smoothstep(vec2(0.006), vec2(0.006) + sampleWidth, distanceToSeam);
          float seam = max(coverage.x, coverage.y) * (1.0 - smoothstep(0.06, 0.24, max(sampleWidth.x, sampleWidth.y)));
          float fade = 1.0 - smoothstep(0.25, 0.98, length(surfacePosition));
          gl_FragColor = vec4(mix(tileColor, seamColor, seam * 0.65), fade * 0.88);
          #include <tonemapping_fragment>
          #include <colorspace_fragment>
        }
      `,
    }),
  );
  grid.rotation.x = -Math.PI / 2;
  return grid;
}
