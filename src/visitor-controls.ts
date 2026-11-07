import type { CadScene } from './scene';

type Point = { x: number; y: number };
type Pointer = Point & { moved: boolean };

export function attachVisitorControls(canvas: HTMLCanvasElement, scene: CadScene, options: {
  available: () => boolean;
  pauseHands: (active: boolean) => void;
  select: (id: string | null) => void;
  used: () => void;
}) {
  const pointers = new Map<number, Pointer>();
  let pinching = false;

  function distance() {
    const [a, b] = pointers.values();
    return Math.hypot(a.x - b.x, a.y - b.y);
  }

  function down(event: PointerEvent) {
    if (event.button !== 0 || pointers.size >= 2 || !options.available()) return;
    canvas.focus({ preventScroll: true });
    canvas.setPointerCapture(event.pointerId);
    pointers.set(event.pointerId, { x: event.clientX, y: event.clientY, moved: false });
    if (pointers.size === 2) pinching = true;
    options.pauseHands(true);
  }

  function move(event: PointerEvent) {
    const pointer = pointers.get(event.pointerId);
    if (!pointer || !options.available()) return;
    const x = event.clientX;
    const y = event.clientY;
    const dx = x - pointer.x;
    const dy = y - pointer.y;
    if (pointers.size === 2) {
      const previous = distance();
      Object.assign(pointer, { x, y, moved: true });
      const next = distance();
      if (previous >= 8 && next >= 8) scene.zoom(next / previous);
    } else {
      if (!pointer.moved && Math.hypot(dx, dy) < 5) return;
      Object.assign(pointer, { x, y, moved: true });
      scene.orbit(dx / canvas.clientWidth, dy / canvas.clientWidth);
    }
    options.used();
  }

  function end(event: PointerEvent) {
    const pointer = pointers.get(event.pointerId);
    if (!pointer) return;
    pointers.delete(event.pointerId);
    if (!pointers.size) {
      if (!pinching && !pointer.moved && event.type === 'pointerup' && options.available()) {
        const bounds = canvas.getBoundingClientRect();
        const part = scene.pick((event.clientX - bounds.left) / bounds.width, (event.clientY - bounds.top) / bounds.height);
        options.select(part);
        if (part) options.used();
      }
      pinching = false;
      options.pauseHands(false);
    }
  }

  function wheel(event: WheelEvent) {
    if (!options.available() || event.ctrlKey) return;
    event.preventDefault();
    const pixels = event.deltaY * (event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? canvas.clientHeight : 1);
    options.pauseHands(pointers.size > 0);
    scene.zoom(Math.exp(-Math.max(-150, Math.min(150, pixels)) * .002));
    options.used();
  }

  const commands: Record<string, () => void> = {
    ArrowLeft: () => scene.orbit(-.025, 0), ArrowRight: () => scene.orbit(.025, 0),
    ArrowUp: () => scene.orbit(0, -.025), ArrowDown: () => scene.orbit(0, .025),
    '+': () => scene.zoom(1.12), '=': () => scene.zoom(1.12), '-': () => scene.zoom(1 / 1.12),
    Enter: () => options.select(scene.pick(.5, .5)), Escape: () => options.select(null),
    Home: () => scene.fit('isometric'),
  };

  function key(event: KeyboardEvent) {
    if (!options.available() || event.ctrlKey || event.metaKey || event.altKey) return;
    const command = commands[event.key];
    if (!Object.hasOwn(commands, event.key)) return;
    event.preventDefault();
    options.pauseHands(pointers.size > 0);
    command();
    options.used();
  }

  const events = new AbortController();
  const signal = events.signal;
  canvas.addEventListener('pointerdown', down, { signal });
  canvas.addEventListener('pointermove', move, { signal });
  canvas.addEventListener('pointerup', end, { signal });
  canvas.addEventListener('pointercancel', end, { signal });
  canvas.addEventListener('lostpointercapture', end, { signal });
  canvas.addEventListener('wheel', wheel, { passive: false, signal });
  canvas.addEventListener('keydown', key, { signal });
  return () => events.abort();
}
