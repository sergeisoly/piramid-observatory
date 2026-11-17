import type { Box } from './framing';

/** One affine map from the authored figure to its magnified viewport, in frame pixels. */
export class FigureViewport {
  scale = 1;
  x = 0;
  y = 0;
  readonly bounds: Box = [0, 0, 1600, 900];
  get changed() { return this.scale > 1; }
  reset() { this.scale = 1; this.x = this.y = 0; }
  setBounds(box: Box) { this.bounds.splice(0, 4, ...box); this.clamp(); }
  contains(x: number, y: number) {
    const b = this.bounds;
    return x >= b[0] && x <= b[2] && y >= b[1] && y <= b[3];
  }
  project(x: number, y: number): [number, number] { return [x * this.scale + this.x, y * this.scale + this.y]; }
  unproject(x: number, y: number): [number, number] { return [(x - this.x) / this.scale, (y - this.y) / this.scale]; }
  zoom(x: number, y: number, delta: number) {
    const next = Math.max(1, Math.min(3, this.scale * Math.exp(-delta * 0.0012)));
    const ratio = next / this.scale;
    this.x = x - (x - this.x) * ratio;
    this.y = y - (y - this.y) * ratio;
    this.scale = next;
    this.clamp();
  }
  pan(dx: number, dy: number) { this.x += dx; this.y += dy; this.clamp(); }
  private clamp() {
    if (!this.changed) { this.x = this.y = 0; return; }
    const b = this.bounds, k = 1 - this.scale;
    this.x = Math.max(k * b[2], Math.min(k * b[0], this.x));
    this.y = Math.max(k * b[3], Math.min(k * b[1], this.y));
  }
}
