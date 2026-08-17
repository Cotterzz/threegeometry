/**
 * CubeSphereGeometry
 * @author cotterzz / https://github.com/cotterzz
 * 
 * Based on the following work:
 * randomdudewhocodes' Shadertoy example: https://shadertoy.com/view/NfcSRH
 * And: https://mathproofs.blogspot.com/2005/07/mapping-cube-to-sphere.html
 * And SebastianLague's video: https://youtube.com/watch?v=sLqXFF8mlEU
 *
 * A cube that morphs into a sphere using an (almost) equal-area cube->sphere
 * mapping, so the grid stays nicely uniform on the sphere instead of bunching
 * up at the corners like the usual normalize(cubePoint) trick.
 *
 *   new CubeSphereGeometry(size, segments, morph, mapping, preserveArea)
 *
 *   size          overall width.  At morph=0 it is a cube of edge `size`,
 *                 at morph=1 a sphere of diameter `size` (or larger if
 *                 preserveArea is on).                        (default 1)
 *   segments      grid subdivisions per cube-face edge.       (default 16)
 *   morph         0 = cube, 1 = sphere, linear blend.        (default 1)
 *   mapping       'equalArea' (default) or 'normalize'.
 *   preserveArea  if true, the shape is uniformly scaled at each morph
 *                 value so total surface area equals the cube's 6·size².
 *                                                              (default false)
 *
 * Cheap live updates (no reallocation):
 *   geometry.setMorph(t)
 *   geometry.setSize(s)
 *   geometry.setPreserveArea(bool)
 *   geometry.set({ morph, size, preserveArea })   // batched, single _apply
 */

import { BufferGeometry, BufferAttribute } from 'three';

const PI = Math.PI;
const EPS = 1e-4;
const _m = [0, 0];

// mapping

export function cubeFaceMap(x, y, out = [0, 0]) {
  if (x * x + y * y < 1e-14) { out[0] = 0; out[1] = 0; return out; }
  const px = Math.abs(x), py = Math.abs(y);
  const flip = py > px;
  const X = flip ? py : px, Y = flip ? px : py;
  const theta = (PI / 12) * (Y / X);
  const t  = Math.tan(theta), t2 = t * t;
  const k  = t * (2 + Math.sqrt(2 * t2 + 2)) / (1 - t2), k2 = k * k;
  const B  = 1 + k2;
  const A  = 2 + k2 + Math.sqrt(2 + k2);
  const X2 = X * X;
  const u  = (X * Math.sqrt(2 * A - X2 * B)) / (A - X2 * B);
  const v  = k * u;
  out[0] = (x < 0 ? -1 : 1) * (flip ? v : u);
  out[1] = (y < 0 ? -1 : 1) * (flip ? u : v);
  return out;
}

export function cubeToSphere(x, y, z, out = [0, 0, 0]) {
  const ax = Math.abs(x), ay = Math.abs(y), az = Math.abs(z);
  let px, py, pz;
  if (ax >= ay && ax >= az) {
    cubeFaceMap(y / ax, z / ax, _m); px = Math.sign(x); py = _m[0]; pz = _m[1];
  } else if (ay >= ax && ay >= az) {
    cubeFaceMap(x / ay, z / ay, _m); px = _m[0]; py = Math.sign(y); pz = _m[1];
  } else {
    cubeFaceMap(x / az, y / az, _m); px = _m[0]; py = _m[1]; pz = Math.sign(z);
  }
  const inv = 1 / Math.hypot(px, py, pz);
  out[0] = px * inv; out[1] = py * inv; out[2] = pz * inv;
  return out;
}

// face frames
// point = w + a·ua + b·vb,  cross(ua, vb) = outward normal
const FACES = [
  { w: [ 1, 0, 0], ua: [0, 1, 0], vb: [ 0, 0,  1] },   // +X
  { w: [-1, 0, 0], ua: [0, 1, 0], vb: [ 0, 0, -1] },   // −X
  { w: [ 0, 1, 0], ua: [0, 0, 1], vb: [ 1, 0,  0] },   // +Y
  { w: [ 0,-1, 0], ua: [0, 0, 1], vb: [-1, 0,  0] },   // −Y
  { w: [ 0, 0, 1], ua: [1, 0, 0], vb: [ 0, 1,  0] },   // +Z
  { w: [ 0, 0,-1], ua: [1, 0, 0], vb: [ 0,-1,  0] }    // −Z
];

function spherePoint(F, a, b, out, o, equalArea) {
  let ma = a, mb = b;
  if (equalArea) { cubeFaceMap(a, b, _m); ma = _m[0]; mb = _m[1]; }
  const x = F.w[0] + ma * F.ua[0] + mb * F.vb[0];
  const y = F.w[1] + ma * F.ua[1] + mb * F.vb[1];
  const z = F.w[2] + ma * F.ua[2] + mb * F.vb[2];
  const inv = 1 / Math.hypot(x, y, z);
  out[o] = x * inv; out[o + 1] = y * inv; out[o + 2] = z * inv;
}

// geometry

export class CubeSphereGeometry extends BufferGeometry {

  constructor(size = 1, segments = 16, morph = 1,
              mapping = 'equalArea', preserveArea = false) {
    super();
    this.type = 'CubeSphereGeometry';
    this.parameters = {
      size,
      segments: Math.max(1, Math.floor(segments)),
      morph:    Math.min(1, Math.max(0, morph)),
      mapping,
      preserveArea: !!preserveArea
    };
    this._build();
  }

  /* single-property setters (each triggers one _apply) */
  setMorph(t) { this.parameters.morph = Math.min(1, Math.max(0, t)); this._apply(); return this; }
  setSize(s)  { this.parameters.size = s;                           this._apply(); return this; }
  setPreserveArea(b) { this.parameters.preserveArea = !!b;          this._apply(); return this; }

  /* batched setter — one _apply no matter how many keys change */
  set(p) {
    if ('morph'        in p) this.parameters.morph        = Math.min(1, Math.max(0, p.morph));
    if ('size'         in p) this.parameters.size         = p.size;
    if ('preserveArea' in p) this.parameters.preserveArea = !!p.preserveArea;
    this._apply();
    return this;
  }

  // internals

  _build() {
    const seg      = this.parameters.segments;
    const equalArea = this.parameters.mapping !== 'normalize';
    const perFace  = (seg + 1) * (seg + 1);
    const count    = 6 * perFace;

    const cube   = new Float32Array(count * 3);
    const sphere = new Float32Array(count * 3);
    const tanA   = new Float32Array(count * 3);
    const tanB   = new Float32Array(count * 3);
    const faceOf = new Uint8Array(count);
    const uv     = new Float32Array(count * 2);
    const indices = count > 65535
      ? new Uint32Array(36 * seg * seg)
      : new Uint16Array(36 * seg * seg);

    const s0 = [0, 0, 0], s1 = [0, 0, 0];
    let v = 0, ii = 0;

    for (let f = 0; f < 6; f++) {
      const F = FACES[f], base = f * perFace;
      for (let i = 0; i <= seg; i++) {
        const a = (i / seg) * 2 - 1;
        for (let j = 0; j <= seg; j++, v++) {
          const b = (j / seg) * 2 - 1, o = v * 3;

          cube[o    ] = F.w[0] + a * F.ua[0] + b * F.vb[0];
          cube[o + 1] = F.w[1] + a * F.ua[1] + b * F.vb[1];
          cube[o + 2] = F.w[2] + a * F.ua[2] + b * F.vb[2];
          spherePoint(F, a, b, sphere, o, equalArea);

          let p0 = Math.max(-1, a - EPS), p1 = Math.min(1, a + EPS), inv = 1 / (p1 - p0);
          spherePoint(F, p0, b, s0, 0, equalArea);
          spherePoint(F, p1, b, s1, 0, equalArea);
          tanA[o] = (s1[0]-s0[0])*inv; tanA[o+1] = (s1[1]-s0[1])*inv; tanA[o+2] = (s1[2]-s0[2])*inv;

          p0 = Math.max(-1, b - EPS); p1 = Math.min(1, b + EPS); inv = 1 / (p1 - p0);
          spherePoint(F, a, p0, s0, 0, equalArea);
          spherePoint(F, a, p1, s1, 0, equalArea);
          tanB[o] = (s1[0]-s0[0])*inv; tanB[o+1] = (s1[1]-s0[1])*inv; tanB[o+2] = (s1[2]-s0[2])*inv;

          uv[v * 2] = i / seg; uv[v * 2 + 1] = j / seg;
          faceOf[v] = f;
        }
      }
      for (let i = 0; i < seg; i++) for (let j = 0; j < seg; j++) {
        const k = base + i * (seg + 1) + j, k1 = k + seg + 1;
        indices[ii++] = k; indices[ii++] = k1;     indices[ii++] = k1 + 1;
        indices[ii++] = k; indices[ii++] = k1 + 1; indices[ii++] = k + 1;
      }
    }

    this.setIndex(new BufferAttribute(indices, 1));
    this.setAttribute('position', new BufferAttribute(new Float32Array(count * 3), 3));
    this.setAttribute('normal',   new BufferAttribute(new Float32Array(count * 3), 3));
    this.setAttribute('uv',       new BufferAttribute(uv, 2));

    this.clearGroups();
    const perGroup = seg * seg * 6;
    for (let f = 0; f < 6; f++) this.addGroup(f * perGroup, perGroup, f);

    this.gridPatches = [];
    for (let f = 0; f < 6; f++)
      this.gridPatches.push({ offset: f * perFace, cols: seg, rows: seg });

    this._cube = cube; this._sphere = sphere;
    this._tanA = tanA; this._tanB = tanB; this._faceOf = faceOf;
    this._apply();
  }

  _apply() {
    const t = this.parameters.morph;
    const h = this.parameters.size * 0.5;

    const pos = this.attributes.position.array;
    const nrm = this.attributes.normal.array;
    const C = this._cube, S = this._sphere;
    const TA = this._tanA, TB = this._tanB, FO = this._faceOf;
    const n = FO.length;

    for (let i = 0; i < n; i++) {
      const o = i * 3, F = FACES[FO[i]];
      pos[o    ] = (C[o    ] + (S[o    ] - C[o    ]) * t) * h;
      pos[o + 1] = (C[o + 1] + (S[o + 1] - C[o + 1]) * t) * h;
      pos[o + 2] = (C[o + 2] + (S[o + 2] - C[o + 2]) * t) * h;

      const ax = F.ua[0] + (TA[o    ] - F.ua[0]) * t;
      const ay = F.ua[1] + (TA[o + 1] - F.ua[1]) * t;
      const az = F.ua[2] + (TA[o + 2] - F.ua[2]) * t;
      const bx = F.vb[0] + (TB[o    ] - F.vb[0]) * t;
      const by = F.vb[1] + (TB[o + 1] - F.vb[1]) * t;
      const bz = F.vb[2] + (TB[o + 2] - F.vb[2]) * t;

      let nx = ay * bz - az * by;
      let ny = az * bx - ax * bz;
      let nz = ax * by - ay * bx;
      const l = Math.hypot(nx, ny, nz) || 1;
      nrm[o] = nx / l; nrm[o + 1] = ny / l; nrm[o + 2] = nz / l;
    }

    // preserve surface area
    if (this.parameters.preserveArea && t > 0) {
      const target = 6 * this.parameters.size * this.parameters.size; // cube area
      const idx = this.index.array;
      let area = 0;
      for (let i = 0; i < idx.length; i += 3) {
        const a = idx[i] * 3, b = idx[i + 1] * 3, c = idx[i + 2] * 3;
        const ex = pos[b] - pos[a], ey = pos[b+1] - pos[a+1], ez = pos[b+2] - pos[a+2];
        const fx = pos[c] - pos[a], fy = pos[c+1] - pos[a+1], fz = pos[c+2] - pos[a+2];
        const cx = ey*fz - ez*fy, cy = ez*fx - ex*fz, cz = ex*fy - ey*fx;
        area += Math.sqrt(cx*cx + cy*cy + cz*cz);
      }
      area *= 0.5;
      if (area > 1e-12) {
        const s = Math.sqrt(target / area);
        for (let i = 0; i < pos.length; i++) pos[i] *= s;
      }
    }


    this.attributes.position.needsUpdate = true;
    this.attributes.normal.needsUpdate = true;
    this.computeBoundingSphere();
    this.computeBoundingBox();
  }
}

export default CubeSphereGeometry;