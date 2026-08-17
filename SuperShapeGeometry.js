/**
 * SuperShapeGeometry
 * @author cotterzz / https://github.com/cotterzz
 * Gielis superformula as a 3D surface (the "spherical product" of two
 * superformula curves)
 * 
 * See: https://paulbourke.net/geometry/supershape/
 * And: https://en.wikipedia.org/wiki/Superformula
 * 
 *   r1 = superFormula(u)          u = longitude, -PI   .. PI
 *   r2 = superFormula(v)          v = latitude,  -PI/2 .. PI/2
 *
 *   x = size * r1*cos(u) * r2*cos(v)
 *   y = size * r2*sin(v)                  (Y-up for three.js)
 *   z = size * r1*sin(u) * r2*cos(v)
 *
 *   new SuperShapeGeometry(size, m, n1, n2, n3, a, b, segments)
 *
 *   size      overall scale (= radius when m=0, which gives a sphere)
 *   m         number of rotational lobes (non-integers give odd, fun shapes)
 *   n1        overall "pinch": <1 spiky/concave, 1 diamond, >1 rounded/boxy
 *   n2, n3    exponents on the cos / sin terms (asymmetry between lobes)
 *   a, b      scale of the cos / sin terms (stretch)
 *   segments  detail: longitude gets `segments`, latitude `segments/2`
 *
 * Live tweaks with no reallocation:
 *   geometry.setParams({ m: 7, n1: 0.4 })
 * (changing `segments` rebuilds automatically)
 */

import { BufferGeometry, BufferAttribute } from 'three';

const PI = Math.PI;
const TAU = PI * 2;
const EPS = 1e-4;    // finite-difference step, in radians
const INSET = 1e-3;  // keep normal sampling off the exact poles

export function superFormula(theta, m, n1, n2, n3, a, b) {
  const h = m * theta * 0.25;
  const t1 = Math.pow(Math.abs(Math.cos(h) / a), n2);
  const t2 = Math.pow(Math.abs(Math.sin(h) / b), n3);
  const s = t1 + t2;
  if (!(s > 0)) return 0;
  const r = Math.pow(s, -1 / n1);
  return Number.isFinite(r) ? Math.min(r, 1e3) : 0;
}

export class SuperShapeGeometry extends BufferGeometry {

  constructor(size = 1, m = 4, n1 = 1, n2 = 1, n3 = 1, a = 1, b = 1, segments = 64) {
    super();
    this.type = 'SuperShapeGeometry';
    this.parameters = { size, m, n1, n2, n3, a, b, segments };
    this._build();
  }

  setParams(patch) {
    const before = this.parameters.segments;
    Object.assign(this.parameters, patch);
    if (this.parameters.segments !== before) this._build();
    else this._apply();
    return this;
  }

  // internals

  _build() {
    const s = Math.max(4, Math.floor(this.parameters.segments));
    const segU = s;
    const segV = Math.max(3, Math.round(s / 2));
    this._segU = segU;
    this._segV = segV;

    const M = segV + 1;
    const count = (segU + 1) * M;

    const uv = new Float32Array(count * 2);
    for (let i = 0; i <= segU; i++) {
      for (let j = 0; j < M; j++) {
        const k = i * M + j;
        uv[k * 2] = i / segU;
        uv[k * 2 + 1] = j / segV;
      }
    }

    // winding is CCW-from-outside for v-then-u ordering
    const idx = [];
    for (let i = 0; i < segU; i++) {
      for (let j = 0; j < segV; j++) {
        const k = i * M + j, k1 = k + M;
        if (j !== segV - 1) idx.push(k, k + 1, k1 + 1);   // skip top pole sliver
        if (j !== 0)        idx.push(k, k1 + 1, k1);      // skip bottom pole sliver
      }
    }

    this.setIndex(new BufferAttribute(
      new (count > 65535 ? Uint32Array : Uint16Array)(idx), 1));
    this.setAttribute('position', new BufferAttribute(new Float32Array(count * 3), 3));
    this.setAttribute('normal',   new BufferAttribute(new Float32Array(count * 3), 3));
    this.setAttribute('uv',       new BufferAttribute(uv, 2));

    this.gridPatches = [{ offset: 0, cols: segU, rows: segV }];

    this._apply();
  }

  _apply() {
    const P = this.parameters;
    const m = P.m;
    const n1 = Math.max(1e-3, P.n1);
    const n2 = Math.max(1e-3, P.n2);
    const n3 = Math.max(1e-3, P.n3);
    const a = Math.max(1e-3, P.a);
    const b = Math.max(1e-3, P.b);
    const size = P.size;

    const segU = this._segU, segV = this._segV;
    const N = segU + 1, M = segV + 1;
    const SF = t => superFormula(t, m, n1, n2, n3, a, b);

    // r1 only depends on u and r2 only on v, so cache per column / per row.
    const cu = new Float64Array(N), su = new Float64Array(N), r1 = new Float64Array(N);
    const dux = new Float64Array(N), duz = new Float64Array(N);
    for (let i = 0; i < N; i++) {
      const u = -PI + TAU * i / segU;
      cu[i] = Math.cos(u); su[i] = Math.sin(u); r1[i] = SF(u);
      const rm = SF(u - EPS), rp = SF(u + EPS);
      dux[i] = rp * Math.cos(u + EPS) - rm * Math.cos(u - EPS);
      duz[i] = rp * Math.sin(u + EPS) - rm * Math.sin(u - EPS);
    }

    const cv = new Float64Array(M), sv = new Float64Array(M), r2 = new Float64Array(M);
    const dA = new Float64Array(M), dB = new Float64Array(M);
    for (let j = 0; j < M; j++) {
      const v = -PI / 2 + PI * j / segV;
      cv[j] = Math.cos(v); sv[j] = Math.sin(v); r2[j] = SF(v);
      const vn = Math.min(Math.max(v, -PI / 2 + INSET), PI / 2 - INSET);
      const rm = SF(vn - EPS), rp = SF(vn + EPS);
      dA[j] = rp * Math.cos(vn + EPS) - rm * Math.cos(vn - EPS);
      dB[j] = rp * Math.sin(vn + EPS) - rm * Math.sin(vn - EPS);
    }

    const pos = this.attributes.position.array;
    const nrm = this.attributes.normal.array;

    for (let i = 0; i < N; i++) {
      const tux = dux[i], tuz = duz[i];           // dP/du  = (tux, 0, tuz) * k
      for (let j = 0; j < M; j++) {
        const k = (i * M + j) * 3;
        const rr = r1[i] * r2[j];

        pos[k    ] = size * rr * cu[i] * cv[j];
        pos[k + 1] = size * r2[j] * sv[j];
        pos[k + 2] = size * rr * su[i] * cv[j];

        // dP/dv (positive scalar factors dropped — only direction matters)
        const tvx = r1[i] * cu[i] * dA[j];
        const tvy = dB[j];
        const tvz = r1[i] * su[i] * dA[j];

        // n = cross(dP/dv, dP/du)  -> outward
        let nx = tvy * tuz;
        let ny = tvz * tux - tvx * tuz;
        let nz = -tvy * tux;

        const L = Math.hypot(nx, ny, nz);
        if (L > 1e-20) { nx /= L; ny /= L; nz /= L; }
        else {
          const l = Math.hypot(pos[k], pos[k + 1], pos[k + 2]) || 1;
          nx = pos[k] / l; ny = pos[k + 1] / l; nz = pos[k + 2] / l;
        }
        nrm[k] = nx; nrm[k + 1] = ny; nrm[k + 2] = nz;
      }
    }

    this.attributes.position.needsUpdate = true;
    this.attributes.normal.needsUpdate = true;
    this.computeBoundingSphere();
    this.computeBoundingBox();
  }
}

export default SuperShapeGeometry;