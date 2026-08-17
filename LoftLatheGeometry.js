/**
 * LoftLatheGeometry
 * @author cotterzz / https://github.com/cotterzz
 * this class originally adapted from THREE.LatheBufferGeometry by:
 * @author zz85 / https://github.com/zz85
 * @author bhouston / http://clara.io
 * @author Mugen87 / https://github.com/Mugen87
 * 
 * A lathe whose profile changes as it sweeps: `profiles` is a 2D array of
 * Vector2-ish points ({x: radius, y: height}). Every profile must have the
 * same number of points. Profile 0 sits at phi = 0 and they are distributed
 * evenly (and periodically) around the full turn, so the last one blends back
 * into the first.
 *
 * smooth interpolation in BOTH directions (sweep + profile) via a Hermite
 * spline whose tang1ents blend linear -> Catmull-Rom (`smooth` 0..1)
 * independent `rings` subdivision so the profile can actually curve
 * size / radius / height multipliers, auto-normalised & centred
 * analytic grid normals (periodic across the seam — no seam fix-up needed)
 * pole slivers removed, in-place updates with no reallocation
 *
 *   new LoftLatheGeometry(profiles, {
 *     segments = 64,        // sweep subdivisions
 *     rings    = 0,         // profile subdivisions (0 = one per control point)
 *     smooth   = 1,         // 0 = linear (original look), 1 = Catmull-Rom
 *     size     = 1,         // overall scale (= world height when height = 1)
 *     radius   = 1,         // radial multiplier
 *     height   = 1,         // vertical multiplier
 *     phiStart = 0, phiLength = 2PI,
 *     normalize = true      // scale to unit height + centre on origin
 *   })
 *
 * Live updates:
 *   geometry.set({ size, radius, height, smooth, profiles, segments, rings })
 *   — it picks the cheapest update level automatically.
 */

import { BufferGeometry, BufferAttribute } from 'three';

const TAU = Math.PI * 2;

/** Cubic Hermite whose tangents lerp from "linear" (s=0) to Catmull-Rom (s=1) */
export function hermite(p0, p1, p2, p3, t, s) {
  const d  = p2 - p1;
  const m1 = d + s * (0.5 * (p2 - p0) - d);
  const m2 = d + s * (0.5 * (p3 - p1) - d);
  const t2 = t * t, t3 = t2 * t;
  return (2*t3 - 3*t2 + 1) * p1 +
         (t3 - 2*t2 + t)   * m1 +
         (-2*t3 + 3*t2)    * p2 +
         (t3 - t2)         * m2;
}

export function cloneProfiles(profiles) {
  return profiles.map(row => row.map(p => ({ x: p.x, y: p.y })));
}

/**
 * Sample the loft surface in profile space.
 * u = sweep fraction of the full turn (periodic), v = 0..1 along the profile.
 * Returns { x: radius, y: height } in raw (un-normalised) data units.
 */
export function loftSample(profiles, u, v, smooth = 1, out = { x: 0, y: 0 }) {
  const n = profiles.length, len = profiles[0].length;

  const fu = (((u % 1) + 1) % 1) * n;
  const fl = Math.floor(fu), tu = fu - fl;
  const i1 = ((fl % n) + n) % n;
  const i0 = (i1 - 1 + n) % n, i2 = (i1 + 1) % n, i3 = (i1 + 2) % n;

  const fv = Math.min(1, Math.max(0, v)) * (len - 1);
  let j1 = Math.floor(fv); if (j1 > len - 2) j1 = Math.max(0, len - 2);
  const tv = fv - j1;
  const j0 = Math.max(j1 - 1, 0), j2 = Math.min(j1 + 1, len - 1), j3 = Math.min(j1 + 2, len - 1);

  const rx = [0, 0, 0, 0], ry = [0, 0, 0, 0];
  const ii = [i0, i1, i2, i3];
  for (let q = 0; q < 4; q++) {
    const P = profiles[ii[q]];
    rx[q] = hermite(P[j0].x, P[j1].x, P[j2].x, P[j3].x, tv, smooth);
    ry[q] = hermite(P[j0].y, P[j1].y, P[j2].y, P[j3].y, tv, smooth);
  }
  out.x = hermite(rx[0], rx[1], rx[2], rx[3], tu, smooth);
  out.y = hermite(ry[0], ry[1], ry[2], ry[3], tu, smooth);
  return out;
}

/** Resample a profile set to a new rows × points grid (uses the spline) */
export function resampleProfiles(profiles, rows, pts) {
  const out = [];
  const tmp = { x: 0, y: 0 };
  for (let i = 0; i < rows; i++) {
    const row = [];
    for (let j = 0; j < pts; j++) {
      loftSample(profiles, i / rows, pts > 1 ? j / (pts - 1) : 0, 1, tmp);
      row.push({ x: Math.max(0, tmp.x), y: tmp.y });
    }
    out.push(row);
  }
  return out;
}

const LEVEL = { size:1, radius:1, height:1,
                smooth:2, normalize:2,
                segments:3, rings:3, phiStart:3, phiLength:3 };

export class LoftLatheGeometry extends BufferGeometry {

  constructor(profiles, options = {}) {
    super();
    this.type = 'LoftLatheGeometry';
    this.parameters = Object.assign({
      segments: 64, rings: 0, smooth: 1,
      size: 1, radius: 1, height: 1,
      phiStart: 0, phiLength: TAU, normalize: true
    }, options);
    this.parameters.profiles = cloneProfiles(profiles);
    this._build();
  }

  /** batched setter — performs the cheapest possible update */
  set(patch) {
    const p = this.parameters;
    let lvl = 0;
    for (const k in patch) {
      if (k === 'profiles') {
        const a = patch.profiles;
        const sameDims = p.profiles.length === a.length &&
                         p.profiles[0].length === a[0].length;
        p.profiles = cloneProfiles(a);
        lvl = Math.max(lvl, sameDims ? 2 : 3);
      } else if (p[k] !== patch[k]) {
        p[k] = patch[k];
        lvl = Math.max(lvl, LEVEL[k] ?? 3);
      }
    }
    if (lvl >= 3)      this._build();
    else if (lvl === 2) { this._sample(); this._apply(); }
    else if (lvl === 1) this._apply();
    return this;
  }

  /** world-space polyline of the cross-section at sweep fraction u (for UI) */
  sectionPoints(u, samples = 80) {
    const p = this.parameters;
    const phi = p.phiStart + u * TAU;
    const sin = Math.sin(phi), cos = Math.cos(phi);
    const sx = p.size * p.radius, sy = p.size * p.height;
    const out = new Float32Array((samples + 1) * 3);
    const tmp = { x: 0, y: 0 };
    for (let k = 0; k <= samples; k++) {
      loftSample(p.profiles, u, k / samples, p.smooth, tmp);
      const r = Math.max(0, tmp.x) * this._inv * sx;
      out[k*3    ] = r * sin;
      out[k*3 + 1] = (tmp.y - this._yc) * this._inv * sy;
      out[k*3 + 2] = r * cos;
    }
    return out;
  }

  // internals

  _build() {
    const p = this.parameters;
    const len = p.profiles[0].length;
    const seg   = this._seg   = Math.max(3, Math.floor(p.segments));
    const rings = this._rings = Math.max(1, Math.floor(p.rings > 0 ? p.rings : len - 1));
    const W = rings + 1;
    const count = (seg + 1) * W;

    this._rr  = new Float32Array(count);
    this._yy  = new Float32Array(count);
    this._sin = new Float64Array(seg + 1);
    this._cos = new Float64Array(seg + 1);

    const uv = new Float32Array(count * 2);
    for (let i = 0; i <= seg; i++)
      for (let k = 0; k < W; k++) {
        const o = (i * W + k) * 2;
        uv[o] = i / seg; uv[o + 1] = k / rings;
      }

    this.setAttribute('position', new BufferAttribute(new Float32Array(count * 3), 3));
    this.setAttribute('normal',   new BufferAttribute(new Float32Array(count * 3), 3));
    this.setAttribute('uv',       new BufferAttribute(uv, 2));
    this.gridPatches = [{ offset: 0, cols: seg, rows: rings }];

    this._sample();
    this._apply();
  }

  _sample() {
    const p = this.parameters, prof = p.profiles;
    const n = prof.length, len = prof[0].length;
    const seg = this._seg, rings = this._rings, W = rings + 1;
    const s = Math.min(1, Math.max(0, p.smooth));

    // normalisation from the raw control data
    let minY = Infinity, maxY = -Infinity;
    for (let i = 0; i < n; i++)
      for (let j = 0; j < len; j++) {
        const y = prof[i][j].y;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    const span = p.normalize ? ((maxY - minY) || 1) : 1;
    this._yc  = p.normalize ? (minY + maxY) * 0.5 : 0;
    this._inv = 1 / span;

    // pass 1 — interpolate each control profile along its own length
    const VR = new Float32Array(n * W), VY = new Float32Array(n * W);
    for (let i = 0; i < n; i++) {
      const P = prof[i];
      for (let k = 0; k < W; k++) {
        const fv = (k / rings) * (len - 1);
        let j1 = Math.floor(fv); if (j1 > len - 2) j1 = Math.max(0, len - 2);
        const tv = fv - j1;
        const j0 = Math.max(j1 - 1, 0), j2 = Math.min(j1 + 1, len - 1), j3 = Math.min(j1 + 2, len - 1);
        VR[i * W + k] = hermite(P[j0].x, P[j1].x, P[j2].x, P[j3].x, tv, s);
        VY[i * W + k] = hermite(P[j0].y, P[j1].y, P[j2].y, P[j3].y, tv, s);
      }
    }

    // pass 2 — sweep (periodic across the profile array)
    const rr = this._rr, yy = this._yy, inv = this._inv, yc = this._yc;
    for (let i = 0; i <= seg; i++) {
      const phi = p.phiStart + (i / seg) * p.phiLength;
      this._sin[i] = Math.sin(phi);
      this._cos[i] = Math.cos(phi);

      const fu = (phi / TAU) * n;
      const fl = Math.floor(fu), tu = fu - fl;
      const i1 = ((fl % n) + n) % n;
      const i0 = (i1 - 1 + n) % n, i2 = (i1 + 1) % n, i3 = (i1 + 2) % n;

      for (let k = 0; k < W; k++) {
        const r = hermite(VR[i0*W+k], VR[i1*W+k], VR[i2*W+k], VR[i3*W+k], tu, s);
        const y = hermite(VY[i0*W+k], VY[i1*W+k], VY[i2*W+k], VY[i3*W+k], tu, s);
        rr[i*W + k] = Math.max(0, r) * inv;       // clamp spline overshoot
        yy[i*W + k] = (y - yc) * inv;
      }
    }

    // index — drop the sliver triangles on rings that sit on the axis
    const onAxis = new Uint8Array(W);
    for (let k = 0; k < W; k++) {
      let pole = true;
      for (let i = 0; i <= seg; i++) if (rr[i*W + k] > 1e-7) { pole = false; break; }
      onAxis[k] = pole ? 1 : 0;
    }
    const idx = [];
    for (let i = 0; i < seg; i++)
      for (let k = 0; k < rings; k++) {
        const a = i*W + k, b = a + W, c = b + 1, d = a + 1;
        if (!onAxis[k])     idx.push(a, b, d);
        if (!onAxis[k + 1]) idx.push(b, c, d);
      }
    const count = (seg + 1) * W;
    this.setIndex(new BufferAttribute(
      new (count > 65535 ? Uint32Array : Uint16Array)(idx), 1));
  }

  _apply() {
    const p = this.parameters;
    const sx = p.size * p.radius, sy = p.size * p.height;
    const seg = this._seg, W = this._rings + 1;
    const pos = this.attributes.position.array;

    for (let i = 0; i <= seg; i++) {
      const si = this._sin[i], ci = this._cos[i];
      for (let k = 0; k < W; k++) {
        const o = i*W + k, t = o * 3;
        const r = this._rr[o] * sx;
        pos[t] = r * si; pos[t + 1] = this._yy[o] * sy; pos[t + 2] = r * ci;
      }
    }
    this._normals();

    this.attributes.position.needsUpdate = true;
    this.attributes.normal.needsUpdate = true;
    this.computeBoundingSphere();
    this.computeBoundingBox();
  }

  /** grid central differences; wraps in the sweep so the seam matches exactly */
  _normals() {
    const p = this.parameters;
    const seg = this._seg, rings = this._rings, W = rings + 1;
    const closed = Math.abs(p.phiLength - TAU) < 1e-6;
    const pos = this.attributes.position.array;
    const nrm = this.attributes.normal.array;
    const at = (i, k) => (i * W + k) * 3;

    for (let i = 0; i <= seg; i++) {
      let ia = i - 1, ib = i + 1;
      if (ia < 0)   ia = closed ? seg - 1 : 0;
      if (ib > seg) ib = closed ? 1       : seg;

      for (let k = 0; k < W; k++) {
        const ka = Math.max(k - 1, 0), kb = Math.min(k + 1, rings);
        const A = at(ia, k), B = at(ib, k), C = at(i, ka), D = at(i, kb);

        const ux = pos[B]   - pos[A],   uy = pos[B+1] - pos[A+1], uz = pos[B+2] - pos[A+2];
        const vx = pos[D]   - pos[C],   vy = pos[D+1] - pos[C+1], vz = pos[D+2] - pos[C+2];

        let nx = uy*vz - uz*vy, ny = uz*vx - ux*vz, nz = ux*vy - uy*vx;
        const L = Math.hypot(nx, ny, nz);
        const o = at(i, k);
        if (L > 1e-12) { nrm[o] = nx/L; nrm[o+1] = ny/L; nrm[o+2] = nz/L; }
        else { nrm[o] = 0; nrm[o+1] = k > rings * 0.5 ? 1 : -1; nrm[o+2] = 0; }
      }
    }
  }
}

export default LoftLatheGeometry;