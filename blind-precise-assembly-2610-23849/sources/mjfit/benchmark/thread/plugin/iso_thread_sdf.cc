// Analytic SDF of the benchmark's ISO bolt and nut, as a MuJoCo SDF plugin.
//
// The distance is the ground truth of benchmark/thread/geometry.py
// (bolt_sdf / nut_sdf): the ISO 68-1 tooth, flanks moved back c/2 along their
// normals and crest / tip lowered c/2, swept along the helix, measured in the
// meridian plane. No octree: this isolates MuJoCo's SDF contact search from
// any discretization of the shape.
//
// Plugin "iso.sdf.thread", attributes (SI units):
//   part      0 = bolt (threads on 0 <= z <= length), 1 = nut (0 <= z <= length)
//   diameter  major diameter D
//   pitch     P
//   clearance c
//   length    bolt thread length, or nut height
//   outer     nut outer radius (ignored for the bolt)
// Any other thread (another mjfit thread form, several starts, left-handed,
// or chamfered) gives its tooth instead:
//   nvert     number of tooth vertices (0 = the ISO tooth above)
//   base      radius the tooth's dr is measured from (bolt core / nut root)
//   shell     nut bore radius (ignored for the bolt)
//   lead      advance per turn (0 = pitch: single start)
//   hand      +1 right-handed, -1 left-handed
//   chamfer_r, chamfer_k  chamfer cones at both ends (0 = none): the bolt
//             keeps r <= chamfer_r + chamfer_k * (distance from the end);
//             the nut keeps r >= chamfer_r - chamfer_k * (distance from the face)
//   v<i>r, v<i>u  vertex i: (dr, u), u in the bolt's phase for both parts,
//             at most kMaxVert of them (geometry.py's bolt_tooth / nut_tooth)
//
// Built on first use by __init__.py (Linux, g++ with C++17), against the
// installed mujoco wheel: g++ -std=c++17 -O2 -fPIC -shared -I<mujoco>/include
//   iso_thread_sdf.cc <mujoco>/libmujoco.so.X -o libiso_thread_sdf.so

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <mujoco/mujoco.h>

namespace {

constexpr int kMaxVert = 32;
constexpr int kNfixed = 13;  // part .. outer, nvert, base, shell, lead, hand, chamfer_r, _k
constexpr int kNattr = kNfixed + 2 * kMaxVert;
constexpr mjtNum kPi = 3.14159265358979323846;

// Attribute names: the fixed ones, then v0r, v0u, v1r, ... Built on first
// use: the plugin registers from a load-time constructor, which can run before
// namespace-scope initializers.
const char* const* AttributeNames() {
  static std::vector<std::string> names;
  static std::vector<const char*> ptrs;
  if (ptrs.empty()) {
    names = {"part", "diameter", "pitch", "clearance", "length", "outer", "nvert", "base",
             "shell", "lead", "hand", "chamfer_r", "chamfer_k"};
    for (int i = 0; i < kMaxVert; i++) {
      names.push_back("v" + std::to_string(i) + "r");
      names.push_back("v" + std::to_string(i) + "u");
    }
    for (const auto& n : names) ptrs.push_back(n.c_str());
  }
  return ptrs.data();
}

mjtNum Default(int i) {
  static const mjtNum fixed[kNfixed] = {0, 0.010, 0.0015, 1e-4, 0.016, 0.008, 0, 0, 0, 0, 1, 0,
                                        0};
  return i < kNfixed ? fixed[i] : 0;
}

struct Thread {
  mjtNum a[kNattr];

  // Signed distance from q = (dr, u) to a convex polygon (either winding).
  static mjtNum PolygonSD(const mjtNum q[2], const mjtNum (*poly)[2], int n) {
    mjtNum area = 0;
    for (int i = 0; i < n; i++) {
      const mjtNum* p0 = poly[i];
      const mjtNum* p1 = poly[(i + 1) % n];
      area += p0[0] * p1[1] - p1[0] * p0[1];
    }
    const mjtNum wind = area > 0 ? 1 : -1;  // outward normal is wind * (ey, -ex)
    mjtNum inside = -1e30, seg = 1e30;
    for (int i = 0; i < n; i++) {
      const mjtNum* p0 = poly[i];
      const mjtNum* p1 = poly[(i + 1) % n];
      mjtNum ex = p1[0] - p0[0], ey = p1[1] - p0[1];
      mjtNum len2 = ex * ex + ey * ey, len = std::sqrt(len2);
      mjtNum rx = q[0] - p0[0], ry = q[1] - p0[1];
      inside = std::max(inside, wind * (rx * ey - ry * ex) / len);
      mjtNum t = std::clamp((rx * ex + ry * ey) / len2, (mjtNum)0, (mjtNum)1);
      mjtNum dx = rx - t * ex, dy = ry - t * ey;
      seg = std::min(seg, std::sqrt(dx * dx + dy * dy));
    }
    return inside <= 0 ? inside : seg;
  }

  // Signed distance to any simple polygon: nearest edge, negative inside
  // (crossing number). Used on the polygon path, where teeth may be hooks.
  static mjtNum GeneralSD(const mjtNum q[2], const mjtNum (*poly)[2], int n) {
    mjtNum seg = 1e30;
    bool inside = false;
    for (int i = 0; i < n; i++) {
      const mjtNum* p0 = poly[i];
      const mjtNum* p1 = poly[(i + 1) % n];
      mjtNum ex = p1[0] - p0[0], ey = p1[1] - p0[1];
      mjtNum rx = q[0] - p0[0], ry = q[1] - p0[1];
      mjtNum t = std::clamp((rx * ex + ry * ey) / (ex * ex + ey * ey), (mjtNum)0, (mjtNum)1);
      mjtNum dx = rx - t * ex, dy = ry - t * ey;
      seg = std::min(seg, std::sqrt(dx * dx + dy * dy));
      if ((p0[1] > q[1]) != (p1[1] > q[1]) &&
          q[0] < p0[0] + (q[1] - p0[1]) * ex / ey) {
        inside = !inside;
      }
    }
    return inside ? -seg : seg;
  }

  // Tooth repeated every pitch along u.
  static mjtNum Periodic(mjtNum dr, mjtNum u, const mjtNum (*poly)[2], int n, mjtNum P,
                         bool general = false) {
    u = std::fmod(u, P);
    if (u < 0) u += P;
    mjtNum best = 1e30;
    for (int k = -1; k <= 1; k++) {
      mjtNum q[2] = {dr, u + k * P};
      best = std::min(best, general ? GeneralSD(q, poly, n) : PolygonSD(q, poly, n));
    }
    return best;
  }

  mjtNum Distance(const mjtNum p[3]) const { return Static(p, a); }

  static mjtNum Static(const mjtNum p[3], const mjtNum* a) {
    if (a[6] > 0.5) return Polygon(p, a);
    const bool nut = a[0] > 0.5;
    const mjtNum D = a[1], P = a[2], c = a[3], L = a[4], Ro = a[5];
    const mjtNum h = 5 * P * std::sqrt(3.0) / 16, rmaj = D / 2, rmin = rmaj - h;
    const mjtNum s = c / std::sqrt(3.0);
    const mjtNum r = std::sqrt(p[0] * p[0] + p[1] * p[1]);
    const mjtNum phi = std::atan2(p[1], p[0]);
    const mjtNum slab = std::max(-p[2], p[2] - L);
    if (!nut) {
      const mjtNum b = -2e-4;  // base extended into the core
      const mjtNum poly[6][2] = {{b, P / 4 + s},
                                 {b, P - s},
                                 {0, P - s},
                                 {h - c / 2, 11 * P / 16 - s / 2},
                                 {h - c / 2, 9 * P / 16 + s / 2},
                                 {0, P / 4 + s}};
      mjtNum u = p[2] - P * phi / (2 * kPi);
      mjtNum sd = std::min(r - rmin, Periodic(r - rmin, u, poly, 6, P));
      return std::max(sd, slab);
    }
    const mjtNum b = c / 2 + 2e-4;  // base extended into the shell
    const mjtNum poly[6][2] = {{b, P / 8 + s},
                               {b, P - s},
                               {0, P - s},
                               {-(h - c / 2), 11 * P / 16 - s / 2},
                               {-(h - c / 2), 7 * P / 16 + s / 2},
                               {0, P / 8 + s}};
    mjtNum u = p[2] - P * phi / (2 * kPi) + 7 * P / 16;
    mjtNum inner = std::min(rmaj + c / 2 - r, Periodic(r - rmaj, u, poly, 6, P));
    return std::max(std::max(inner, r - Ro), slab);
  }

  // Any thread form: the tooth from the attributes, in the bolt's phase.
  static mjtNum Polygon(const mjtNum p[3], const mjtNum* a) {
    const bool nut = a[0] > 0.5;
    const mjtNum P = a[2], L = a[4], Ro = a[5], base = a[7], shell = a[8];
    const int n = std::min(static_cast<int>(a[6] + 0.5), kMaxVert);
    mjtNum poly[kMaxVert][2];
    for (int i = 0; i < n; i++) {
      poly[i][0] = a[kNfixed + 2 * i];
      poly[i][1] = a[kNfixed + 2 * i + 1];
    }
    const mjtNum r = std::sqrt(p[0] * p[0] + p[1] * p[1]);
    const mjtNum rise = (a[10] < 0 ? -1 : 1) * (a[9] > 0 ? a[9] : P);
    const mjtNum u = p[2] - rise * std::atan2(p[1], p[0]) / (2 * kPi);
    const mjtNum slab = std::max(-p[2], p[2] - L);
    const mjtNum tooth = Periodic(r - base, u, poly, n, P, true);
    const mjtNum cr = a[11], ck = a[12];
    const mjtNum end = std::min(p[2], L - p[2]);   // distance from the nearer end
    if (!nut) {
      mjtNum sd = std::max(std::min(r - base, tooth), slab);
      return ck > 0 ? std::max(sd, r - cr - ck * end) : sd;
    }
    mjtNum sd = std::max(std::max(std::min(shell - r, tooth), r - Ro), slab);
    return ck > 0 ? std::max(sd, cr - ck * end - r) : sd;
  }

  void Gradient(mjtNum g[3], const mjtNum p[3]) const {
    const mjtNum eps = 1e-8;
    for (int i = 0; i < 3; i++) {
      mjtNum q1[3] = {p[0], p[1], p[2]}, q0[3] = {p[0], p[1], p[2]};
      q1[i] += eps;
      q0[i] -= eps;
      g[i] = (Distance(q1) - Distance(q0)) / (2 * eps);
    }
  }
};

void Attributes(mjtNum out[], const char* name[], const char* value[]) {
  for (int i = 0; i < kNattr; i++) {
    out[i] = Default(i);
    for (int j = 0; j < kNattr; j++) {
      if (name[j] && std::strcmp(name[j], AttributeNames()[i]) == 0 && value[j] && value[j][0]) {
        out[i] = std::strtod(value[j], nullptr);
      }
    }
  }
}

}  // namespace

mjPLUGIN_LIB_INIT(iso_thread_sdf) {
  mjpPlugin plugin;
  mjp_defaultPlugin(&plugin);
  plugin.name = "iso.sdf.thread";
  plugin.capabilityflags |= mjPLUGIN_SDF;
  plugin.nattribute = kNattr;
  plugin.attributes = AttributeNames();
  plugin.nstate = +[](const mjModel*, int) { return 0; };
  plugin.init = +[](const mjModel* m, mjData* d, int instance) {
    auto* t = new Thread;
    for (int i = 0; i < kNattr; i++) {
      const char* v = mj_getPluginConfig(m, instance, AttributeNames()[i]);
      t->a[i] = (v && v[0]) ? std::strtod(v, nullptr) : Default(i);
    }
    d->plugin_data[instance] = reinterpret_cast<uintptr_t>(t);
    return 0;
  };
  plugin.destroy = +[](mjData* d, int instance) {
    delete reinterpret_cast<Thread*>(d->plugin_data[instance]);
    d->plugin_data[instance] = 0;
  };
  plugin.reset = +[](const mjModel*, mjtNum*, void*, int) {};
  plugin.compute = +[](const mjModel*, mjData*, int, int) {};
  plugin.sdf_distance = +[](const mjtNum point[3], const mjData* d, int instance) {
    return reinterpret_cast<const Thread*>(d->plugin_data[instance])->Distance(point);
  };
  plugin.sdf_gradient = +[](mjtNum gradient[3], const mjtNum point[3], const mjData* d,
                            int instance) {
    reinterpret_cast<const Thread*>(d->plugin_data[instance])->Gradient(gradient, point);
  };
  plugin.sdf_staticdistance = +[](const mjtNum point[3], const mjtNum* attributes) {
    return Thread::Static(point, attributes);
  };
  plugin.sdf_attribute = +[](mjtNum attribute[], const char* name[], const char* value[]) {
    Attributes(attribute, name, value);
  };
  plugin.sdf_aabb = +[](mjtNum aabb[6], const mjtNum* a) {
    // centre, then half-sizes, with a little room for marching cubes.
    const mjtNum r = (a[0] > 0.5 ? a[5] : a[1] / 2) * 1.05;
    aabb[0] = aabb[1] = 0;
    aabb[2] = a[4] / 2;
    aabb[3] = aabb[4] = r;
    aabb[5] = a[4] / 2 * 1.05;
  };
  mjp_registerPlugin(&plugin);
}
