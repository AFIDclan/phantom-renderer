#pragma once

#include <cmath>
#include <vector>

// ----------------------------------------------------------------------------
// Vec3 — simple 3-component float vector with free-function helpers
// ----------------------------------------------------------------------------

struct Vec3 {
    float x = 0.f, y = 0.f, z = 0.f;
};

inline float dot(Vec3 a, Vec3 b)      { return a.x*b.x + a.y*b.y + a.z*b.z; }
inline float length(Vec3 v)           { return std::sqrt(dot(v, v)); }
inline Vec3  normalize(Vec3 v)        { float l = length(v); return {v.x/l, v.y/l, v.z/l}; }
inline Vec3  cross(Vec3 a, Vec3 b)    { return {a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x}; }
inline Vec3  sub(Vec3 a, Vec3 b)      { return {a.x-b.x, a.y-b.y, a.z-b.z}; }
inline Vec3  add(Vec3 a, Vec3 b)      { return {a.x+b.x, a.y+b.y, a.z+b.z}; }
inline Vec3  mul(Vec3 v, float s)     { return {v.x*s, v.y*s, v.z*s}; }

// ----------------------------------------------------------------------------
// Mat4 — column-major 4×4 float matrix.  m[col][row] matches GLSL mat4
// and the glTF node.matrix layout (16 doubles, column-major).
// ----------------------------------------------------------------------------

struct Mat4 {
    float m[4][4] = {};

    // ---- factories ----------------------------------------------------------

    static Mat4 identity() {
        Mat4 r;
        r.m[0][0] = r.m[1][1] = r.m[2][2] = r.m[3][3] = 1.f;
        return r;
    }

    // Vulkan perspective: Y-axis flipped in NDC, depth mapped to [0, 1].
    static Mat4 perspective(float fovY, float aspect, float nearZ, float farZ) {
        float t = std::tan(fovY * 0.5f);
        Mat4 r;
        r.m[0][0] =  1.f / (aspect * t);
        r.m[1][1] = -1.f / t;                          // flip Y for Vulkan NDC
        r.m[2][2] =  farZ / (nearZ - farZ);
        r.m[2][3] = -1.f;
        r.m[3][2] = -(farZ * nearZ) / (farZ - nearZ);
        return r;
    }

    // Cube-face perspective: same as perspective() but WITHOUT the Y-flip.
    // Cube map face textures are natively Y-down (matching Vulkan's framebuffer
    // layout), so the main-camera Y-flip must not be applied here — it would
    // invert V in every face and misalign shadow lookups.
    static Mat4 perspectiveCube(float fovY, float aspect, float nearZ, float farZ) {
        float t = std::tan(fovY * 0.5f);
        Mat4 r;
        r.m[0][0] =  1.f / (aspect * t);
        r.m[1][1] =  1.f / t;                          // no Y-flip for cube faces
        r.m[2][2] =  farZ / (nearZ - farZ);
        r.m[2][3] = -1.f;
        r.m[3][2] = -(farZ * nearZ) / (farZ - nearZ);
        return r;
    }

    // Orthographic projection for Vulkan depth range [0, 1], without the
    // main-camera Y-flip. Used for offscreen directional shadow maps.
    static Mat4 ortho(float left, float right,
                      float bottom, float top,
                      float nearZ, float farZ) {
        Mat4 r = identity();
        r.m[0][0] =  2.f / (right - left);
        r.m[1][1] =  2.f / (top - bottom);
        r.m[2][2] =  1.f / (nearZ - farZ);
        r.m[3][0] = -(right + left) / (right - left);
        r.m[3][1] = -(top + bottom) / (top - bottom);
        r.m[3][2] =  nearZ / (nearZ - farZ);
        return r;
    }

    static Mat4 lookAt(Vec3 eye, Vec3 center, Vec3 up) {
        Vec3 f  = normalize(sub(center, eye));
        Vec3 r2 = normalize(cross(f, up));
        Vec3 u  = cross(r2, f);
        Mat4 out;
        out.m[0][0] =  r2.x; out.m[1][0] =  r2.y; out.m[2][0] =  r2.z; out.m[3][0] = -dot(r2, eye);
        out.m[0][1] =   u.x; out.m[1][1] =   u.y; out.m[2][1] =   u.z; out.m[3][1] = -dot(u,  eye);
        out.m[0][2] =  -f.x; out.m[1][2] =  -f.y; out.m[2][2] =  -f.z; out.m[3][2] =  dot(f,  eye);
        out.m[3][3] = 1.f;
        return out;
    }

    static Mat4 translate(Vec3 t) {
        Mat4 r = identity();
        r.m[3][0] = t.x; r.m[3][1] = t.y; r.m[3][2] = t.z;
        return r;
    }

    static Mat4 rotateY(float a) {
        Mat4 r = identity();
        r.m[0][0] =  std::cos(a); r.m[2][0] = std::sin(a);
        r.m[0][2] = -std::sin(a); r.m[2][2] = std::cos(a);
        return r;
    }

    static Mat4 scale(float s) {
        Mat4 r = identity();
        r.m[0][0] = r.m[1][1] = r.m[2][2] = s;
        return r;
    }

    // ---- glTF loaders -------------------------------------------------------

    // From a 16-element column-major double array (node.matrix).
    static Mat4 fromGLTF(const std::vector<double>& v) {
        Mat4 out;
        for (int c = 0; c < 4; c++)
            for (int row = 0; row < 4; row++)
                out.m[c][row] = static_cast<float>(v[c * 4 + row]);
        return out;
    }

    // From glTF TRS components: translation, quaternion [qx,qy,qz,qw], scale.
    static Mat4 fromTRS(const std::vector<double>& t,
                        const std::vector<double>& q,
                        const std::vector<double>& s) {
        float sx = s.size() >= 3 ? (float)s[0] : 1.f,
              sy = s.size() >= 3 ? (float)s[1] : 1.f,
              sz = s.size() >= 3 ? (float)s[2] : 1.f;
        float qx = q.size() >= 4 ? (float)q[0] : 0.f,
              qy = q.size() >= 4 ? (float)q[1] : 0.f,
              qz = q.size() >= 4 ? (float)q[2] : 0.f,
              qw = q.size() >= 4 ? (float)q[3] : 1.f;
        Mat4 out = identity();
        out.m[0][0] = (1 - 2*(qy*qy + qz*qz)) * sx;
        out.m[0][1] = (    2*(qx*qy + qz*qw)) * sx;
        out.m[0][2] = (    2*(qx*qz - qy*qw)) * sx;
        out.m[1][0] = (    2*(qx*qy - qz*qw)) * sy;
        out.m[1][1] = (1 - 2*(qx*qx + qz*qz)) * sy;
        out.m[1][2] = (    2*(qy*qz + qx*qw)) * sy;
        out.m[2][0] = (    2*(qx*qz + qy*qw)) * sz;
        out.m[2][1] = (    2*(qy*qz - qx*qw)) * sz;
        out.m[2][2] = (1 - 2*(qx*qx + qy*qy)) * sz;
        out.m[3][0] = t.size() >= 3 ? (float)t[0] : 0.f;
        out.m[3][1] = t.size() >= 3 ? (float)t[1] : 0.f;
        out.m[3][2] = t.size() >= 3 ? (float)t[2] : 0.f;
        out.m[3][3] = 1.f;
        return out;
    }

    // ---- operators ----------------------------------------------------------

    Mat4 operator*(const Mat4& o) const {
        Mat4 res;
        for (int c = 0; c < 4; c++)
            for (int row = 0; row < 4; row++)
                for (int k = 0; k < 4; k++)
                    res.m[c][row] += m[k][row] * o.m[c][k];
        return res;
    }
};

inline Vec3 transformPoint(const Mat4& m, Vec3 p) {
    return {
        m.m[0][0] * p.x + m.m[1][0] * p.y + m.m[2][0] * p.z + m.m[3][0],
        m.m[0][1] * p.x + m.m[1][1] * p.y + m.m[2][1] * p.z + m.m[3][1],
        m.m[0][2] * p.x + m.m[1][2] * p.y + m.m[2][2] * p.z + m.m[3][2],
    };
}
