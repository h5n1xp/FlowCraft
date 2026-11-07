#pragma once

#include <cmath>

#include "Mesh.h"

// Column-major 4x4 matrix, laid out to match Metal's simd::float4x4 memory
// layout (4 columns of 4 floats) so it can be copied straight into a GPU
// uniform buffer without conversion.
struct Mat4 {
    float m[16] = {};

    static Mat4 Identity() {
        Mat4 r;
        r.m[0] = r.m[5] = r.m[10] = r.m[15] = 1.0f;
        return r;
    }

    // Left-handed perspective projection with NDC z in [0,1] (near=0),
    // matching SDL_GPU's documented coordinate-system convention.
    static Mat4 Perspective(float fovYRadians, float aspect, float nearZ, float farZ) {
        float yScale = 1.0f / std::tan(fovYRadians * 0.5f);
        float xScale = yScale / aspect;
        float range = farZ - nearZ;

        Mat4 r;
        r.m[0] = xScale;
        r.m[5] = yScale;
        r.m[10] = farZ / range;
        r.m[11] = 1.0f;
        r.m[14] = -nearZ * farZ / range;
        return r;
    }

    // Left-handed look-at view matrix (camera looks down +Z in view space).
    static Mat4 LookAt(const Vec3& eye, const Vec3& target, const Vec3& up) {
        Vec3 z = Normalize(target - eye);
        Vec3 x = Normalize(Cross(up, z));
        Vec3 y = Cross(z, x);

        Mat4 r;
        r.m[0] = x.x;  r.m[1] = y.x;  r.m[2] = z.x;  r.m[3] = 0.0f;
        r.m[4] = x.y;  r.m[5] = y.y;  r.m[6] = z.y;  r.m[7] = 0.0f;
        r.m[8] = x.z;  r.m[9] = y.z;  r.m[10] = z.z; r.m[11] = 0.0f;
        r.m[12] = -Dot(x, eye); r.m[13] = -Dot(y, eye); r.m[14] = -Dot(z, eye); r.m[15] = 1.0f;
        return r;
    }

    static Mat4 Translation(const Vec3& t) {
        Mat4 r = Identity();
        r.m[12] = t.x;
        r.m[13] = t.y;
        r.m[14] = t.z;
        return r;
    }

    static Mat4 UniformScale(float s) {
        Mat4 r;
        r.m[0] = r.m[5] = r.m[10] = s;
        r.m[15] = 1.0f;
        return r;
    }

    // Rodrigues' rotation formula: rotates by `radians` about a unit `axis`.
    static Mat4 RotationAxisAngle(const Vec3& axis, float radians) {
        float c = std::cos(radians);
        float s = std::sin(radians);
        float t = 1.0f - c;
        float x = axis.x, y = axis.y, z = axis.z;

        Mat4 r;
        r.m[0] = t * x * x + c;      r.m[4] = t * x * y - s * z;  r.m[8] = t * x * z + s * y;
        r.m[1] = t * x * y + s * z;  r.m[5] = t * y * y + c;      r.m[9] = t * y * z - s * x;
        r.m[2] = t * x * z - s * y;  r.m[6] = t * y * z + s * x;  r.m[10] = t * z * z + c;
        r.m[15] = 1.0f;
        return r;
    }

    // Column-major multiply: (a * b) applied to a column vector v means b
    // is applied first, then a — i.e. result = a * (b * v).
    friend Mat4 operator*(const Mat4& a, const Mat4& b) {
        Mat4 r;
        for (int col = 0; col < 4; ++col) {
            for (int row = 0; row < 4; ++row) {
                float sum = 0.0f;
                for (int k = 0; k < 4; ++k) sum += a.m[k * 4 + row] * b.m[col * 4 + k];
                r.m[col * 4 + row] = sum;
            }
        }
        return r;
    }
};
