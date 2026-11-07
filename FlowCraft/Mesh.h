#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

// Shared triangle-mesh representation used by both the 3D viewer and, later,
// the LBM solid-voxelization step. Kept free of SDL/rendering types so it can
// be reused unchanged by the CFD code.

struct Vec3 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;

    constexpr Vec3() = default;
    constexpr Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}

    constexpr Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    constexpr Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    constexpr Vec3 operator*(float s) const { return {x * s, y * s, z * s}; }
};

inline Vec3 Cross(const Vec3& a, const Vec3& b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

inline float Dot(const Vec3& a, const Vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline float Length(const Vec3& v) {
    return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

inline Vec3 Normalize(const Vec3& v) {
    float len = Length(v);
    if (len < 1e-12f) return {0.0f, 0.0f, 0.0f};
    return {v.x / len, v.y / len, v.z / len};
}

// One STL facet: three vertices plus the facet normal (from the file, or
// recomputed from winding order if the file's normal is degenerate).
struct Triangle {
    Vec3 v[3];
    Vec3 normal;
};

struct Bounds {
    Vec3 min{std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max()};
    Vec3 max{std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest()};

    void Expand(const Vec3& p) {
        min.x = std::min(min.x, p.x); min.y = std::min(min.y, p.y); min.z = std::min(min.z, p.z);
        max.x = std::max(max.x, p.x); max.y = std::max(max.y, p.y); max.z = std::max(max.z, p.z);
    }

    Vec3 Center() const { return (min + max) * 0.5f; }
    Vec3 Size() const { return max - min; }
};

// Per-vertex data ready for upload to a GPU vertex buffer (position + normal),
// deduplicated from the raw facet list via position+normal hashing.
struct Mesh {
    std::string sourcePath;
    std::string name;

    std::vector<Vec3> vertices;
    std::vector<Vec3> normals;      // parallel to vertices
    std::vector<uint32_t> indices;  // 3 per triangle

    Bounds bounds;

    size_t TriangleCount() const { return indices.size() / 3; }
};
