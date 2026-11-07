#include "STLLoader.h"

#include <cctype>
#include <cstdio>
#include <cstring>
#include <functional>
#include <sstream>
#include <unordered_map>

namespace {

constexpr size_t kBinaryHeaderSize = 80;
constexpr size_t kBinaryTriangleRecordSize = 50;  // 12 floats (normal+3 verts) + 2-byte attribute count

// STL's binary format is specified as little-endian regardless of host
// byte order, so these helpers decode explicitly rather than memcpy'ing
// into a float/uint32_t (which would break on a big-endian host).
uint32_t ReadU32LE(const uint8_t* p) {
    return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
}

float ReadFloatLE(const uint8_t* p) {
    uint32_t bits = ReadU32LE(p);
    float f;
    static_assert(sizeof(f) == sizeof(bits));
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

Vec3 ReadVec3LE(const uint8_t* p) {
    return {ReadFloatLE(p), ReadFloatLE(p + 4), ReadFloatLE(p + 8)};
}

Vec3 FaceNormal(const Vec3& fileNormal, const Vec3 v[3]) {
    if (Length(fileNormal) > 1e-6f) return Normalize(fileNormal);
    return Normalize(Cross(v[1] - v[0], v[2] - v[0]));
}

}  // namespace

bool STLLoader::LooksLikeAsciiSTL(const uint8_t* data, size_t size) {
    size_t probe = std::min<size_t>(size, 512);
    for (size_t i = 0; i < probe; ++i) {
        uint8_t c = data[i];
        // Any byte outside printable ASCII/whitespace rules out a text file.
        if (c == 0 || (c > 0x7E && c != '\r' && c != '\n' && c != '\t')) return false;
    }
    std::string head(reinterpret_cast<const char*>(data), probe);
    for (char& c : head) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return head.find("solid") != std::string::npos && head.find("facet") != std::string::npos;
}

STLLoader::Result STLLoader::Load(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        Result r;
        r.error = "Could not open file: " + path;
        return r;
    }

    std::fseek(f, 0, SEEK_END);
    long sizeLong = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (sizeLong < 0) {
        std::fclose(f);
        Result r;
        r.error = "Could not determine file size: " + path;
        return r;
    }
    size_t size = static_cast<size_t>(sizeLong);

    std::vector<uint8_t> data(size);
    size_t readCount = size > 0 ? std::fread(data.data(), 1, size, f) : 0;
    std::fclose(f);
    if (readCount != size) {
        Result r;
        r.error = "Short read on file: " + path;
        return r;
    }

    return LoadFromMemory(data.data(), size, path);
}

STLLoader::Result STLLoader::LoadFromMemory(const uint8_t* data, size_t size, const std::string& name) {
    // Binary STL has a fixed-size header + triangle count; if that count
    // predicts the file's actual size exactly, treat it as binary. This is
    // the standard heuristic (used by e.g. Assimp) since both STL flavors
    // use the same .stl extension and binary headers can start with "solid".
    bool isBinary = false;
    if (size >= kBinaryHeaderSize + 4) {
        uint32_t triCount = ReadU32LE(data + kBinaryHeaderSize);
        uint64_t expectedSize = kBinaryHeaderSize + 4 + static_cast<uint64_t>(triCount) * kBinaryTriangleRecordSize;
        if (expectedSize == size) isBinary = true;
    }
    if (!isBinary && !LooksLikeAsciiSTL(data, size)) {
        // Ambiguous (e.g. truncated or non-conforming file): prefer binary
        // when the file is at least large enough to hold a header+count.
        isBinary = size >= kBinaryHeaderSize + 4;
    }

    return isBinary ? LoadBinary(data, size, name) : LoadAscii(data, size, name);
}

STLLoader::Result STLLoader::LoadBinary(const uint8_t* data, size_t size, const std::string& path) {
    Result r;
    if (size < kBinaryHeaderSize + 4) {
        r.error = "File too small to be a binary STL: " + path;
        return r;
    }

    uint32_t triCount = ReadU32LE(data + kBinaryHeaderSize);
    const uint8_t* cursor = data + kBinaryHeaderSize + 4;
    const uint8_t* end = data + size;

    std::vector<Triangle> triangles;
    triangles.reserve(triCount);
    for (uint32_t i = 0; i < triCount; ++i) {
        if (cursor + kBinaryTriangleRecordSize > end) {
            r.error = "Binary STL truncated at triangle " + std::to_string(i) + ": " + path;
            return r;
        }
        Triangle t;
        Vec3 fileNormal = ReadVec3LE(cursor);
        t.v[0] = ReadVec3LE(cursor + 12);
        t.v[1] = ReadVec3LE(cursor + 24);
        t.v[2] = ReadVec3LE(cursor + 36);
        t.normal = FaceNormal(fileNormal, t.v);
        triangles.push_back(t);
        cursor += kBinaryTriangleRecordSize;
    }

    r.ok = true;
    r.mesh = BuildMesh(std::move(triangles), path);
    return r;
}

STLLoader::Result STLLoader::LoadAscii(const uint8_t* data, size_t size, const std::string& path) {
    Result r;
    std::istringstream ss(std::string(reinterpret_cast<const char*>(data), size));

    std::vector<Triangle> triangles;
    std::string token;
    Vec3 pendingNormal;
    Vec3 verts[3];
    int vertCount = 0;
    bool inFacet = false;

    while (ss >> token) {
        if (token == "facet") {
            std::string normalKw;
            if (!(ss >> normalKw >> pendingNormal.x >> pendingNormal.y >> pendingNormal.z)) {
                r.error = "Malformed 'facet normal' line in ASCII STL: " + path;
                return r;
            }
            inFacet = true;
            vertCount = 0;
        } else if (token == "vertex") {
            if (!inFacet || vertCount >= 3) {
                r.error = "Unexpected 'vertex' line in ASCII STL: " + path;
                return r;
            }
            if (!(ss >> verts[vertCount].x >> verts[vertCount].y >> verts[vertCount].z)) {
                r.error = "Malformed 'vertex' line in ASCII STL: " + path;
                return r;
            }
            ++vertCount;
        } else if (token == "endfacet") {
            if (!inFacet || vertCount != 3) {
                r.error = "Facet with != 3 vertices in ASCII STL: " + path;
                return r;
            }
            Triangle t;
            t.v[0] = verts[0];
            t.v[1] = verts[1];
            t.v[2] = verts[2];
            t.normal = FaceNormal(pendingNormal, t.v);
            triangles.push_back(t);
            inFacet = false;
        }
        // "solid", "endsolid", "outer", "loop", "endloop" and the solid name
        // are structural/decorative and don't need explicit handling here.
    }

    if (triangles.empty()) {
        r.error = "No facets found in ASCII STL: " + path;
        return r;
    }

    r.ok = true;
    r.mesh = BuildMesh(std::move(triangles), path);
    return r;
}

namespace {

// Integer key for snapping nearly-identical vertex positions together so
// adjacent facets can share a vertex. Quantized relative to the mesh's own
// scale so this works whether the STL is authored in mm or metres.
struct WeldKey {
    int64_t x, y, z;
    bool operator==(const WeldKey& o) const { return x == o.x && y == o.y && z == o.z; }
};
struct WeldKeyHash {
    size_t operator()(const WeldKey& k) const {
        size_t h = std::hash<int64_t>{}(k.x);
        h ^= std::hash<int64_t>{}(k.y) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        h ^= std::hash<int64_t>{}(k.z) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        return h;
    }
};

}  // namespace

Mesh STLLoader::BuildMesh(std::vector<Triangle>&& triangles, const std::string& path) {
    Mesh mesh;
    mesh.sourcePath = path;
    size_t slash = path.find_last_of("/\\");
    mesh.name = slash == std::string::npos ? path : path.substr(slash + 1);

    // Facet normals alone make a finely-tessellated curved surface look
    // like TV static under specular lighting (each tiny facet reflects
    // light slightly differently). Welding coincident vertices and
    // averaging their incident face normals gives smooth, continuously
    // interpolated shading instead — standard for STL viewers.
    Bounds rawBounds;
    for (const Triangle& t : triangles)
        for (const Vec3& v : t.v) rawBounds.Expand(v);
    float diagonal = Length(rawBounds.Size());
    float epsilon = std::max(diagonal * 1e-5f, 1e-6f);
    float invEpsilon = 1.0f / epsilon;

    std::unordered_map<WeldKey, uint32_t, WeldKeyHash> indexOfPosition;
    std::vector<Vec3> normalAccum;

    mesh.vertices.reserve(triangles.size());
    normalAccum.reserve(triangles.size());
    mesh.indices.reserve(triangles.size() * 3);

    for (const Triangle& t : triangles) {
        for (int i = 0; i < 3; ++i) {
            const Vec3& p = t.v[i];
            WeldKey key{static_cast<int64_t>(std::llround(p.x * invEpsilon)),
                        static_cast<int64_t>(std::llround(p.y * invEpsilon)),
                        static_cast<int64_t>(std::llround(p.z * invEpsilon))};

            auto [it, inserted] = indexOfPosition.try_emplace(key, static_cast<uint32_t>(mesh.vertices.size()));
            if (inserted) {
                mesh.vertices.push_back(p);
                normalAccum.push_back(t.normal);
                mesh.bounds.Expand(p);
            } else {
                normalAccum[it->second] = normalAccum[it->second] + t.normal;
            }
            mesh.indices.push_back(it->second);
        }
    }

    mesh.normals.resize(mesh.vertices.size());
    for (size_t i = 0; i < normalAccum.size(); ++i) mesh.normals[i] = Normalize(normalAccum[i]);

    return mesh;
}
