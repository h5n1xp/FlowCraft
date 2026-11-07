#pragma once

#include <string>

#include "Mesh.h"

// Loads a binary or ASCII STL file from disk into a Mesh. Format is detected
// from content, not file extension, since both STL flavors commonly use .stl.
class STLLoader {
public:
    struct Result {
        bool ok = false;
        std::string error;  // populated when ok == false
        Mesh mesh;          // populated when ok == true
    };

    static Result Load(const std::string& path);

    // Same format-detection-and-parse logic as Load(), on an in-memory
    // buffer instead of a file on disk — e.g. an STL embedded in a saved
    // session file (see SessionFile.h). `name` is purely cosmetic (goes
    // into the resulting Mesh's sourcePath/name, same as Load()'s `path`
    // argument does), not a file that's opened.
    static Result LoadFromMemory(const uint8_t* data, size_t size, const std::string& name);

private:
    static bool LooksLikeAsciiSTL(const uint8_t* data, size_t size);
    static Result LoadBinary(const uint8_t* data, size_t size, const std::string& path);
    static Result LoadAscii(const uint8_t* data, size_t size, const std::string& path);
    static Mesh BuildMesh(std::vector<Triangle>&& triangles, const std::string& path);
};
