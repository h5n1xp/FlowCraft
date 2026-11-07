#pragma once

#include <array>

// Discrete velocity sets for the Lattice Boltzmann Method. Two are offered
// so the solver can trade accuracy for speed: D3Q19 (19 directions —
// rest + 6 face + 12 edge neighbours) is the common choice for
// aerodynamics; D3Q15 (15 directions — rest + 6 face + 8 corner
// neighbours) is a genuinely different, faster lattice (fewer directions
// to stream/collide per cell), not just a lower-quality setting — it's a
// real, if less accurate, momentum-conserving LBM model used in practice.
enum class LatticeModel { D3Q15, D3Q19 };

struct LatticeDirection {
    int ex, ey, ez;  // integer lattice velocity (in cells/step)
    float weight;
    int opposite;  // index of the direction with velocity -e_i, within the same table
};

inline const std::array<LatticeDirection, 19>& D3Q19Table() {
    static const std::array<LatticeDirection, 19> table = {{
        {0, 0, 0, 1.0f / 3.0f, 0},
        {1, 0, 0, 1.0f / 18.0f, 2},
        {-1, 0, 0, 1.0f / 18.0f, 1},
        {0, 1, 0, 1.0f / 18.0f, 4},
        {0, -1, 0, 1.0f / 18.0f, 3},
        {0, 0, 1, 1.0f / 18.0f, 6},
        {0, 0, -1, 1.0f / 18.0f, 5},
        {1, 1, 0, 1.0f / 36.0f, 8},
        {-1, -1, 0, 1.0f / 36.0f, 7},
        {1, -1, 0, 1.0f / 36.0f, 10},
        {-1, 1, 0, 1.0f / 36.0f, 9},
        {1, 0, 1, 1.0f / 36.0f, 12},
        {-1, 0, -1, 1.0f / 36.0f, 11},
        {1, 0, -1, 1.0f / 36.0f, 14},
        {-1, 0, 1, 1.0f / 36.0f, 13},
        {0, 1, 1, 1.0f / 36.0f, 16},
        {0, -1, -1, 1.0f / 36.0f, 15},
        {0, 1, -1, 1.0f / 36.0f, 18},
        {0, -1, 1, 1.0f / 36.0f, 17},
    }};
    return table;
}

inline const std::array<LatticeDirection, 15>& D3Q15Table() {
    static const std::array<LatticeDirection, 15> table = {{
        {0, 0, 0, 2.0f / 9.0f, 0},
        {1, 0, 0, 1.0f / 9.0f, 2},
        {-1, 0, 0, 1.0f / 9.0f, 1},
        {0, 1, 0, 1.0f / 9.0f, 4},
        {0, -1, 0, 1.0f / 9.0f, 3},
        {0, 0, 1, 1.0f / 9.0f, 6},
        {0, 0, -1, 1.0f / 9.0f, 5},
        {1, 1, 1, 1.0f / 72.0f, 8},
        {-1, -1, -1, 1.0f / 72.0f, 7},
        {1, 1, -1, 1.0f / 72.0f, 10},
        {-1, -1, 1, 1.0f / 72.0f, 9},
        {1, -1, 1, 1.0f / 72.0f, 12},
        {-1, 1, -1, 1.0f / 72.0f, 11},
        {-1, 1, 1, 1.0f / 72.0f, 14},
        {1, -1, -1, 1.0f / 72.0f, 13},
    }};
    return table;
}

struct LatticeSet {
    const LatticeDirection* directions;
    int count;
};

inline LatticeSet GetLatticeSet(LatticeModel model) {
    switch (model) {
        case LatticeModel::D3Q15:
            return {D3Q15Table().data(), 15};
        case LatticeModel::D3Q19:
        default:
            return {D3Q19Table().data(), 19};
    }
}

inline const char* LatticeModelName(LatticeModel model) {
    switch (model) {
        case LatticeModel::D3Q15:
            return "D3Q15 (fast)";
        case LatticeModel::D3Q19:
        default:
            return "D3Q19 (accurate)";
    }
}
