#include "SessionFile.h"

#include <cstdio>
#include <cstring>

namespace {

// "FCS8" — FlowCraft Session, format version 8 (bumped from 7 when
// lbmSmagorinskyCs and the UI-state fields were added, same as every
// earlier bump — an older "FCS1"-"FCS7" file just fails the magic check cleanly below
// rather than being misread field-by-field out of alignment). Written/
// read field-by-field (not a struct-level memcpy) in case alignment/
// padding ever differs across a compiler/SDK update — cheap insurance
// for a format meant to outlive any one build of the app.
constexpr char kMagic[4] = {'F', 'C', 'S', '8'};

void WriteU32(std::vector<uint8_t>& buf, uint32_t v) {
    for (int i = 0; i < 4; ++i) buf.push_back(static_cast<uint8_t>((v >> (8 * i)) & 0xFF));
}
void WriteI32(std::vector<uint8_t>& buf, int32_t v) { WriteU32(buf, static_cast<uint32_t>(v)); }
void WriteU64(std::vector<uint8_t>& buf, uint64_t v) {
    for (int i = 0; i < 8; ++i) buf.push_back(static_cast<uint8_t>((v >> (8 * i)) & 0xFF));
}
void WriteF32(std::vector<uint8_t>& buf, float v) {
    uint32_t bits;
    std::memcpy(&bits, &v, sizeof(bits));
    WriteU32(buf, bits);
}
void WriteBool(std::vector<uint8_t>& buf, bool v) { buf.push_back(v ? 1 : 0); }
void WriteVec3(std::vector<uint8_t>& buf, const Vec3& v) {
    WriteF32(buf, v.x);
    WriteF32(buf, v.y);
    WriteF32(buf, v.z);
}
void WriteMat4(std::vector<uint8_t>& buf, const Mat4& m) {
    for (float f : m.m) WriteF32(buf, f);
}
void WriteString(std::vector<uint8_t>& buf, const std::string& s) {
    WriteU32(buf, static_cast<uint32_t>(s.size()));
    buf.insert(buf.end(), s.begin(), s.end());
}
void WriteBytes(std::vector<uint8_t>& buf, const std::vector<uint8_t>& data) {
    WriteU64(buf, static_cast<uint64_t>(data.size()));
    buf.insert(buf.end(), data.begin(), data.end());
}

// Bounds-checked cursor over a loaded file's bytes — every Read* returns
// false (and leaves `ok` false for every call after) the moment a read
// would run past the end, so a truncated/corrupt file fails cleanly with
// one error instead of an out-of-bounds read.
struct Reader {
    const uint8_t* data;
    size_t size;
    size_t pos = 0;
    bool ok = true;

    bool Need(size_t n) {
        if (!ok || pos + n > size) {
            ok = false;
            return false;
        }
        return true;
    }
    uint32_t ReadU32() {
        if (!Need(4)) return 0;
        uint32_t v = 0;
        for (int i = 0; i < 4; ++i) v |= static_cast<uint32_t>(data[pos + i]) << (8 * i);
        pos += 4;
        return v;
    }
    int32_t ReadI32() { return static_cast<int32_t>(ReadU32()); }
    uint64_t ReadU64() {
        if (!Need(8)) return 0;
        uint64_t v = 0;
        for (int i = 0; i < 8; ++i) v |= static_cast<uint64_t>(data[pos + i]) << (8 * i);
        pos += 8;
        return v;
    }
    float ReadF32() {
        uint32_t bits = ReadU32();
        float f;
        std::memcpy(&f, &bits, sizeof(f));
        return f;
    }
    bool ReadBool() {
        if (!Need(1)) return false;
        return data[pos++] != 0;
    }
    Vec3 ReadVec3() {
        Vec3 v;
        v.x = ReadF32();
        v.y = ReadF32();
        v.z = ReadF32();
        return v;
    }
    Mat4 ReadMat4() {
        Mat4 m;
        for (float& f : m.m) f = ReadF32();
        return m;
    }
    std::string ReadString() {
        uint32_t len = ReadU32();
        if (!Need(len)) return {};
        std::string s(reinterpret_cast<const char*>(data + pos), len);
        pos += len;
        return s;
    }
    std::vector<uint8_t> ReadBytes() {
        uint64_t len = ReadU64();
        if (!Need(static_cast<size_t>(len))) return {};
        std::vector<uint8_t> v(data + pos, data + pos + len);
        pos += static_cast<size_t>(len);
        return v;
    }
};

// Field order is the format — keep Save()'s writes and Load()'s reads in
// exact lockstep if either ever changes.
void WriteSettings(std::vector<uint8_t>& buf, const SessionSettings& s) {
    WriteMat4(buf, s.modelOrientation);
    WriteF32(buf, s.scaleMM);
    WriteVec3(buf, s.modelPivot);
    WriteMat4(buf, s.lockedOrientation);

    WriteI32(buf, s.voxelResolution);
    WriteVec3(buf, s.voxelPaddingFactor);

    WriteI32(buf, s.lbmBackend);
    WriteI32(buf, s.lbmModel);
    WriteF32(buf, s.lbmTau);
    WriteBool(buf, s.lbmReverseFlow);
    WriteBool(buf, s.lbmFreeSlipY);
    WriteBool(buf, s.lbmFreeSlipZ);
    WriteF32(buf, s.lbmSmagorinskyCs);
    WriteF32(buf, s.lbmInletVelocityMmPerS);
    WriteI32(buf, s.lbmSubstepsPerFrame);
    WriteBool(buf, s.lbmSeedMidSpeed);
    WriteBool(buf, s.warmStartEnabled);
    WriteI32(buf, s.warmStartSteps);
    WriteI32(buf, s.warmStartResolutionDivisor);
    WriteBool(buf, s.buildOverlayOnRun);

    WriteI32(buf, s.overlayStyle);
    for (int i = 0; i < 3; ++i) WriteBool(buf, s.sliceEnabled[i]);
    for (int i = 0; i < 3; ++i) WriteI32(buf, s.sliceIndex[i]);
    WriteI32(buf, s.overlayColorMode);
    WriteBool(buf, s.overlayAutoUpdate);
    WriteI32(buf, s.overlayUpdateIntervalMs);
    WriteI32(buf, s.overlayAutoUpdateWarmupSteps);
    WriteBool(buf, s.overlayShowModel);
    WriteBool(buf, s.sliceShowParticles);
    WriteF32(buf, s.voxelAlphaPressureLo);
    WriteF32(buf, s.voxelAlphaPressureHi);
    WriteBool(buf, s.voxelAlphaRangeInitialized);
    WriteBool(buf, s.detrendPressure);
    WriteBool(buf, s.voxelNegativePressureGrey);

    WriteI32(buf, s.smokeTrailParticleCount);
    WriteI32(buf, s.smokeTrailLength);
    WriteF32(buf, s.smokeTrailSpeedScale);
    WriteF32(buf, s.smokeTrailPuffDurationSeconds);
    WriteBool(buf, s.smokeTrailPressureAlphaEnabled);

    WriteI32(buf, s.activeTab);
    WriteBool(buf, s.showInspectorPanel);
    WriteBool(buf, s.showTelemetryHud);
    WriteBool(buf, s.showLiftArrow);

    WriteF32(buf, s.cameraYaw);
    WriteF32(buf, s.cameraPitch);
    WriteF32(buf, s.cameraDistance);
}

SessionSettings ReadSettings(Reader& r) {
    SessionSettings s;
    s.modelOrientation = r.ReadMat4();
    s.scaleMM = r.ReadF32();
    s.modelPivot = r.ReadVec3();
    s.lockedOrientation = r.ReadMat4();

    s.voxelResolution = r.ReadI32();
    s.voxelPaddingFactor = r.ReadVec3();

    s.lbmBackend = r.ReadI32();
    s.lbmModel = r.ReadI32();
    s.lbmTau = r.ReadF32();
    s.lbmReverseFlow = r.ReadBool();
    s.lbmFreeSlipY = r.ReadBool();
    s.lbmFreeSlipZ = r.ReadBool();
    s.lbmSmagorinskyCs = r.ReadF32();
    s.lbmInletVelocityMmPerS = r.ReadF32();
    s.lbmSubstepsPerFrame = r.ReadI32();
    s.lbmSeedMidSpeed = r.ReadBool();
    s.warmStartEnabled = r.ReadBool();
    s.warmStartSteps = r.ReadI32();
    s.warmStartResolutionDivisor = r.ReadI32();
    s.buildOverlayOnRun = r.ReadBool();

    s.overlayStyle = r.ReadI32();
    for (int i = 0; i < 3; ++i) s.sliceEnabled[i] = r.ReadBool();
    for (int i = 0; i < 3; ++i) s.sliceIndex[i] = r.ReadI32();
    s.overlayColorMode = r.ReadI32();
    s.overlayAutoUpdate = r.ReadBool();
    s.overlayUpdateIntervalMs = r.ReadI32();
    s.overlayAutoUpdateWarmupSteps = r.ReadI32();
    s.overlayShowModel = r.ReadBool();
    s.sliceShowParticles = r.ReadBool();
    s.voxelAlphaPressureLo = r.ReadF32();
    s.voxelAlphaPressureHi = r.ReadF32();
    s.voxelAlphaRangeInitialized = r.ReadBool();
    s.detrendPressure = r.ReadBool();
    s.voxelNegativePressureGrey = r.ReadBool();

    s.smokeTrailParticleCount = r.ReadI32();
    s.smokeTrailLength = r.ReadI32();
    s.smokeTrailSpeedScale = r.ReadF32();
    s.smokeTrailPuffDurationSeconds = r.ReadF32();
    s.smokeTrailPressureAlphaEnabled = r.ReadBool();

    s.activeTab = r.ReadI32();
    s.showInspectorPanel = r.ReadBool();
    s.showTelemetryHud = r.ReadBool();
    s.showLiftArrow = r.ReadBool();

    s.cameraYaw = r.ReadF32();
    s.cameraPitch = r.ReadF32();
    s.cameraDistance = r.ReadF32();
    return s;
}

}  // namespace

bool SessionFile::Save(const std::string& path, const SessionSettings& settings, const std::string& stlFileName,
                        const std::vector<uint8_t>& stlBytes, std::string& outError) {
    std::vector<uint8_t> buf;
    buf.insert(buf.end(), kMagic, kMagic + 4);
    WriteSettings(buf, settings);
    WriteString(buf, stlFileName);
    WriteBytes(buf, stlBytes);

    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) {
        outError = "Could not open file for writing: " + path;
        return false;
    }
    size_t written = std::fwrite(buf.data(), 1, buf.size(), f);
    std::fclose(f);
    if (written != buf.size()) {
        outError = "Short write on file: " + path;
        return false;
    }
    return true;
}

SessionFile::LoadResult SessionFile::Load(const std::string& path) {
    LoadResult result;

    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        result.error = "Could not open file: " + path;
        return result;
    }
    std::fseek(f, 0, SEEK_END);
    long sizeLong = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (sizeLong < 0) {
        std::fclose(f);
        result.error = "Could not determine file size: " + path;
        return result;
    }
    std::vector<uint8_t> raw(static_cast<size_t>(sizeLong));
    size_t readCount = raw.empty() ? 0 : std::fread(raw.data(), 1, raw.size(), f);
    std::fclose(f);
    if (readCount != raw.size()) {
        result.error = "Short read on file: " + path;
        return result;
    }

    if (raw.size() < 4 || std::memcmp(raw.data(), kMagic, 4) != 0) {
        result.error = "Not a FlowCraft session file (bad magic): " + path;
        return result;
    }

    Reader r{raw.data(), raw.size(), 4};
    result.settings = ReadSettings(r);
    result.stlFileName = r.ReadString();
    result.stlBytes = r.ReadBytes();
    if (!r.ok) {
        result.error = "Session file is truncated or corrupt: " + path;
        return result;
    }
    if (result.stlBytes.empty()) {
        result.error = "Session file has no embedded STL data: " + path;
        return result;
    }

    result.ok = true;
    return result;
}
