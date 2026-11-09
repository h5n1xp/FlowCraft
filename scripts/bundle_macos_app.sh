#!/usr/bin/env bash
# Bundles the compiled FlowCraft executable into a standalone macOS FlowCraft.app bundle.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Find FlowCraft binary
BUILD_BIN=""
if [ -f "${REPO_DIR}/build/FlowCraft" ]; then
    BUILD_BIN="${REPO_DIR}/build/FlowCraft"
else
    # Check Xcode DerivedData
    XCODE_BIN="$(find "${HOME}/Library/Developer/Xcode/DerivedData" -type f -path "*/Build/Products/Debug/FlowCraft" -perm +111 2>/dev/null | head -n 1 || true)"
    if [ -n "${XCODE_BIN}" ] && [ -f "${XCODE_BIN}" ]; then
        BUILD_BIN="${XCODE_BIN}"
    fi
fi

if [ -z "${BUILD_BIN}" ] || [ ! -f "${BUILD_BIN}" ]; then
    echo "Error: FlowCraft binary not found. Build the project first via Xcode or CMake." >&2
    exit 1
fi

APP_DIR="${REPO_DIR}/FlowCraft.app"
echo "Creating application bundle at: ${APP_DIR}..."
rm -rf "${APP_DIR}"
mkdir -p "${APP_DIR}/Contents/MacOS"
mkdir -p "${APP_DIR}/Contents/Resources"

cp -p "${BUILD_BIN}" "${APP_DIR}/Contents/MacOS/FlowCraft"

if [ -f "${REPO_DIR}/FlowCraft/Assets/FlowCraft.icns" ]; then
    cp -p "${REPO_DIR}/FlowCraft/Assets/FlowCraft.icns" "${APP_DIR}/Contents/Resources/FlowCraft.icns"
fi

cat << 'EOF' > "${APP_DIR}/Contents/Info.plist"
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>FlowCraft</string>
    <key>CFBundleIconFile</key>
    <string>FlowCraft</string>
    <key>CFBundleIdentifier</key>
    <string>com.mattparsons.flowcraft</string>
    <key>CFBundleName</key>
    <string>FlowCraft</string>
    <key>CFBundleDisplayName</key>
    <string>FlowCraft</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>0.1</string>
    <key>CFBundleVersion</key>
    <string>1</string>
    <key>NSHighResolutionCapable</key>
    <true/>
</dict>
</plist>
EOF

echo "FlowCraft.app bundle successfully created!"
