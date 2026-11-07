# Building FlowCraft

## Any platform (CMake)
    cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
    cmake --build build --config Release
SDL3 is used from the system if found, otherwise fetched and built from source.
On Windows/Linux the GPU backend is Vulkan (SPIR-V); on macOS it is Metal (MSL).
File dialogs use SDL3's built-in dialogs, except on macOS (native Cocoa panels).

## macOS (Xcode)
Open `FlowCraft.xcodeproj` as before (needs Homebrew SDL3).

## Shaders
Shaders are written once in GLSL (`shaders/*.glsl`). The compiled SPIR-V + MSL are
committed in `FlowCraft/generated/ShaderBlobs.h`, so building needs no shader tools.
After editing a `.glsl` file, regenerate with `python3 shaders/build_shaders.py`
(needs `glslc` from shaderc, and `shadercross` from SDL_shadercross).

## Tests
`tests/run_gpu_vs_cpu_test.sh` (or CMake target `gpu_vs_cpu_test`) runs the GPU LBM kernel
and the CPU reference on the same domain and checks they agree. Run it after touching the
kernel, collision model, or population layout.
