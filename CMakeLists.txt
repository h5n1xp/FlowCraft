# Cross-platform build for FlowCraft (macOS / Windows / Linux).
#
#   cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
#   cmake --build build --config Release
#
# SDL3 is taken from the system if found (e.g. Homebrew), otherwise fetched and
# built from source. Shaders are pre-generated into FlowCraft/generated/
# (see shaders/build_shaders.py), so no shader tools are needed to build.
# The Xcode project remains a supported way to build on macOS.
cmake_minimum_required(VERSION 3.20)
project(FlowCraft LANGUAGES C CXX)

if(APPLE)
    enable_language(OBJCXX)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release)
endif()

option(FLOWCRAFT_STATIC_SDL3 "Fetch and statically link SDL3 into the executable" OFF)

if(NOT FLOWCRAFT_STATIC_SDL3)
    find_package(SDL3 CONFIG QUIET)
endif()
if(NOT SDL3_FOUND)
    message(STATUS "Building SDL3 from source (static library)")
    set(SDL_SHARED OFF CACHE BOOL "" FORCE)
    set(SDL_STATIC ON CACHE BOOL "" FORCE)
    set(SDL_TEST_LIBRARY OFF CACHE BOOL "" FORCE)
    include(FetchContent)
    FetchContent_Declare(SDL3
        GIT_REPOSITORY https://github.com/libsdl-org/SDL.git
        GIT_TAG release-3.4.8
        GIT_SHALLOW TRUE)
    FetchContent_MakeAvailable(SDL3)
endif()

set(FC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/FlowCraft)
set(IMGUI_DIR ${FC_DIR}/ThirdParty/imgui)

set(FC_SOURCES
    ${FC_DIR}/FieldOverlay.cpp
    ${FC_DIR}/LbmReference.cpp
    ${FC_DIR}/LbmSolver.cpp
    ${FC_DIR}/OverlayRenderer.cpp
    ${FC_DIR}/PanelSolver.cpp
    ${FC_DIR}/PotentialFlowSolver.cpp
    ${FC_DIR}/Renderer.cpp
    ${FC_DIR}/STLLoader.cpp
    ${FC_DIR}/SessionFile.cpp
    ${FC_DIR}/SmokeTrail.cpp
    ${FC_DIR}/SystemMeters.cpp
    ${FC_DIR}/Voxelizer.cpp
    ${FC_DIR}/main.cpp
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_demo.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/imgui_impl_sdl3.cpp
    ${IMGUI_DIR}/imgui_impl_sdlgpu3.cpp)

# Native Cocoa file dialogs + About menu on macOS; SDL3's own portable dialogs
# elsewhere (FileDialog_SDL.cpp compiles to nothing on Apple platforms).
if(APPLE)
    list(APPEND FC_SOURCES ${FC_DIR}/FileDialog.mm)
else()
    list(APPEND FC_SOURCES ${FC_DIR}/FileDialog_SDL.cpp)
endif()

if(WIN32)
    list(APPEND FC_SOURCES ${FC_DIR}/Assets/FlowCraft.rc)
endif()

add_executable(FlowCraft ${FC_SOURCES})
target_include_directories(FlowCraft PRIVATE ${FC_DIR} ${IMGUI_DIR})
target_link_libraries(FlowCraft PRIVATE SDL3::SDL3)

if(WIN32)
    # Avoid <windows.h>'s min/max macros clashing with std::min/std::max, and
    # keep the console off in release builds.
    target_compile_definitions(FlowCraft PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS)
    if(MSVC)
        target_compile_options(FlowCraft PRIVATE /utf-8 /permissive-)
    endif()
endif()
if(APPLE)
    target_link_libraries(FlowCraft PRIVATE "-framework Cocoa" "-framework UniformTypeIdentifiers"
        "-framework IOKit" "-framework CoreFoundation")  # IOKit: SystemMeters.cpp's GPU utilisation query
    set_source_files_properties(${FC_DIR}/FileDialog.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")
endif()

# The GPU-vs-CPU solver consistency test (headless): `cmake --build build --target gpu_vs_cpu_test`
add_executable(gpu_vs_cpu_test EXCLUDE_FROM_ALL
    tests/gpu_vs_cpu_test.cpp ${FC_DIR}/LbmSolver.cpp ${FC_DIR}/LbmReference.cpp ${FC_DIR}/PanelSolver.cpp
    ${FC_DIR}/PotentialFlowSolver.cpp)
target_include_directories(gpu_vs_cpu_test PRIVATE ${FC_DIR})
target_link_libraries(gpu_vs_cpu_test PRIVATE SDL3::SDL3)
