#!/bin/bash
# Builds and runs the GPU-vs-CPU consistency test (macOS, Homebrew SDL3).
set -e
cd "$(dirname "$0")"
SRC=../FlowCraft
clang++ -std=c++20 -O2 gpu_vs_cpu_test.cpp $SRC/LbmSolver.cpp $SRC/LbmReference.cpp $SRC/PanelSolver.cpp \
  -I/opt/homebrew/include -L/opt/homebrew/lib -lSDL3 -Wl,-rpath,/opt/homebrew/lib -o /tmp/gpu_vs_cpu_test
/tmp/gpu_vs_cpu_test
