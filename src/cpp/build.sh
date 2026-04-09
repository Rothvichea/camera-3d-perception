#!/bin/bash
set -e
echo "Building perception_cpp module..."

PYBIND_INCLUDES=$(python3 -m pybind11 --includes)
PYTHON_EXT=$(python3-config --extension-suffix)

g++ -O3 -Wall -shared -std=c++17 -fPIC \
    ${PYBIND_INCLUDES} \
    src/cpp/perception_cpp.cpp \
    -o src/cpp/perception_cpp${PYTHON_EXT}

echo "Built: src/cpp/perception_cpp${PYTHON_EXT}"
