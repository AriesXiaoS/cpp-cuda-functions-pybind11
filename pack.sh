#!/bin/bash
TARGET_DIR="cpp_cuda_functions"
BUILD_SRC="build/src"
SO="cpp_cuda_functions.so"
TEST_PY="test.py"
GZ="cpp_cuda_functions.tar.gz"

if [ ! -d "$TARGET_DIR" ]; then
    mkdir $TARGET_DIR
fi

# Copy the shared library
ldd $BUILD_SRC/$SO | awk '{print $3}' | xargs -I '{}' cp -v '{}' $TARGET_DIR/
cp $BUILD_SRC/$SO $TARGET_DIR/
cp $BUILD_SRC/$TEST_PY $TARGET_DIR/

# Compress the directory
# tar -czvf $GZ $TARGET_DIR/*







