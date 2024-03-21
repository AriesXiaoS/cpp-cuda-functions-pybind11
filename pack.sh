#!/bin/bash
TARGET_DIR="cuda_functions"
BUILD_SRC="build/src"
SO="cuda_functions.so"
TEST_PY="test.py"
GZ="cuda_functions.tar.gz"

if [ ! -d "$TARGET_DIR" ]; then
    mkdir $TARGET_DIR
fi

# Copy the shared library
ldd $BUILD_SRC/$SO | awk '{print $3}' | xargs -I '{}' cp -v '{}' $TARGET_DIR/
cp $BUILD_SRC/$SO $TARGET_DIR/
cp $BUILD_SRC/$TEST_PY $TARGET_DIR/

# Compress the directory
# tar -czvf $GZ $TARGET_DIR/*

# mkdir cuda_functions
# ldd build/src/cuda_functions.so | awk '{print $3}' | xargs -I '{}' cp -v '{}' cuda_functions/
# cp build/src/cuda_functions.so cuda_functions/
# cp build/src/test.py cuda_functions/

# tar -czvf cuda_functions.tar.gz cuda_functions/*






