#!/bin/bash
set -e

# Define installation directory inside the project
PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
TOOLS_DIR="$PROJECT_DIR/tools"
mkdir -p "$TOOLS_DIR"

# LLVM Version to install (Using 14.0.0 as a stable baseline)
LLVM_VER="14.0.0"
# Choose a build compatible with most Linux kernels (Ubuntu 18.04 build is usually glibc compatible enough)
LLVM_URL="https://github.com/llvm/llvm-project/releases/download/llvmorg-${LLVM_VER}/clang+llvm-${LLVM_VER}-x86_64-linux-gnu-ubuntu-18.04.tar.xz"
DIR_NAME="clang+llvm-${LLVM_VER}-x86_64-linux-gnu-ubuntu-18.04"

cd "$TOOLS_DIR"

if [ -d "$DIR_NAME" ]; then
    echo "LLVM seems to be already installed in $TOOLS_DIR/$DIR_NAME"
    echo "Skipping download."
else
    echo "Downloading LLVM $LLVM_VER..."
    wget -O llvm.tar.xz "$LLVM_URL"
    
    echo "Extracting LLVM..."
    tar -xf llvm.tar.xz
    rm llvm.tar.xz
    
    echo "LLVM installed successfully to $TOOLS_DIR/$DIR_NAME"
fi

# Verify installation
BIN_PATH="$TOOLS_DIR/$DIR_NAME/bin"
echo "Verifying llvm-as..."
"$BIN_PATH/llvm-as" --version

echo ""
echo "Setup complete! The evaluation script has been configured to look in:"
echo "  $BIN_PATH"
