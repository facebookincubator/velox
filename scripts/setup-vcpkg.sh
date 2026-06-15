#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This script sets up vcpkg for Velox dependency management.
#
# Environment variables:
# * VCPKG_ROOT: Path to vcpkg installation (default: ./vcpkg)
# * VCPKG_TRIPLET: Target triplet (auto-detected if not set)
# * VCPKG_FEATURE_FLAGS: Additional vcpkg features (e.g., "s3,gcs,abfs")
#
# Usage:
# $ scripts/setup-vcpkg.sh [install|configure|clean]
#
# Commands:
#   install   - Install vcpkg and bootstrap it
#   configure - Configure CMake with vcpkg toolchain
#   clean     - Remove vcpkg installation
#

set -e # Exit on error
set -x # Print commands that are executed

SCRIPT_DIR=$(dirname "${BASH_SOURCE[0]}")
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
VCPKG_ROOT=${VCPKG_ROOT:-"$PROJECT_ROOT/dev/vcpkg/.vcpkg"}
VCPKG_MANIFEST_DIR="$PROJECT_ROOT/dev/vcpkg"

# Detect OS and architecture
OS=$(uname -s)
ARCH=$(uname -m)

# Auto-detect triplet if not set
if [ -z "$VCPKG_TRIPLET" ]; then
    if [ "$OS" = "Darwin" ]; then
        if [ "$ARCH" = "arm64" ]; then
            VCPKG_TRIPLET="arm64-osx-release"
        else
            VCPKG_TRIPLET="x64-osx-release"
        fi
    elif [ "$OS" = "Linux" ]; then
        if [ "$ARCH" = "aarch64" ]; then
            VCPKG_TRIPLET="arm64-linux-release"
        else
            VCPKG_TRIPLET="x64-linux-release"
        fi
    else
        echo "Unsupported OS: $OS"
        exit 1
    fi
fi

echo "Using vcpkg triplet: $VCPKG_TRIPLET"
echo "VCPKG_ROOT: $VCPKG_ROOT"

function install_vcpkg {
    if [ -d "$VCPKG_ROOT" ]; then
        echo "vcpkg already exists at $VCPKG_ROOT"
        echo "Updating vcpkg..."
        cd "$VCPKG_ROOT"
        git pull
    else
        echo "Cloning vcpkg..."
        git clone --branch 2026.03.18  https://github.com/Microsoft/vcpkg.git "$VCPKG_ROOT"
    fi

    cd "$VCPKG_ROOT"
    
    # Bootstrap vcpkg
    if [ "$OS" = "Darwin" ] || [ "$OS" = "Linux" ]; then
        ./bootstrap-vcpkg.sh
    else
        echo "Unsupported OS for bootstrap: $OS"
        exit 1
    fi

    echo "vcpkg installed successfully at $VCPKG_ROOT"
}

function configure_cmake {
    BUILD_DIR=${BUILD_DIR:-"$PROJECT_ROOT/_build"}
    mkdir -p "$BUILD_DIR"
    
    cd "$BUILD_DIR"
    
    # Detect and set generator
    if command -v ninja >/dev/null 2>&1; then
        GENERATOR="Ninja"
    else
        GENERATOR="Unix Makefiles"
    fi
    
    # Build CMake arguments
    CMAKE_ARGS=(
        -G "$GENERATOR"
        -DCMAKE_BUILD_TYPE=Release
        -DCMAKE_TOOLCHAIN_FILE="$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake"
        -DVCPKG_TARGET_TRIPLET="$VCPKG_TRIPLET"
        -DVCPKG_HOST_TRIPLET="$VCPKG_TRIPLET"
        -DVCPKG_OVERLAY_PORTS="$VCPKG_MANIFEST_DIR/ports"
	    -DVCPKG_OVERLAY_TRIPLETS="$VCPKG_MANIFEST_DIR/triplets"
        -DVCPKG_MANIFEST_DIR="$VCPKG_MANIFEST_DIR"
        -DVCPKG_INSTALLED_DIR="$BUILD_DIR/vcpkg_installed"
    )
    
    # Set compilers if not already set
    CC=gcc
    CXX=g++
    if [ -z "$CC" ]; then
        if command -v clang >/dev/null 2>&1; then
            export CC=clang
        elif command -v gcc >/dev/null 2>&1; then
            export CC=gcc
        fi
    fi
    
    if [ -z "$CXX" ]; then
        if command -v clang++ >/dev/null 2>&1; then
            export CXX=clang++
        elif command -v g++ >/dev/null 2>&1; then
            export CXX=g++
        fi
    fi
    
    echo "Using generator: $GENERATOR"
    echo "Using C compiler: ${CC:-default}"
    echo "Using C++ compiler: ${CXX:-default}"
    
    # Add feature flags if specified
    # Default to duckdb if not set, but allow override via environment variable
    # Multiple features can be specified separated by semicolons (e.g., "duckdb;s3;gcs")
    VCPKG_FEATURE_FLAGS=${VCPKG_FEATURE_FLAGS:-"duckdb"}
    if [ -n "$VCPKG_FEATURE_FLAGS" ]; then
        echo "Enabling vcpkg features: $VCPKG_FEATURE_FLAGS"
        CMAKE_ARGS+=(-DVCPKG_MANIFEST_FEATURES="$VCPKG_FEATURE_FLAGS")
    fi
    
    echo "Configuring CMake with vcpkg..."
    cmake "${CMAKE_ARGS[@]}" "$PROJECT_ROOT"
    
    echo "CMake configured successfully"
    echo "To build, run: cmake --build $BUILD_DIR -j\$(nproc)"
}

function clean_vcpkg {
    if [ -d "$VCPKG_ROOT" ]; then
        echo "Removing vcpkg at $VCPKG_ROOT"
        rm -rf "$VCPKG_ROOT"
    fi
    
    BUILD_DIR=${BUILD_DIR:-"$PROJECT_ROOT/_build"}
    if [ -d "$BUILD_DIR/vcpkg_installed" ]; then
        echo "Removing vcpkg_installed at $BUILD_DIR/vcpkg_installed"
        rm -rf "$BUILD_DIR/vcpkg_installed"
    fi
    
    echo "vcpkg cleaned successfully"
}

function install_os() {
    if [ "$OS" = "Darwin" ]; then
        install_macos
    elif [ "$OS" = "Linux" ]; then
        if [ -f /etc/os-release ]; then
            . /etc/os-release
            case "$ID" in
                ubuntu)
                    case "$VERSION_ID" in
                        22.04)
                            install_ubuntu_22.04
                            ;;
                        *)
                            echo "Unsupported Ubuntu version: $VERSION_ID"
                            exit 1
                            ;;
                    esac
                    ;;
                centos)
                    case "$VERSION_ID" in
                        9)
                            install_centos_9
                            ;;
                        *)
                            echo "Unsupported CentOS version: $VERSION_ID"
                            exit 1
                            ;;
                    esac
                    ;;
                *)
                    echo "Unsupported Linux distribution: $ID"
                    exit 1
                    ;;
            esac
        else
            echo "Could not determine Linux distribution"
            exit 1
        fi
    else
        echo "Unsupported OS: $OS"
        exit 1
    fi
}

function install_centos_9() {
    yum -y install \
        wget tar zip unzip git which sudo patch \
        cmake perl-IPC-Cmd autoconf automake libtool \
        gcc-toolset-12 \
        flex bison python3 python3-pip

    dnf -y install perl-FindBin perl-Time-Piece
    dnf -y --enablerepo=crb install autoconf-archive ninja-build
}

function install_macos() {
    brew install wget pkgconfig ccache cmake autoconf automake libtool autoconf-archive bison flex
}


function install_ubuntu_22.04() {
    apt-get update && apt-get -y install \
        wget curl tar zip unzip git \
        build-essential ccache cmake ninja-build pkg-config autoconf autoconf-archive libtool \
        flex bison python3 python3-pip

        pip3 install cmake==3.31.1
}

function show_help {
    echo "Usage: $0 [install|configure|clean|help]"
    echo ""
    echo "Commands:"
    echo "  install   - Install vcpkg and bootstrap it"
    echo "  configure - Configure CMake with vcpkg toolchain"
    echo "  clean     - Remove vcpkg installation"
    echo "  help      - Show this help message"
    echo ""
    echo "Environment variables:"
    echo "  VCPKG_ROOT          - Path to vcpkg installation (default: ./vcpkg)"
    echo "  VCPKG_TRIPLET       - Target triplet (auto-detected if not set)"
    echo "  VCPKG_FEATURE_FLAGS - Additional features (e.g., 's3;gcs;abfs')"
    echo "  BUILD_DIR           - Build directory (default: ./_build)"
}

# Main execution
COMMAND=${1:-install}

case "$COMMAND" in
    install)
        install_os
        install_vcpkg
        ;;
    configure)
        configure_cmake
        ;;
    clean)
        clean_vcpkg
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo "Unknown command: $COMMAND"
        show_help
        exit 1
        ;;
esac

echo "Done!"

# Made with Bob
