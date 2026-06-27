# Velox vcpkg Integration

This directory contains vcpkg configuration files for managing Velox dependencies using vcpkg package manager.

## Overview

vcpkg is a cross-platform package manager that simplifies dependency management. This integration provides an alternative to the traditional script-based dependency installation used in Velox.

## Files

- **vcpkg.json**: Manifest file declaring all Velox dependencies and their versions
- **vcpkg-configuration.json**: Configuration for vcpkg registries and overlays
- **triplets/**: Custom triplet files for different platforms and build configurations
- **ports/**: Custom port overlays for packages not in vcpkg or requiring patches

## Quick Start

### 1. Install vcpkg

```bash
# From the Velox root directory
./scripts/setup-vcpkg.sh install
```

This will clone vcpkg and bootstrap it.

### 2. Configure CMake with vcpkg

```bash
./scripts/setup-vcpkg.sh configure
```

This will configure CMake to use vcpkg for dependency management.

### 3. Build Velox

```bash
cmake --build _build -j$(nproc)
```

## Features

The vcpkg manifest supports optional features that can be enabled:

- **s3**: AWS S3 support
- **gcs**: Google Cloud Storage support
- **abfs**: Azure Blob File System support
- **hdfs**: Hadoop Distributed File System support
- **duckdb**: DuckDB integration
- **arrow**: Apache Arrow support

### Enabling Features

Set the `VCPKG_FEATURE_FLAGS` environment variable before configuring:

```bash
export VCPKG_FEATURE_FLAGS="s3;gcs;abfs"
./scripts/setup-vcpkg.sh configure
```

Or pass it directly to CMake:

```bash
cmake -DCMAKE_TOOLCHAIN_FILE=vcpkg/scripts/buildsystems/vcpkg.cmake \
      -DVCPKG_MANIFEST_FEATURES="s3;gcs" \
      -DVCPKG_TARGET_TRIPLET=arm64-osx-release \
      -DVCPKG_MANIFEST_DIR=dev/vcpkg \
      ..
```

## Triplets

Custom triplets are provided for different platforms:

- **arm64-osx-release**: macOS ARM64 (Apple Silicon)
- **x64-osx-release**: macOS x86_64 (Intel)
- **arm64-linux-release**: Linux ARM64
- **x64-linux-release**: Linux x86_64

All triplets build dependencies as static libraries (except glog and gflags which are built as shared to avoid flag registration issues).

## Environment Variables

- `VCPKG_ROOT`: Path to vcpkg installation (default: `./vcpkg`)
- `VCPKG_TRIPLET`: Target triplet (auto-detected if not set)
- `VCPKG_FEATURE_FLAGS`: Semicolon-separated list of features to enable
- `BUILD_DIR`: Build directory (default: `./_build`)

## Dependency Versions

The manifest pins specific versions of dependencies to ensure compatibility:

- fmt: 11.2.0
- protobuf: 21.8
- xsimd: 10.0.0
- glog: 0.6.0
- gflags: 2.2.2
- re2: 2024-07-02
- And more...

See `vcpkg.json` for the complete list.

## Advantages over Script-based Installation

1. **Cross-platform**: Works consistently across macOS, Linux, and Windows
2. **Reproducible**: Locked versions ensure consistent builds
3. **Faster**: Binary caching reduces build times
4. **Easier maintenance**: Centralized dependency management
5. **Better integration**: Native CMake integration

## Comparison with Traditional Setup

### Traditional (scripts/setup-macos.sh)
```bash
./scripts/setup-macos.sh
make
```

### vcpkg-based
```bash
./scripts/setup-vcpkg.sh install
./scripts/setup-vcpkg.sh configure
cmake --build _build -j$(nproc)
```

## Troubleshooting

### Clean vcpkg installation

```bash
./scripts/setup-vcpkg.sh clean
```

### Force rebuild of dependencies

```bash
rm -rf _build/vcpkg_installed
./scripts/setup-vcpkg.sh configure
```

### Use specific vcpkg version

```bash
cd vcpkg
git checkout <commit-hash>
./bootstrap-vcpkg.sh
cd ..
./scripts/setup-vcpkg.sh configure
```

## Contributing

When adding new dependencies:

1. Add the dependency to `vcpkg.json`
2. If a specific version is required, add it to the `overrides` section
3. If the package needs patches, create a port overlay in `ports/`
4. Update this README with any new features or requirements

## References

- [vcpkg Documentation](https://vcpkg.io/)
- [vcpkg Manifest Mode](https://vcpkg.io/en/docs/users/manifests.html)
- [vcpkg Triplets](https://vcpkg.io/en/docs/users/triplets.html)
- [Gluten vcpkg Integration](https://github.com/apache/gluten/pull/11563)