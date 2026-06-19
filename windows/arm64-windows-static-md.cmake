# Custom triplet for using NuGet-based MSVC toolset with vcpkg
# Based on x64-windows-static.cmake but with DYNAMIC CRT linkage (/MD, /MDd)
# This is for compatibility with CosmosAnalytics which uses /MD
#
# This triplet sets environment variables that MSBuild uses to find the
# NuGet-based MSVC toolset (14.38.33133) and Windows SDK (10.0.22621.3233)
# instead of the default Visual Studio installation.
#
# CRITICAL: This approach solves two problems:
# 1. Abseil/CMake builds: MSBuild finds NuGet toolset via VC* environment variables
#    (avoids "Platform Toolset = external" error from explicit CMAKE_C_COMPILER)
# 2. ICU/MSYS2 builds: PATH has NuGet toolset first, no Windows paths in CC/CXX
#    (avoids path mangling like C:\path becoming C:.path in bash)

# NuGet package configuration
set(VCTOOLS_VERSION "14.38.33133")
set(WINSDK_PKG_VERSION "10.0.22621.3233")
set(WINSDK_VERSION "10.0.22621.0")

# NuGet packages location  
if(DEFINED ENV{NUGET_PACKAGES})
    set(NUGET_ROOT "$ENV{NUGET_PACKAGES}")
else()
    set(NUGET_ROOT "C:/.tools/.nuget/packages")
endif()

# Use standard vcpkg Windows toolchain (not custom one which triggers "external" toolset)
set(VCPKG_CHAINLOAD_TOOLCHAIN_FILE ${VCPKG_ROOT_DIR}/scripts/toolchains/windows.cmake)
set(VCPKG_PLATFORM_TOOLSET v143)

# Don't recompile if the folder locations change
set(VCPKG_ENV_PASSTHROUGH_UNTRACKED NUGET_PACKAGES)

# VELOX_BUILD_TYPE is tracked (in ABI hash) so debug/release get separate binary caches
set(VCPKG_ENV_PASSTHROUGH VELOX_BUILD_TYPE)

# For Release builds: set VCPKG_BUILD_TYPE=release so vcpkg only builds the release
# config (~40 min). For Debug builds: do NOT set VCPKG_BUILD_TYPE because several vcpkg
# ports (rapidjson, zstd, ...) fail with debug-only builds due to cmake install rules
# not generating the main *Targets.cmake file without a release pass. Debug builds take
# ~80 min for vcpkg deps but are more stable.
if(DEFINED ENV{VELOX_BUILD_TYPE} AND NOT "$ENV{VELOX_BUILD_TYPE}" STREQUAL "debug")
    set(VCPKG_BUILD_TYPE "$ENV{VELOX_BUILD_TYPE}")
endif()

# Define paths to NuGet packages (use forward slashes for CMake compatibility)
set(VCPP_TOOLS_PKG_DIR "${NUGET_ROOT}/VisualCppTools/${VCTOOLS_VERSION}")
set(WINSDK_DIR "${NUGET_ROOT}/Microsoft.Windows.Sdk.Cpp/${WINSDK_PKG_VERSION}/c")
set(WINSDK_LIB_DIR "${NUGET_ROOT}/Microsoft.Windows.Sdk.Cpp.arm64/${WINSDK_PKG_VERSION}/c")

# ============================================================================
# MSBuild Environment Variables (for CMake/Visual Studio generator builds)
# These allow MSBuild to find the NuGet toolset without explicit compiler paths
# ============================================================================

set(ENV{DisableRegistryUse} "true")
set(ENV{VCInstallDir} "${VCPP_TOOLS_PKG_DIR}/lib/native/")
set(ENV{VCToolsInstallDir} "${VCPP_TOOLS_PKG_DIR}/lib/native/")
set(ENV{VCToolsInstallDir_160} "${VCPP_TOOLS_PKG_DIR}/lib/native/")
set(ENV{VCToolsVersion} "${VCTOOLS_VERSION}")
set(ENV{CheckMSVCComponents} "false")

# Executable paths for compilers and tools (MSBuild uses these)
set(ENV{VC_ExecutablePath_x64_x64} "${VCPP_TOOLS_PKG_DIR}/lib/native/bin/amd64_arm64")
set(ENV{VC_ExecutablePath_x86_x64} "${VCPP_TOOLS_PKG_DIR}/lib/native/bin/x86_amd64")
set(ENV{VC_ExecutablePath_x64_x86} "${VCPP_TOOLS_PKG_DIR}/lib/native/bin/amd64_x86")
set(ENV{VC_ExecutablePath_x86_x86} "${VCPP_TOOLS_PKG_DIR}/lib/native/bin")

# Library paths (MSBuild uses these)
set(ENV{VC_LibraryPath_VC_x64} "${VCPP_TOOLS_PKG_DIR}/lib/native/lib/arm64")
set(ENV{VC_LibraryPath_VC_x64_Desktop} "${VCPP_TOOLS_PKG_DIR}/lib/native/lib/arm64;")

# Windows SDK environment variables
set(ENV{WindowsSdkDir} "${WINSDK_DIR}")
set(ENV{WindowsSdkDir_10} "${WINSDK_DIR}")
set(ENV{WindowsSDKVersion} "${WINSDK_VERSION}\\")
set(ENV{WindowsSDKLibVersion} "${WINSDK_VERSION}\\")
set(ENV{UCRTVersion} "${WINSDK_VERSION}")
set(ENV{WindowsSdkBinPath} "${WINSDK_DIR}/bin/")
set(ENV{WindowsSdkVerBinPath} "${WINSDK_DIR}/bin/${WINSDK_VERSION}")
set(ENV{WindowsSDKVersionedBinRoot} "${WINSDK_DIR}/bin/${WINSDK_VERSION}")
# SDK tools (rc.exe, mt.exe) must be x64-hosted even when cross-compiling for ARM64
set(ENV{WindowsSDK_ExecutablePath_x64} "${WINSDK_DIR}/bin/${WINSDK_VERSION}/x64")
set(ENV{WindowsSDK_ExecutablePath_x86} "${WINSDK_DIR}/bin/${WINSDK_VERSION}/x86")
set(ENV{WindowsLibPath} "${WINSDK_DIR}/UnionMetadata/${WINSDK_VERSION}")
set(ENV{WindowsSdkLibDir} "${WINSDK_LIB_DIR}")

# ============================================================================
# PATH, INCLUDE, LIB (for both MSBuild and MSYS2/make builds like ICU)
# PATH has NuGet toolset first so cl.exe is found without needing CC/CXX
# ============================================================================

# PATH: cross-compiler first, then x64-hosted SDK tools (rc.exe, mt.exe can't be ARM64 on x64 host)
set(ENV{PATH} "${VCPP_TOOLS_PKG_DIR}/lib/native/bin/amd64_arm64;${VCPP_TOOLS_PKG_DIR}/lib/native/bin/amd64;${WINSDK_DIR}/bin/${WINSDK_VERSION}/x64;$ENV{PATH}")

set(ENV{INCLUDE} "${VCPP_TOOLS_PKG_DIR}/lib/native/include;${WINSDK_DIR}/Include/${WINSDK_VERSION}/ucrt;${WINSDK_DIR}/Include/${WINSDK_VERSION}/shared;${WINSDK_DIR}/Include/${WINSDK_VERSION}/um;${WINSDK_DIR}/Include/${WINSDK_VERSION}/winrt;${VCPP_TOOLS_PKG_DIR}/lib/native/atlmfc/include")

set(ENV{LIB} "${VCPP_TOOLS_PKG_DIR}/lib/native/lib/arm64;${WINSDK_LIB_DIR}/ucrt/arm64;${WINSDK_LIB_DIR}/um/arm64;${VCPP_TOOLS_PKG_DIR}/lib/native/atlmfc/lib/arm64")
set(ENV{LibraryPath} "$ENV{LIB}")

# ============================================================================
# Base triplet settings - DYNAMIC CRT for CosmosAnalytics compatibility
# ============================================================================

set(VCPKG_TARGET_ARCHITECTURE arm64)
set(VCPKG_CRT_LINKAGE dynamic)
set(VCPKG_LIBRARY_LINKAGE static)

# ============================================================================
# Compiler flags with /I for include paths
# These ensure NuGet headers are used even if MSBuild adds VS headers later
# ============================================================================

# For ARM64 (Ninja generator), do NOT use /I flags in VCPKG_CXX_FLAGS.
# Ninja/cl.exe reads the INCLUDE environment variable directly (unlike VS generator).
# Putting /I paths in VCPKG_CXX_FLAGS causes cmake command-line arg splitting issues
# because spaces in the multi-path value break the -D quoting.
# The INCLUDE env var is already set above with the correct paths.

set(VCPKG_CXX_FLAGS "")
set(VCPKG_CXX_FLAGS_DEBUG "")
set(VCPKG_CXX_FLAGS_RELEASE "")
set(VCPKG_C_FLAGS "")
set(VCPKG_C_FLAGS_DEBUG "")
set(VCPKG_C_FLAGS_RELEASE "")

message(STATUS "Using NuGet MSVC Toolset ${VCTOOLS_VERSION} with DYNAMIC CRT (/MD) - ARM64 cross-compile")
message(STATUS "  Toolset: ${VCPP_TOOLS_PKG_DIR}")
message(STATUS "  Windows SDK: ${WINSDK_DIR}")
