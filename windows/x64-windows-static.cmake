# Custom triplet for using NuGet-based MSVC toolset with vcpkg
# Based on CosmosAnalytics approach for ABI compatibility
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

# Define paths to NuGet packages (use forward slashes for CMake compatibility)
set(VCPP_TOOLS_PKG_DIR "${NUGET_ROOT}/VisualCppTools/${VCTOOLS_VERSION}")
set(WINSDK_DIR "${NUGET_ROOT}/Microsoft.Windows.Sdk.Cpp/${WINSDK_PKG_VERSION}/c")
set(WINSDK_LIB_DIR "${NUGET_ROOT}/Microsoft.Windows.Sdk.Cpp.x64/${WINSDK_PKG_VERSION}/c")

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
set(ENV{VC_ExecutablePath_x64_x64} "${VCPP_TOOLS_PKG_DIR}/lib/native/bin/amd64")
set(ENV{VC_ExecutablePath_x86_x64} "${VCPP_TOOLS_PKG_DIR}/lib/native/bin/x86_amd64")
set(ENV{VC_ExecutablePath_x64_x86} "${VCPP_TOOLS_PKG_DIR}/lib/native/bin/amd64_x86")
set(ENV{VC_ExecutablePath_x86_x86} "${VCPP_TOOLS_PKG_DIR}/lib/native/bin")

# Library paths (MSBuild uses these)
set(ENV{VC_LibraryPath_VC_x64} "${VCPP_TOOLS_PKG_DIR}/lib/native/lib/amd64")
set(ENV{VC_LibraryPath_VC_x64_Desktop} "${VCPP_TOOLS_PKG_DIR}/lib/native/lib/amd64;")

# Windows SDK environment variables
set(ENV{WindowsSdkDir} "${WINSDK_DIR}")
set(ENV{WindowsSdkDir_10} "${WINSDK_DIR}")
set(ENV{WindowsSDKVersion} "${WINSDK_VERSION}\\")
set(ENV{WindowsSDKLibVersion} "${WINSDK_VERSION}\\")
set(ENV{UCRTVersion} "${WINSDK_VERSION}")
set(ENV{WindowsSdkBinPath} "${WINSDK_DIR}/bin/")
set(ENV{WindowsSdkVerBinPath} "${WINSDK_DIR}/bin/${WINSDK_VERSION}")
set(ENV{WindowsSDKVersionedBinRoot} "${WINSDK_DIR}/bin/${WINSDK_VERSION}")
set(ENV{WindowsSDK_ExecutablePath_x64} "${WINSDK_DIR}/bin/${WINSDK_VERSION}/x64")
set(ENV{WindowsSDK_ExecutablePath_x86} "${WINSDK_DIR}/bin/${WINSDK_VERSION}/x86")
set(ENV{WindowsLibPath} "${WINSDK_DIR}/UnionMetadata/${WINSDK_VERSION}")
set(ENV{WindowsSdkLibDir} "${WINSDK_LIB_DIR}")

# ============================================================================
# PATH, INCLUDE, LIB (for both MSBuild and MSYS2/make builds like ICU)
# PATH has NuGet toolset first so cl.exe is found without needing CC/CXX
# ============================================================================

set(ENV{PATH} "${VCPP_TOOLS_PKG_DIR}/lib/native/bin/amd64;${WINSDK_DIR}/bin/${WINSDK_VERSION}/x64;$ENV{PATH}")

set(ENV{INCLUDE} "${VCPP_TOOLS_PKG_DIR}/lib/native/include;${WINSDK_DIR}/Include/${WINSDK_VERSION}/ucrt;${WINSDK_DIR}/Include/${WINSDK_VERSION}/shared;${WINSDK_DIR}/Include/${WINSDK_VERSION}/um;${WINSDK_DIR}/Include/${WINSDK_VERSION}/winrt;${VCPP_TOOLS_PKG_DIR}/lib/native/atlmfc/include")

set(ENV{LIB} "${VCPP_TOOLS_PKG_DIR}/lib/native/lib/amd64;${WINSDK_LIB_DIR}/ucrt/x64;${WINSDK_LIB_DIR}/um/x64;${VCPP_TOOLS_PKG_DIR}/lib/native/atlmfc/lib/amd64")
set(ENV{LibraryPath} "$ENV{LIB}")

# ============================================================================
# Base triplet settings
# ============================================================================

set(VCPKG_TARGET_ARCHITECTURE x64)
set(VCPKG_CRT_LINKAGE static)
set(VCPKG_LIBRARY_LINKAGE static)

# ============================================================================
# Compiler flags with /I for include paths
# These ensure NuGet headers are used even if MSBuild adds VS headers later
# ============================================================================

set(NUGET_INCLUDE_FLAGS "/I\"${VCPP_TOOLS_PKG_DIR}/lib/native/include\" /I\"${VCPP_TOOLS_PKG_DIR}/lib/native/atlmfc/include\" /I\"${WINSDK_DIR}/Include/${WINSDK_VERSION}/ucrt\" /I\"${WINSDK_DIR}/Include/${WINSDK_VERSION}/um\" /I\"${WINSDK_DIR}/Include/${WINSDK_VERSION}/shared\" /I\"${WINSDK_DIR}/Include/${WINSDK_VERSION}/winrt\"")

set(VCPKG_CXX_FLAGS "${NUGET_INCLUDE_FLAGS}")
set(VCPKG_CXX_FLAGS_DEBUG "${NUGET_INCLUDE_FLAGS}")
set(VCPKG_CXX_FLAGS_RELEASE "${NUGET_INCLUDE_FLAGS}")
set(VCPKG_C_FLAGS "${NUGET_INCLUDE_FLAGS}")
set(VCPKG_C_FLAGS_DEBUG "${NUGET_INCLUDE_FLAGS}")
set(VCPKG_C_FLAGS_RELEASE "${NUGET_INCLUDE_FLAGS}")

message(STATUS "Using NuGet MSVC Toolset ${VCTOOLS_VERSION}")
message(STATUS "  Toolset: ${VCPP_TOOLS_PKG_DIR}")
message(STATUS "  Windows SDK: ${WINSDK_DIR}")
