# CMake Toolchain file for NuGet MSVC Toolset and Windows SDK
# This toolchain forces CMake to use NuGet-installed MSVC headers/libraries
# and Windows SDK to match CosmosAnalytics build environment.
#
# Required NuGet packages:
#   - VisualCppTools 14.38.33133
#   - Microsoft.Windows.Sdk.Cpp 10.0.22621.3233
#   - Microsoft.Windows.Sdk.Cpp.x64 10.0.22621.3233

# NuGet packages location
if(DEFINED ENV{NUGET_PACKAGES})
    set(NUGET_PACKAGES "$ENV{NUGET_PACKAGES}")
else()
    set(NUGET_PACKAGES "C:/.tools/.nuget/packages")
endif()

# Toolset configuration (matches CosmosAnalytics)
set(VCTOOLS_VERSION "14.38.33133" CACHE STRING "MSVC Toolset Version")
set(NUGET_TOOLSET_ROOT "${NUGET_PACKAGES}/visualcpptools/${VCTOOLS_VERSION}/lib/native" CACHE PATH "NuGet Toolset Root")

# Windows SDK configuration from NuGet (matches CosmosAnalytics)
set(WINSDK_PKG_VERSION "10.0.22621.3233" CACHE STRING "Windows SDK NuGet Package Version")
set(WINSDK_VERSION "10.0.22621.0" CACHE STRING "Windows SDK Version (folder name inside package)")
set(WINSDK_ROOT "${NUGET_PACKAGES}/microsoft.windows.sdk.cpp/${WINSDK_PKG_VERSION}/c" CACHE PATH "Windows SDK Root")
set(WINSDK_LIB_ROOT "${NUGET_PACKAGES}/microsoft.windows.sdk.cpp.x64/${WINSDK_PKG_VERSION}/c" CACHE PATH "Windows SDK x64 Lib Root")

# CRITICAL: Set CMAKE_SYSTEM_VERSION to prevent MSBuild from looking for a different SDK
# This tells CMake/MSBuild exactly which Windows SDK version to use
set(CMAKE_SYSTEM_VERSION "${WINSDK_VERSION}" CACHE STRING "Windows SDK Version for CMake")

# NOTE: Do NOT set CMAKE_C_COMPILER/CMAKE_CXX_COMPILER explicitly!
# This causes CMake to use "Platform Toolset = external" which MSBuild doesn't support.
# Instead, we rely on:
# 1. CC/CXX environment variables to guide compiler detection
# 2. PATH environment variable having NuGet toolset bin first
# 3. INCLUDE/LIB environment variables for header/library search paths
# 4. Compiler flags (/I, /LIBPATH) to ensure correct paths

# Set build tools from NuGet toolset (these don't trigger "external" toolset)
set(CMAKE_AR "${NUGET_TOOLSET_ROOT}/bin/amd64/lib.exe" CACHE FILEPATH "Archiver")
set(CMAKE_LINKER "${NUGET_TOOLSET_ROOT}/bin/amd64/link.exe" CACHE FILEPATH "Linker")
set(CMAKE_MT "${WINSDK_ROOT}/bin/${WINSDK_VERSION}/x64/mt.exe" CACHE FILEPATH "Manifest Tool")
set(CMAKE_RC_COMPILER "${WINSDK_ROOT}/bin/${WINSDK_VERSION}/x64/rc.exe" CACHE FILEPATH "Resource Compiler")

# Force include directories - NuGet toolset FIRST, then Windows SDK from NuGet
# This ensures we use the correct STL version (14.38) instead of VS 2022's (14.44)
set(NUGET_INCLUDE_DIRS
    "${NUGET_TOOLSET_ROOT}/include"
    "${NUGET_TOOLSET_ROOT}/atlmfc/include"
)

set(WINSDK_INCLUDE_DIRS
    "${WINSDK_ROOT}/Include/${WINSDK_VERSION}/ucrt"
    "${WINSDK_ROOT}/Include/${WINSDK_VERSION}/um"
    "${WINSDK_ROOT}/Include/${WINSDK_VERSION}/shared"
    "${WINSDK_ROOT}/Include/${WINSDK_VERSION}/winrt"
    "${WINSDK_ROOT}/Include/${WINSDK_VERSION}/cppwinrt"
)

# Set library directories (using x64 package for libs)
set(NUGET_LIB_DIRS
    "${NUGET_TOOLSET_ROOT}/lib/amd64"
    "${NUGET_TOOLSET_ROOT}/atlmfc/lib/amd64"
)

set(WINSDK_LIB_DIRS
    "${WINSDK_LIB_ROOT}/ucrt/x64"
    "${WINSDK_LIB_ROOT}/um/x64"
)

# Combine all include and library directories
set(ALL_INCLUDE_DIRS ${NUGET_INCLUDE_DIRS} ${WINSDK_INCLUDE_DIRS})
set(ALL_LIB_DIRS ${NUGET_LIB_DIRS} ${WINSDK_LIB_DIRS})

# Convert to semicolon-separated lists for CMake
string(REPLACE ";" " " INCLUDE_FLAGS_LIST "${ALL_INCLUDE_DIRS}")

# Build /I flags for compiler
set(INCLUDE_FLAGS "")
foreach(dir ${ALL_INCLUDE_DIRS})
    string(APPEND INCLUDE_FLAGS " /I\"${dir}\"")
endforeach()

# Build /LIBPATH flags for linker
set(LIBPATH_FLAGS "")
foreach(dir ${ALL_LIB_DIRS})
    string(APPEND LIBPATH_FLAGS " /LIBPATH:\"${dir}\"")
endforeach()

# Set compiler flags to include our directories FIRST
# Using CMAKE_CXX_FLAGS ensures these are prepended before VS defaults
set(CMAKE_C_FLAGS_INIT "${INCLUDE_FLAGS}" CACHE STRING "Initial C Flags")
set(CMAKE_CXX_FLAGS_INIT "${INCLUDE_FLAGS}" CACHE STRING "Initial C++ Flags")

# Set linker flags
set(CMAKE_EXE_LINKER_FLAGS_INIT "${LIBPATH_FLAGS}" CACHE STRING "Initial EXE Linker Flags")
set(CMAKE_SHARED_LINKER_FLAGS_INIT "${LIBPATH_FLAGS}" CACHE STRING "Initial Shared Linker Flags")
set(CMAKE_STATIC_LINKER_FLAGS_INIT "" CACHE STRING "Initial Static Linker Flags")

# Tell CMake about our include directories
include_directories(SYSTEM ${ALL_INCLUDE_DIRS})
link_directories(${ALL_LIB_DIRS})

# Prevent CMake from detecting the wrong MSVC installation
set(CMAKE_VS_PLATFORM_TOOLSET_VERSION "${VCTOOLS_VERSION}" CACHE STRING "VS Platform Toolset Version")

message(STATUS "Using NuGet MSVC Toolset ${VCTOOLS_VERSION}")
message(STATUS "  Compiler: ${NUGET_TOOLSET_ROOT}/bin/amd64/cl.exe")
message(STATUS "  Include paths: ${NUGET_INCLUDE_DIRS}")
message(STATUS "  Windows SDK: ${WINSDK_VERSION}")
