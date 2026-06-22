# build-velox-oss.ps1 - Clean open-source Windows build (no Microsoft-internal deps)
# Uses the installed Visual Studio 2022 toolset + public vcpkg. This is the recipe
# intended for upstream CI (.github/workflows/windows.yml).
#
# Dependencies are provisioned from the public microsoft/vcpkg registry via the
# manifest in windows/vcpkg/vcpkg.json, the MSVC overlay ports in
# windows/vcpkg/ports, and the clean overlay triplet in windows/oss-triplets
# (installed VS2022 toolset, no internal NuGet toolset).
param(
  [ValidateSet("Debug","Release")][string]$BuildType="Release",
  [string]$VcpkgRoot=$(if($env:VCPKG_ROOT){$env:VCPKG_ROOT}elseif($env:VCPKG_INSTALLATION_ROOT){$env:VCPKG_INSTALLATION_ROOT}else{$null}),
  [string]$BuildDir="build-oss",
  [switch]$Configure  # configure only
)
$ErrorActionPreference="Stop"
$repo=Split-Path -Parent $PSScriptRoot
Set-Location $repo
if(-not $VcpkgRoot -or -not (Test-Path "$VcpkgRoot\vcpkg.exe")){ throw "vcpkg not found; set VCPKG_ROOT" }
$toolchain="$VcpkgRoot\scripts\buildsystems\vcpkg.cmake"

# Release-only dependency builds unless a Debug Velox build is requested.
if($BuildType -eq "Debug"){ $env:VELOX_BUILD_TYPE="debug" } else { $env:VELOX_BUILD_TYPE="release" }

# Parser generation strategy: use the pre-generated bison/flex sources that are
# committed under each parser's generated/ directory. Live generation with
# WinFlexBison on Windows is fragile (a known missing-comma codegen bug and
# parallel-invocation races), so we disable FindBISON/FindFLEX and compile the
# checked-in *.yy.cc / Scanner.cpp instead. The generated scanners include
# <FlexLexer.h>; each generated/ dir ships a copy, and we also point
# FLEX_INCLUDE_DIRS at the WinFlexBison headers so the root CMakeLists adds it
# to the global include path (see the WIN32 AND NOT FLEX_FOUND branch).
$flexIncludeDir=(Get-ChildItem "$env:LOCALAPPDATA\Microsoft\WinGet\Packages" -Recurse -Filter "FlexLexer.h" -ErrorAction SilentlyContinue | Select-Object -First 1).DirectoryName
if($flexIncludeDir){ Write-Host "FLEX_INCLUDE_DIRS=$flexIncludeDir" }
Write-Host "Configuring (VS2022, $BuildType, vcpkg=$VcpkgRoot)..." -ForegroundColor Cyan
$flags=@(
  "-S",".","-B",$BuildDir,
  "-G","Visual Studio 17 2022","-A","x64",
  "-DCMAKE_TOOLCHAIN_FILE=$toolchain",
  "-DVCPKG_TARGET_TRIPLET=x64-windows-static-md",
  "-DVCPKG_OVERLAY_TRIPLETS=$repo\windows\oss-triplets",
  "-DVCPKG_OVERLAY_PORTS=$repo\windows\vcpkg\ports",
  "-DVCPKG_MANIFEST_DIR=$repo\windows\vcpkg",
  "-DCMAKE_CXX_STANDARD=20",
  "-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded`$<`$<CONFIG:Debug>:Debug>DLL",
  "-DVELOX_BUILD_STATIC=ON","-DVELOX_BUILD_SHARED=OFF","-DVELOX_BUILD_TESTING=OFF",
  "-DVELOX_ENABLE_BENCHMARKS=OFF","-DVELOX_ENABLE_EXAMPLES=OFF",
  "-DVELOX_ENABLE_S3=OFF","-DVELOX_ENABLE_GCS=OFF","-DVELOX_ENABLE_HDFS=OFF","-DVELOX_ENABLE_GEO=OFF",
  "-DVELOX_ENABLE_ABFS=OFF","-DVELOX_ENABLE_HIVE_CONNECTOR=OFF",
  "-DVELOX_ENABLE_TPCH_CONNECTOR=OFF","-DVELOX_ENABLE_TPCDS_CONNECTOR=OFF",
  "-DVELOX_ENABLE_SPARK_FUNCTIONS=OFF",
  "-DVELOX_ENABLE_PARQUET=ON","-DVELOX_ENABLE_ARROW=ON",
  "-DVELOX_ENABLE_PRESTO_FUNCTIONS=ON","-DVELOX_ENABLE_EXEC=ON","-DVELOX_ENABLE_EXPRESSION=ON","-DVELOX_ENABLE_AGGREGATES=ON",
  # Force vcpkg-provided deps to resolve via find_package (SYSTEM) instead of
  # Velox's AUTO fallback, which would try to build them from source (fails on
  # Windows: ICU needs make, Arrow/folly/etc. have no MSVC FetchContent path).
  "-DICU_SOURCE=SYSTEM","-DProtobuf_SOURCE=SYSTEM","-DBoost_SOURCE=SYSTEM",
  "-Dgflags_SOURCE=SYSTEM","-Dglog_SOURCE=SYSTEM","-Dfmt_SOURCE=SYSTEM",
  "-Dre2_SOURCE=SYSTEM","-Dfolly_SOURCE=SYSTEM","-DArrow_SOURCE=SYSTEM",
  "-Dsimdjson_SOURCE=SYSTEM",
  "-DOPENSSL_ROOT_DIR=$repo\$BuildDir\vcpkg_installed\x64-windows-static-md",
  # Force pre-generated parser sources (see flex/bison note above).
  "-DCMAKE_DISABLE_FIND_PACKAGE_BISON=ON","-DCMAKE_DISABLE_FIND_PACKAGE_FLEX=ON"
)
if($flexIncludeDir){ $flags += "-DFLEX_INCLUDE_DIRS=$flexIncludeDir" }
cmake @flags
if($LASTEXITCODE -ne 0){ throw "configure failed ($LASTEXITCODE)" }
Write-Host "Configure OK." -ForegroundColor Green
if(-not $Configure){
  cmake --build $BuildDir --config $BuildType -- /m:8
  if($LASTEXITCODE -ne 0){ throw "build failed ($LASTEXITCODE)" }
  Write-Host "Build OK." -ForegroundColor Green
}