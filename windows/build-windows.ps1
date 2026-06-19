# Velox Windows Build Script with vcpkg
# This script builds Velox on Windows using vcpkg for dependencies
# 
# IMPORTANT: This script uses MSVC toolset v143 version 14.38.33133 to match
#            the CosmosAnalytics build environment and ensure ABI compatibility.
#            Using a different toolset version will cause linker errors (LNK2001, LNK2019)
#            when linking with CosmosAnalytics projects.
#
# Usage:
#   .\build-windows.ps1                        # Standard production build (Release, x64)
#   .\build-windows.ps1 -Architecture arm64    # Build for ARM64
#   .\build-windows.ps1 -BuildType Debug       # Debug build
#   .\build-windows.ps1 -BuildType Release     # Release build (default)
#   .\build-windows.ps1 -Minimal               # Minimal build (types, expressions only)
#   .\build-windows.ps1 -CleanBuild            # Clean rebuild
#   .\build-windows.ps1 -WithTests             # Build with testing enabled
#   .\build-windows.ps1 -VCToolsVersion "14.38.33133"  # Specify custom toolset (advanced)
#
# Prerequisites:
#   - Visual Studio 2022 with C++ tools
#   - MSVC v143 toolset version 14.38.33133 (installed via NuGet package VisualCppTools.14.38.33133)
#   - CMake 3.20+
#   - vcpkg (will auto-detect or use VCPKG_ROOT env var)
#   - WinFlexBison (will auto-install via winget if missing)
#
# Production Build Includes:
#   - Apache Arrow integration
#   - Azure Blob File System (ABFS) connector
#   - Parquet file format support
#   - Hive connector with metastore
#   - Query execution engine
#   - Aggregate functions
#   - Presto scalar functions (default ON)
#
# Windows-Specific Fixes Applied in CMakeLists.txt:
#   - FLEX_INCLUDE_DIRS path resolution for FlexLexer.h
#   - IMPORTED_IMPLIB properties for vcpkg DLL targets (snappy, zstd, lz4, lzo2)
#   - BISON_PKGDATADIR environment variable for parser generation
#   - Type calculation CMake version update (3.0.0 -> 3.10.0)

param(
    [string]$VcpkgRoot = "$env:VCPKG_ROOT",
    [ValidateSet("Debug", "Release")]
    [string]$BuildType = "Release",
    [string]$BuildDir = "build",
    [switch]$CleanBuild = $false,
    [switch]$Minimal = $false,
    [switch]$WithTests = $false,
    [switch]$NoMono = $false,
    [int]$Parallelism = 12,  # MSBuild parallelism (/m:N)
    [string]$VCToolsVersion = "14.38.33133",  # MSVC toolset version (must match CosmosAnalytics)
    [string]$WindowsSDKPkgVersion = "10.0.22621.3233",  # Windows SDK NuGet package version (must match CosmosAnalytics)
    [string]$WindowsSDKVersion = "10.0.22621.0",  # Actual Windows SDK version inside the package
    [string]$VcpkgTriplet = "",  # Triplet name (auto-detected from Architecture if empty)
    [ValidateSet("x64", "arm64")]
    [string]$Architecture = "x64"  # Target architecture: x64 or arm64
)

# Change to script's parent directory (Velox root)
$scriptDir = Split-Path -Parent $PSScriptRoot
Set-Location $scriptDir

# ============================================================================
# Dependency Detection and Installation
# ============================================================================

# 1. Find and verify vcpkg
Write-Host "Checking dependencies..." -ForegroundColor Cyan

# Resolve NuGet packages root early - used for toolset and SDK paths throughout the script
# Respects NUGET_PACKAGES env var (set by CI/pipeline), falls back to local dev default
$nugetPackages = if ($env:NUGET_PACKAGES) { $env:NUGET_PACKAGES } else { "C:\.tools\.nuget\packages" }
Write-Host "NuGet packages root: $nugetPackages" -ForegroundColor Cyan

# Auto-detect vcpkg triplet from architecture if not specified
if ([string]::IsNullOrEmpty($VcpkgTriplet)) {
    $VcpkgTriplet = "$Architecture-windows-static-md"
}

# Architecture-specific path components
$isArm64 = ($Architecture -eq "arm64")
# NuGet toolset uses "amd64" for x64, "arm64" for ARM64 target lib dirs
# Cross-compiler: amd64 -> amd64 (native) or amd64 -> amd64_arm64 (cross)
$targetLibDir = if ($isArm64) { "arm64" } else { "amd64" }
$compilerBinDir = if ($isArm64) { "amd64_arm64" } else { "amd64" }
# Windows SDK platform subfolder (x64 or arm64)
$sdkPlatform = if ($isArm64) { "arm64" } else { "x64" }
# Windows SDK NuGet package name suffix
$sdkPkgPlatform = if ($isArm64) { "arm64" } else { "x64" }
# CMake generator platform
$cmakePlatform = if ($isArm64) { "ARM64" } else { "x64" }

Write-Host "Target architecture: $Architecture" -ForegroundColor Cyan

# 1.1 Verify MSVC toolset version
Write-Host "Verifying MSVC toolset version $VCToolsVersion..." -ForegroundColor Cyan
$vcToolsPath = "$nugetPackages\visualcpptools\$VCToolsVersion"
if (-not (Test-Path $vcToolsPath)) {
    Write-Host "❌ Required MSVC toolset v143 version $VCToolsVersion not found" -ForegroundColor Red
    Write-Host "   Expected location: $vcToolsPath" -ForegroundColor Yellow
    Write-Host "" -ForegroundColor Yellow
    Write-Host "   This toolset is required for ABI compatibility with CosmosAnalytics." -ForegroundColor Yellow
    Write-Host "   It should be installed automatically when building CosmosAnalytics projects." -ForegroundColor Yellow
    Write-Host "" -ForegroundColor Yellow
    Write-Host "   To install manually:" -ForegroundColor Yellow
    Write-Host "   1. Add VisualCppTools package reference to your project" -ForegroundColor Yellow
    Write-Host "   2. Or download from: https://www.nuget.org/packages/VisualCppTools/$VCToolsVersion" -ForegroundColor Yellow
    Write-Host "" -ForegroundColor Yellow
    Write-Host "   Alternative: Use a different toolset version with -VCToolsVersion parameter" -ForegroundColor Yellow
    Write-Host "   (Note: This may cause ABI incompatibility with CosmosAnalytics)" -ForegroundColor Yellow
    exit 1
}
Write-Host "✓ Found MSVC toolset v143 version $VCToolsVersion at: $vcToolsPath" -ForegroundColor Green

# Use the NuGet toolset directly
$vsToolsetPath = Join-Path $vcToolsPath "lib\native"
$nugetToolsetPath = $vsToolsetPath

Write-Host "✓ MSVC toolset accessible at: $vsToolsetPath" -ForegroundColor Green

# Force use of C:\vcpkg to avoid VS-bundled vcpkg
if (Test-Path "C:\vcpkg\vcpkg.exe") {
    $VcpkgRoot = "C:\vcpkg"
} elseif ([string]::IsNullOrEmpty($VcpkgRoot)) {
    # Try other common locations
    $commonPaths = @(
        "C:\src\vcpkg",
        "$env:LOCALAPPDATA\vcpkg",
        "$env:USERPROFILE\vcpkg"
    )
    
    foreach ($path in $commonPaths) {
        if (Test-Path "$path\vcpkg.exe") {
            $VcpkgRoot = $path
            break
        }
    }
    
    if ([string]::IsNullOrEmpty($VcpkgRoot)) {
        Write-Host "❌ vcpkg not found. Please install vcpkg or set VCPKG_ROOT environment variable" -ForegroundColor Red
        Write-Host "   Install vcpkg from: https://github.com/microsoft/vcpkg" -ForegroundColor Yellow
        Write-Host "   Or set `$env:VCPKG_ROOT to your vcpkg installation path" -ForegroundColor Yellow
        exit 1
    }
}

Write-Host "✓ Using vcpkg from: $VcpkgRoot" -ForegroundColor Green

# Verify vcpkg exists
if (-not (Test-Path "$VcpkgRoot\vcpkg.exe")) {
    Write-Host "❌ vcpkg.exe not found at: $VcpkgRoot" -ForegroundColor Red
    exit 1
}

$ToolchainFile = "$VcpkgRoot\scripts\buildsystems\vcpkg.cmake"

# 2. Check for WinFlexBison (required for parser generation)
Write-Host "Checking for WinFlexBison..." -ForegroundColor Cyan
$bisonDataDir = $null

# Try to find WinFlexBison in common locations
$bisonExeDir = $null
$wingetPaths = Get-ChildItem -Path "$env:LOCALAPPDATA\Microsoft\WinGet\Packages" -Filter "WinFlexBison*" -Directory -ErrorAction SilentlyContinue
if ($wingetPaths) {
    $bisonDataDir = Join-Path $wingetPaths[0].FullName "data"
    $bisonExeDir = $wingetPaths[0].FullName
}

# Also check if bison / win_bison is in PATH
$bisonInPath = Get-Command win_bison -ErrorAction SilentlyContinue
if (-not $bisonInPath) { $bisonInPath = Get-Command bison -ErrorAction SilentlyContinue }
if ($bisonInPath -and [string]::IsNullOrEmpty($bisonDataDir)) {
    $bisonExePath = $bisonInPath.Source
    $bisonExeDir = Split-Path -Parent $bisonExePath
    $possibleDataDir = Join-Path $bisonExeDir "data"
    if (Test-Path $possibleDataDir) {
        $bisonDataDir = $possibleDataDir
    }
}

# If not found, try to install via winget
if ([string]::IsNullOrEmpty($bisonDataDir) -or -not (Test-Path $bisonDataDir)) {
    Write-Host "WinFlexBison not found. Attempting to install via winget..." -ForegroundColor Yellow
    
    try {
        $wingetCmd = Get-Command winget -ErrorAction Stop
        Write-Host "Installing WinFlexBison..." -ForegroundColor Yellow
        & winget install --id=lexxmark.WinFlexBison --silent --accept-source-agreements --accept-package-agreements
        
        # Wait a moment for installation to complete
        Start-Sleep -Seconds 3
        
        # Try to find it again
        $wingetPaths = Get-ChildItem -Path "$env:LOCALAPPDATA\Microsoft\WinGet\Packages" -Filter "WinFlexBison*" -Directory -ErrorAction SilentlyContinue
        if ($wingetPaths) {
            $bisonDataDir = Join-Path $wingetPaths[0].FullName "data"
            $bisonExeDir = $wingetPaths[0].FullName
        }
    } catch {
        Write-Host "❌ Could not install WinFlexBison automatically." -ForegroundColor Red
        Write-Host "   Please install manually from: https://github.com/lexxmark/winflexbison" -ForegroundColor Yellow
        Write-Host "   Or install via winget: winget install lexxmark.WinFlexBison" -ForegroundColor Yellow
        exit 1
    }
}

if ([string]::IsNullOrEmpty($bisonDataDir) -or -not (Test-Path $bisonDataDir)) {
    Write-Host "❌ WinFlexBison data directory not found" -ForegroundColor Red
    Write-Host "   Please install WinFlexBison: winget install lexxmark.WinFlexBison" -ForegroundColor Yellow
    exit 1
}

Write-Host "✓ Found WinFlexBison at: $bisonDataDir" -ForegroundColor Green
$env:BISON_PKGDATADIR = $bisonDataDir

# Add WinFlexBison executables to PATH so CMake FindBISON can locate win_bison.exe
if ($bisonExeDir) {
    $env:PATH = "$bisonExeDir;$env:PATH"
    Write-Host "✓ Added WinFlexBison exe dir to PATH: $bisonExeDir" -ForegroundColor Green
}

# Determine bison/flex executable paths for explicit CMake variables
$bisonExe = Join-Path $bisonExeDir "win_bison.exe"
$flexExe  = Join-Path $bisonExeDir "win_flex.exe"
if (-not (Test-Path $bisonExe)) { $bisonExe = $null }
if (-not (Test-Path $flexExe))  { $flexExe  = $null }

# Set triplet for Windows x64 - use parameter value for triplet selection
$env:VCPKG_DEFAULT_TRIPLET = $VcpkgTriplet

# Use custom triplet that chainloads toolchain-nuget-msvc.cmake
# This forces vcpkg to use NuGet compiler while relying on INCLUDE/LIB env vars for paths
# Avoids /I flags in triplet which break MSYS2-based builds (ICU, etc.)
$customTripletPath = $PSScriptRoot
Write-Host "✓ Using custom vcpkg triplet from: $customTripletPath" -ForegroundColor Green

# Set vcpkg manifest directory (contains vcpkg.json and vcpkg-configuration.json)
$vcpkgManifestDir = Join-Path $PSScriptRoot "vcpkg"
Write-Host "✓ Using vcpkg manifest from: $vcpkgManifestDir" -ForegroundColor Green

# Verify Windows SDK NuGet packages exist
# ($nugetPackages already resolved above)
$windowsSdkPath = "$nugetPackages\Microsoft.Windows.Sdk.Cpp\$WindowsSDKPkgVersion"
$windowsSdkPlatformPath = "$nugetPackages\Microsoft.Windows.Sdk.Cpp.$sdkPkgPlatform\$WindowsSDKPkgVersion"

if (-not (Test-Path $windowsSdkPath)) {
    Write-Host "❌ Windows SDK $WindowsSDKPkgVersion not found at: $windowsSdkPath" -ForegroundColor Red
    Write-Host "   Please ensure Microsoft.Windows.Sdk.Cpp NuGet package is installed." -ForegroundColor Yellow
    exit 1
}
if (-not (Test-Path $windowsSdkPlatformPath)) {
    Write-Host "❌ Windows SDK $sdkPkgPlatform libs $WindowsSDKPkgVersion not found at: $windowsSdkPlatformPath" -ForegroundColor Red
    Write-Host "   Please ensure Microsoft.Windows.Sdk.Cpp.$sdkPkgPlatform NuGet package is installed." -ForegroundColor Yellow
    exit 1
}
Write-Host "✓ Found Windows SDK $WindowsSDKPkgVersion (NuGet)" -ForegroundColor Green

# Note: Production build requires these packages in vcpkg.json:
#   - azure-storage-blobs-cpp
#   - azure-storage-files-datalake-cpp  (for ABFS)
#   - azure-identity-cpp
#   - lzo  (provides lzo2.lib)
#   - All other dependencies auto-resolved via vcpkg manifest

# ============================================================================
# Build Configuration
# ============================================================================

Write-Host ""
Write-Host "Building Velox for Windows ($Architecture)" -ForegroundColor Cyan
Write-Host "  Build Type: $BuildType" -ForegroundColor White
Write-Host "  Build Dir: $BuildDir" -ForegroundColor White
Write-Host "  Architecture: $Architecture" -ForegroundColor White
Write-Host "  Triplet: $env:VCPKG_DEFAULT_TRIPLET" -ForegroundColor White
Write-Host "  Parallelism: /m:$Parallelism (MSBuild project-level) + /MP (cl.exe file-level)" -ForegroundColor White
Write-Host "  CPU Cores: $env:NUMBER_OF_PROCESSORS" -ForegroundColor White
Write-Host "  MSVC Toolset: $VCToolsVersion" -ForegroundColor White
if ($Minimal) {
    Write-Host "  Mode: Minimal (types, expressions, basic functions)" -ForegroundColor Yellow
} elseif ($WithTests) {
    Write-Host "  Mode: Production with tests" -ForegroundColor Yellow
} else {
    Write-Host "  Mode: Production (Arrow, ABFS, Parquet, Hive, Exec)" -ForegroundColor Yellow
}
Write-Host ""
Write-Host "NOTE: Using MSVC v143 toolset version $VCToolsVersion to match CosmosAnalytics build environment" -ForegroundColor Cyan
Write-Host "      This ensures ABI compatibility and prevents linker errors (LNK2001, LNK2019)" -ForegroundColor Cyan
Write-Host ""

# Set environment variables to force CMake/MSBuild to use the correct toolset
# NuGet package uses old directory structure (bin\amd64 for native, bin\amd64_arm64 for cross)
$ToolsetBinPath = "$vsToolsetPath\bin\$compilerBinDir"

# NOTE: Do NOT set CC/CXX environment variables!
# - Setting them causes MSYS2/make builds (ICU) to fail due to path mangling
# - The triplet sets PATH with NuGet toolset first, so cl.exe is found automatically
# - The triplet also sets MSBuild environment variables (VC_ExecutablePath_x64_x64, etc.)
#   so Visual Studio generator finds the correct compiler
$env:VCToolsInstallDir = "$vsToolsetPath\"

# For VS 2022 MSBuild (v17.x), VCToolsInstallDir is resolved as $(VCToolsInstallDir_170).
# The '_160' variant set by the triplet is for VS 2019 only and is ignored by VS 2022 MSBuild.
# Setting VCToolsInstallDir_170 here tells MSBuild to find the compiler in the NuGet toolset
# instead of 'C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Tools\MSVC\14.38.33133'
# (which doesn't exist - 14.38.33133 is ONLY available as a NuGet package on this machine).
$env:VCToolsInstallDir_170 = "$vsToolsetPath\"

# Force MSBuild to use NuGet toolset compiler (not VS-installed 14.44.x)
# VC_ExecutablePath_x64_x64 is prepended to PATH by MSBuild when invoking cl.exe tasks;
# setting it here as an env var takes precedence over .props file defaults.
# NOTE: Do NOT set VCToolsVersion here - MSBuild combines it with VCToolsInstallDir_170 to
#       validate the path exists; we satisfy that by setting VCToolsInstallDir_170 above.
$env:VC_ExecutablePath_x64_x64 = "$vsToolsetPath\bin\$compilerBinDir"

# Set INCLUDE and LIB paths to use ONLY NuGet toolset headers/libraries
# We use Windows SDK from NuGet (Microsoft.Windows.Sdk.Cpp) to match CosmosAnalytics
# Detect VS installation path dynamically for build tools (cmake, ninja, msbuild).
# The compiler and SDK come from NuGet packages, NOT from VS.
$vswhere = "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
if (Test-Path $vswhere) {
    $vsPath = & $vswhere -version "[17.0,19.0)" -latest -property installationPath 2>$null
}
if ([string]::IsNullOrEmpty($vsPath) -or -not (Test-Path $vsPath)) {
    # Fallback for local dev machines with Enterprise edition
    $vsPath = "C:\Program Files\Microsoft Visual Studio\2022\Enterprise"
}
Write-Host "✓ Using Visual Studio at: $vsPath" -ForegroundColor Green
$nugetInclude = "$vsToolsetPath\include"
# NOTE: NuGet toolset uses architecture-specific dirs for libraries
$nugetLib = "$vsToolsetPath\lib\$targetLibDir"
$nugetAtlMfcInclude = "$vsToolsetPath\atlmfc\include"
$nugetAtlMfcLib = "$vsToolsetPath\atlmfc\lib\$targetLibDir"

# Use Windows SDK from NuGet packages (matching CosmosAnalytics configuration)
$windowsSdkDir = "$nugetPackages\Microsoft.Windows.Sdk.Cpp\$WindowsSDKPkgVersion\c"
$windowsSdkLibDir = "$nugetPackages\Microsoft.Windows.Sdk.Cpp.$sdkPkgPlatform\$WindowsSDKPkgVersion\c"
$sdkInclude = "$windowsSdkDir\Include\$WindowsSDKVersion"
$sdkLib = "$windowsSdkLibDir"
# SDK bin tools (rc.exe, mt.exe) must always be x64-hosted, even when cross-compiling for ARM64
$windowsSdkBin = "$windowsSdkDir\bin\$WindowsSDKVersion\x64"

Write-Host "✓ Using Windows SDK from NuGet: $windowsSdkDir" -ForegroundColor Green

# Build INCLUDE path: NuGet toolset FIRST, then Windows SDK from NuGet
$env:INCLUDE = "$nugetInclude;$nugetAtlMfcInclude;$sdkInclude\ucrt;$sdkInclude\um;$sdkInclude\shared;$sdkInclude\winrt;$sdkInclude\cppwinrt"

# Build LIB path: NuGet toolset FIRST, then Windows SDK from NuGet
$env:LIB = "$nugetLib;$nugetAtlMfcLib;$sdkLib\ucrt\$sdkPlatform;$sdkLib\um\$sdkPlatform"

# Windows SDK MSBuild env vars.
# MSBuild computes the INCLUDE/LIB it passes to cl.exe from WindowsSdkDir and VCToolsInstallDir.
# When WindowsSdkDir is not set (and DisableRegistryUse=true), MSBuild's computed INCLUDE has
# empty SDK paths, which OVERRIDES our $env:INCLUDE.  That causes "Cannot open include file:
# crtdbg.h" etc. even though our INCLUDE is correct.  Setting WindowsSdkDir explicitly lets
# MSBuild compute correct SDK paths so its INCLUDE matches ours.
$env:WindowsSdkDir        = "$windowsSdkDir\"
$env:WindowsSdkDir_10     = "$windowsSdkDir\"
$env:WindowsSDKVersion    = "$WindowsSDKVersion\"
$env:WindowsSDKLibVersion = "$WindowsSDKVersion\"
$env:UCRTVersion          = $WindowsSDKVersion
$env:WindowsSdkBinPath    = "$windowsSdkDir\bin\"
$env:WindowsSdkVerBinPath = "$windowsSdkDir\bin\$WindowsSDKVersion\"
$env:WindowsSDK_ExecutablePath_x64 = "$windowsSdkDir\bin\$WindowsSDKVersion\x64"
$env:WindowsSDK_ExecutablePath_x86 = "$windowsSdkDir\bin\$WindowsSDKVersion\x86"
if ($isArm64) {
    $env:WindowsSDK_ExecutablePath_arm64 = "$windowsSdkDir\bin\$WindowsSDKVersion\arm64"
}
$env:WindowsLibPath       = "$windowsSdkDir\UnionMetadata\$WindowsSDKVersion"
$env:WindowsSdkLibDir     = $windowsSdkLibDir

# Add build tools to PATH
# - NuGet compiler (cl.exe) FIRST for compilation
# - VS build tools (nmake, msbuild, cmake, ninja) for build system
$vsBuildToolsPath = "$vsPath\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin"
$vsNinjaPath = "$vsPath\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja"
$msbuildPath = "$vsPath\MSBuild\Current\Bin\amd64"

# Windows SDK bin path from NuGet (for rc.exe, mt.exe, etc.)
if (Test-Path $windowsSdkBin) {
    Write-Host "✓ Found Windows SDK tools at: $windowsSdkBin" -ForegroundColor Green
} else {
    Write-Host "⚠ Warning: Windows SDK bin not found at: $windowsSdkBin" -ForegroundColor Yellow
    $windowsSdkBin = ""
}

# Find the VS MSVC tools version dynamically (for nmake.exe, lib.exe, etc.)
$vsToolsVersions = Get-ChildItem "$vsPath\VC\Tools\MSVC" -Directory | Sort-Object Name -Descending
if ($vsToolsVersions) {
    $latestVsToolsVersion = $vsToolsVersions[0].Name
    $vsMsvcToolsPath = "$vsPath\VC\Tools\MSVC\$latestVsToolsVersion\bin\Hostx64\$sdkPlatform"
    Write-Host "✓ Found VS build tools (nmake) at: $vsMsvcToolsPath" -ForegroundColor Green
} else {
    Write-Host "⚠ Warning: Could not find VS MSVC tools for nmake" -ForegroundColor Yellow
    $vsMsvcToolsPath = ""
}

$env:PATH = "$ToolsetBinPath;$vsMsvcToolsPath;$windowsSdkBin;$vsBuildToolsPath;$vsNinjaPath;$msbuildPath;$env:PATH"

# Set Visual Studio version for vcpkg detection (required for VS generator)
$env:VisualStudioVersion = "17.0"

# Prevent MSBuild from using registry to validate toolset location.
# vcpkg port builds set this via the triplet cmake file, but the MAIN cmake configure
# does not load the triplet, so we set it here.  Without this, MSBuild fails to
# validate the NuGet toolset (which is not registered in the Windows registry) and
# cmake reports "No CMAKE_C_COMPILER could be found".
$env:DisableRegistryUse = "true"
$env:CheckMSVCComponents = "false"

# Set VCPKG_ROOT for triplet files
$env:VCPKG_ROOT = $VcpkgRoot

# Tell vcpkg triplet which build type to produce (release or debug).
# The triplet reads VELOX_BUILD_TYPE and sets VCPKG_BUILD_TYPE accordingly,
# so vcpkg only builds the requested configuration (halves build time).
$env:VELOX_BUILD_TYPE = $BuildType.ToLower()

# Set VCPKG environment variables to ensure it uses the correct toolset
# NOTE: Do not set VCPKG_FORCE_SYSTEM_BINARIES - it prevents vcpkg from downloading needed tools (7zip, etc.)
Remove-Item env:VCPKG_FORCE_SYSTEM_BINARIES -ErrorAction SilentlyContinue
# NOTE: Do NOT include CC/CXX - the triplet sets PATH and MSBuild env vars instead
# This avoids MSYS2 path mangling issues (ICU) while still ensuring NuGet toolset is used
$env:VCPKG_KEEP_ENV_VARS = "VCToolsInstallDir;VCToolsInstallDir_170;VC_ExecutablePath_x64_x64;INCLUDE;LIB;PATH;VisualStudioVersion;VCPKG_ROOT;DisableRegistryUse;CheckMSVCComponents;WindowsSdkDir;WindowsSDKVersion;UCRTVersion;VELOX_BUILD_TYPE"

Write-Host "Setting compiler environment for toolset $VCToolsVersion" -ForegroundColor Cyan
Write-Host "  VCToolsInstallDir        = $env:VCToolsInstallDir" -ForegroundColor White
Write-Host "  VCToolsInstallDir_170    = $env:VCToolsInstallDir_170  (VS2022 MSBuild property - points to NuGet toolset)" -ForegroundColor White
Write-Host "  VC_ExecutablePath_x64_x64= $env:VC_ExecutablePath_x64_x64  (NuGet cl.exe for MSBuild, overrides VS default)" -ForegroundColor White
Write-Host "  Toolset BIN              = $ToolsetBinPath (first in PATH for cl.exe discovery)" -ForegroundColor White
Write-Host "  Windows SDK = $WindowsSDKPkgVersion (NuGet package)" -ForegroundColor White
Write-Host "  Windows SDK Version = $WindowsSDKVersion" -ForegroundColor White
Write-Host "  INCLUDE paths (NuGet VC++ first, then NuGet SDK):" -ForegroundColor White
Write-Host "    - $nugetInclude" -ForegroundColor DarkGray
Write-Host "    - $sdkInclude\ucrt" -ForegroundColor DarkGray
Write-Host "  LIB paths (NuGet VC++ first, then NuGet SDK):" -ForegroundColor White
Write-Host "    - $nugetLib" -ForegroundColor DarkGray
Write-Host "    - $sdkLib\ucrt\$sdkPlatform" -ForegroundColor DarkGray
Write-Host ""

# Clean build if requested
if ($CleanBuild -and (Test-Path $BuildDir)) {
    Write-Host "🧹 Cleaning existing build directory..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force $BuildDir
}

# Create build directory
if (-not (Test-Path $BuildDir)) {
    New-Item -ItemType Directory -Path $BuildDir | Out-Null
}

# Set OpenSSL root for vcpkg - use build directory for manifest mode
$opensslRoot = Join-Path (Resolve-Path $BuildDir) "vcpkg_installed\$VcpkgTriplet"
Write-Host "OpenSSL will be searched at: $opensslRoot (after vcpkg install)" -ForegroundColor Cyan

# Custom ports overlay path for Windows-specific ports (arrow, icu)
# These are used to build Velox on Windows with custom toolchain
$customPortsPath = Join-Path $vcpkgManifestDir "ports"
Write-Host "✓ Using custom vcpkg ports from: $customPortsPath" -ForegroundColor Green

# Build the NuGet include/lib paths with forward slashes for CMake
$nugetIncludeCMake = ($nugetInclude -replace '\\', '/')
$nugetAtlMfcIncludeCMake = ($nugetAtlMfcInclude -replace '\\', '/')
$nugetLibCMake = ($nugetLib -replace '\\', '/')
$nugetAtlMfcLibCMake = ($nugetAtlMfcLib -replace '\\', '/')

# Configure CMake options
# Use Visual Studio generator with a custom toolchain that forces NuGet include paths
# The toolchain file ensures our NuGet STL headers are used before VS defaults
$nugetToolchainFile = Join-Path $PSScriptRoot "toolchain-nuget-msvc.cmake"

$cmakeOptions = @(
    "-DCMAKE_TOOLCHAIN_FILE=$ToolchainFile",
    "-DVCPKG_TARGET_TRIPLET=$VcpkgTriplet",
    "-DVCPKG_HOST_TRIPLET=x64-windows-static-md",
    "-DVCPKG_OVERLAY_TRIPLETS=$customTripletPath",
    "-DVCPKG_OVERLAY_PORTS=$customPortsPath",
    "-DVCPKG_MANIFEST_DIR=$vcpkgManifestDir",
    "-DCMAKE_BUILD_TYPE=$BuildType",
    "-DCMAKE_CXX_STANDARD=20",
    "-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded`$<`$<CONFIG:Debug>:Debug>DLL",
    "-DVELOX_ENABLE_CCACHE=OFF",
    "-DVELOX_ENABLE_SPARK_FUNCTIONS=OFF",
    "-DBUILD_SHARED_LIBS=OFF",
    "-DVELOX_BUILD_SHARED=OFF",
    "-DVELOX_BUILD_STATIC=ON",
    "-DPROTOBUF_USE_DLLS=0",
    "-DCMAKE_INCLUDE_PATH=$nugetIncludeCMake;$nugetAtlMfcIncludeCMake",
    "-DCMAKE_LIBRARY_PATH=$nugetLibCMake;$nugetAtlMfcLibCMake",
    # Windows uses native threads, not pthreads - tell CMake to use Win32 threads
    "-DCMAKE_USE_WIN32_THREADS_INIT=ON",
    "-DTHREADS_PREFER_PTHREAD_FLAG=OFF"
)

# For ARM64 cross-compilation, use Ninja generator instead of Visual Studio
# The VS generator's MSBuild probe fails for ARM64 because VCTargetsPath.vcxproj
# doesn't support the ARM64 platform configuration. Ninja avoids this entirely.
if ($isArm64) {
    $cmakeOptions += "-G"
    $cmakeOptions += "Ninja"
    # Must set CMAKE_SYSTEM_NAME=Windows to trigger cross-compilation mode
    # Without this, cmake's project() resets CMAKE_SYSTEM_PROCESSOR to the host value
    $cmakeOptions += "-DCMAKE_SYSTEM_NAME=Windows"
    $cmakeOptions += "-DCMAKE_SYSTEM_PROCESSOR:STRING=ARM64"
    $cmakeOptions += "-DCMAKE_C_COMPILER=$ToolsetBinPath/cl.exe"
    $cmakeOptions += "-DCMAKE_CXX_COMPILER=$ToolsetBinPath/cl.exe"
    $cmakeOptions += "-DCMAKE_LINKER=$ToolsetBinPath/link.exe"
    $cmakeOptions += "-DCMAKE_AR=$ToolsetBinPath/lib.exe"
    # Use VS-bundled Ninja
    $ninjaPath = "$vsNinjaPath/ninja.exe"
    if (Test-Path $ninjaPath) {
        $cmakeOptions += "-DCMAKE_MAKE_PROGRAM=$ninjaPath"
    }
    # Disable flex/bison for ARM64 cross-compilation - use pre-generated parser files
    # The x64 flex tool generates code with x64-specific constructs that fail on ARM64
    $cmakeOptions += "-DCMAKE_DISABLE_FIND_PACKAGE_BISON=ON"
    $cmakeOptions += "-DCMAKE_DISABLE_FIND_PACKAGE_FLEX=ON"
    # Still need FlexLexer.h for pre-generated scanner files
    if ($flexExe) {
        $flexDir = Split-Path $flexExe
        $cmakeOptions += "-DFLEX_INCLUDE_DIRS=$flexDir"
    }
}

# Pass bison/flex executables explicitly so CMake FindBISON/FindFLEX can locate them
if ($bisonExe) { $cmakeOptions += "-DBISON_EXECUTABLE=$bisonExe" }
if ($flexExe)  { $cmakeOptions += "-DFLEX_EXECUTABLE=$flexExe" }

# Configure Debug and Release builds
# CRITICAL: Use /external:I to mark NuGet headers as system headers to avoid include order issues
# The Visual Studio generator ignores INCLUDE env var, so we must embed paths in compiler flags
$nugetIncludeFlag = "/I`"$nugetInclude`""
$nugetAtlMfcIncludeFlag = "/I`"$nugetAtlMfcInclude`""
# Add Windows SDK include paths for NuGet SDK
$sdkUcrtIncludeFlag = "/I`"$sdkInclude\ucrt`""
$sdkUmIncludeFlag = "/I`"$sdkInclude\um`""
$sdkSharedIncludeFlag = "/I`"$sdkInclude\shared`""
$sdkWinrtIncludeFlag = "/I`"$sdkInclude\winrt`""
$allIncludeFlags = "$nugetIncludeFlag $nugetAtlMfcIncludeFlag $sdkUcrtIncludeFlag $sdkUmIncludeFlag $sdkSharedIncludeFlag $sdkWinrtIncludeFlag"

# Strip absolute source paths from __FILE__ macros in compiled objects
# /d1trimfile:<prefix> makes paths relative (e.g., velox\type\Type.cpp instead of C:\Git\Velox\velox\type\Type.cpp)
$repoRoot = (Resolve-Path "$PSScriptRoot\..").Path
if (-not $repoRoot.EndsWith('\')) { $repoRoot += '\' }
$trimFileFlag = "/d1trimfile:$repoRoot"

if ($BuildType -eq "Debug") {
    # Debug builds: Use minimal optimization (/O1) to reduce code size
    # This allows MONO_LIBRARY to stay under 4GB limit
    # /O1 = optimize for size, /Oy- = keep frame pointers
    # /MDd = dynamic runtime library (debug) - for CosmosAnalytics compatibility
    # Debug build with minimal optimization and dynamic CRT for CosmosAnalytics
    # CRITICAL: Must define _DEBUG when using /MDd (debug CRT) to avoid runtime mismatches
    Write-Host "🚀 Configuring Debug production build (MONO_LIBRARY with size optimization)..." -ForegroundColor Yellow
    Write-Host "   Note: Using /O1 (size optimization) and /MDd (dynamic CRT) for CosmosAnalytics" -ForegroundColor Cyan
    if ($NoMono -or $isArm64) { $monoLib = "OFF" } else { $monoLib = "ON" }
    $cmakeOptions += "-DVELOX_MONO_LIBRARY=$monoLib"
    # /D_DEBUG = required for debug CRT consistency
    # /arch:AVX2 = CRITICAL: Enable AVX2 SIMD instructions for Velox performance and correctness
    # /DXSIMD_WITH_AVX512*=0 = CRITICAL: Explicitly disable AVX512 to prevent xsimd from using wider SIMD
    #                         MSVC may define __AVX512F__ even with /arch:AVX2, causing xsimd to use AVX512
    # /O1 = optimize for size (smallest code for template-heavy codebase)
    # /Gw = optimize global data packing (reduces data sections)
    # /Zc:inline = remove unreferenced inline COMDATs (significant for template-heavy code)
    # /D_ITERATOR_DEBUG_LEVEL=2 = force STL iterator debug level 2 to match vcpkg debug libs and
    #   VeloxRunner Debug builds. All linked objects must agree on the level to avoid LNK2038 and ABI crashes.
    # NOTE: Do NOT define NDEBUG in Debug builds - it creates ABI inconsistencies with consumers
    #   (VeloxRunner) that compile Debug without NDEBUG, causing ODR violations in inline/template code.
    # Include NuGet paths FIRST to ensure correct STL version
    # /MP = enable multi-process compilation (cl.exe compiles multiple files in parallel)
    # Without /MP, MONO_LIBRARY builds are single-threaded since MSBuild /m:N only parallelizes across projects
    # Architecture-specific SIMD flags
    # x64: /arch:AVX2 for SIMD performance, disable AVX512 to prevent xsimd width issues
    # ARM64: Define __aarch64__ so xsimd enables NEON64 backend (MSVC uses _M_ARM64 instead)
    # ARM64: xsimd 13.2.0's NEON backend is incompatible with MSVC ARM64.
    # Use emulated backend - configured in CMakeLists.txt via compile definitions.
    if ($isArm64) {
        $archFlags = ""
    } else {
        $archFlags = "/arch:AVX2"
    }
    $cmakeOptions += "-DCMAKE_CXX_FLAGS_DEBUG=/MP /MDd /O1 /D_DEBUG /D_ITERATOR_DEBUG_LEVEL=2 $archFlags /Gw /Zc:inline /wd4716 /bigobj $trimFileFlag $allIncludeFlags"
    $cmakeOptions += "-DCMAKE_C_FLAGS_DEBUG=/MP /MDd /O1 /D_DEBUG $archFlags /Gw /Zc:inline $trimFileFlag $allIncludeFlags"
} else {
    # Release builds with dynamic runtime linking
    # /MD = dynamic runtime library for CosmosAnalytics compatibility
    # /arch:AVX2 = CRITICAL: Enable AVX2 SIMD instructions for Velox performance and correctness
    #              Without this, xsimd will not define XSIMD_WITH_AVX2 and SIMD code paths won't compile
    # /DXSIMD_WITH_AVX512*=0 = CRITICAL: Explicitly disable AVX512 to prevent xsimd from using wider SIMD
    #                         MSVC may define __AVX512F__ even with /arch:AVX2, causing xsimd to use AVX512
    # This keeps velox.lib at ~2.7GB with dynamic CRT
    # Include NuGet paths FIRST to ensure correct STL version
    if ($NoMono -or $isArm64) { $monoLib = "OFF" } else { $monoLib = "ON" }
    $cmakeOptions += "-DVELOX_MONO_LIBRARY=$monoLib"
    # Architecture-specific SIMD flags
    # x64: /arch:AVX2 for SIMD performance, disable AVX512 to prevent xsimd width issues
    # ARM64: No /arch flag needed - NEON is enabled by default on ARM64 Windows
    #        Define __aarch64__ so xsimd enables NEON64 backend (MSVC uses _M_ARM64 instead)
    # ARM64: xsimd 13.2.0's NEON backend is incompatible with MSVC ARM64.
    # Use emulated backend - configured in CMakeLists.txt via compile definitions.
    if ($isArm64) {
        $archFlags = ""
    } else {
        $archFlags = "/arch:AVX2"
    }
    # ARM64 uses /O1 /Ob1 (optimize for size, limited inlining) to keep velox.lib under 4GB COFF limit
    if ($Architecture -eq "arm64") { $optLevel = "/O1"; $inlineLevel = "/Ob1" } else { $optLevel = "/O2"; $inlineLevel = "/Ob2" }
    $cppStdFlag = ""
    $cmakeOptions += "-DCMAKE_CXX_FLAGS_RELEASE=/MP /MD $optLevel $inlineLevel /DNDEBUG $archFlags $cppStdFlag /wd4716 $trimFileFlag $allIncludeFlags"
    $cmakeOptions += "-DCMAKE_C_FLAGS_RELEASE=/MP /MD /O2 /Ob2 /DNDEBUG $archFlags $trimFileFlag $allIncludeFlags"
    $cmakeOptions += "-DCMAKE_EXE_LINKER_FLAGS_RELEASE=/OPT:REF /OPT:ICF"
    $cmakeOptions += "-DCMAKE_SHARED_LINKER_FLAGS_RELEASE=/OPT:REF /OPT:ICF"
}

# Force Windows dependencies to use vcpkg (SYSTEM mode) instead of building from source
$cmakeOptions += "-DICU_SOURCE=SYSTEM"
$cmakeOptions += "-DProtobuf_SOURCE=SYSTEM"
$cmakeOptions += "-DBoost_SOURCE=SYSTEM"
$cmakeOptions += "-Dgflags_SOURCE=SYSTEM"
$cmakeOptions += "-Dglog_SOURCE=SYSTEM"
$cmakeOptions += "-Dfmt_SOURCE=SYSTEM"
$cmakeOptions += "-Dre2_SOURCE=SYSTEM"
$cmakeOptions += "-Dfolly_SOURCE=SYSTEM"
$cmakeOptions += "-DArrow_SOURCE=SYSTEM"
$cmakeOptions += "-Dsimdjson_SOURCE=SYSTEM"
# xsimd: use BUNDLED (FetchContent) 13.2.0 - the vcpkg version conflicts with Velox's API
# Force GTest to build from source (BUNDLED) instead of using system GTest.
# The OneBranch container has a Release-only GTest in Miniconda3 which causes
# LNK2038 mismatch (_ITERATOR_DEBUG_LEVEL, RuntimeLibrary) in Debug builds.
$cmakeOptions += "-DGTest_SOURCE=BUNDLED"

# Set protoc executable path explicitly for static build (always x64 host tool)
$protocPath = "$PSScriptRoot\..\build\vcpkg_installed\x64-windows-static\tools\protobuf\protoc.exe"
if (-not (Test-Path $protocPath)) {
    $protocPath = "$PSScriptRoot\..\build\vcpkg_installed\$VcpkgTriplet\tools\protobuf\protoc.exe"
}
if (Test-Path $protocPath) {
    $cmakeOptions += "-DProtobuf_PROTOC_EXECUTABLE=$protocPath"
}

# Add OpenSSL paths if available
if (Test-Path $opensslRoot) {
    $cmakeOptions += "-DOPENSSL_ROOT_DIR=$opensslRoot"
}

# Minimal build options
if ($Minimal) {
    Write-Host "📦 Configuring minimal build..." -ForegroundColor Yellow
    $cmakeOptions += "-DVELOX_BUILD_MINIMAL=ON"
    $cmakeOptions += "-DVELOX_BUILD_TESTING=OFF"
    $cmakeOptions += "-DVELOX_BUILD_TEST_UTILS=OFF"
} else {
    # ---------- Common feature flags for both production and test builds ----------
    # These flags are shared so that velox.lib has an identical feature surface
    # regardless of whether tests are enabled.  In particular VELOX_ENABLE_GEO=OFF
    # avoids pulling in GEOS symbols that are unavailable in CosmosAnalytic.
    if ($NoMono -or $isArm64) { $monoLib = "OFF" } else { $monoLib = "ON" }
    $cmakeOptions += "-DVELOX_MONO_LIBRARY=$monoLib"
    $cmakeOptions += "-DVELOX_BUILD_MINIMAL=OFF"
    $cmakeOptions += "-DVELOX_ENABLE_ARROW=ON"
    $cmakeOptions += "-DVELOX_ENABLE_ABFS=ON"
    $cmakeOptions += "-DVELOX_ENABLE_PARQUET=ON"
    $cmakeOptions += "-DVELOX_ENABLE_EXPRESSION=ON"
    $cmakeOptions += "-DVELOX_ENABLE_EXEC=ON"
    $cmakeOptions += "-DVELOX_ENABLE_AGGREGATES=ON"
    $cmakeOptions += "-DVELOX_ENABLE_HIVE_CONNECTOR=ON"
    $cmakeOptions += "-DVELOX_ENABLE_PRESTO_FUNCTIONS=ON"
    $cmakeOptions += "-DVELOX_ENABLE_SPARK_FUNCTIONS=OFF"       # Int128 compatibility issues on MSVC
    $cmakeOptions += "-DVELOX_ENABLE_ICEBERG_FUNCTIONS=ON"
    $cmakeOptions += "-DVELOX_ENABLE_S3=OFF"
    $cmakeOptions += "-DVELOX_ENABLE_GCS=OFF"
    $cmakeOptions += "-DVELOX_ENABLE_HDFS=OFF"
    $cmakeOptions += "-DVELOX_ENABLE_GEO=OFF"                   # GEOS unavailable in CosmosAnalytic
    $cmakeOptions += "-DVELOX_ENABLE_BENCHMARKS=OFF"
    $cmakeOptions += "-DVELOX_ENABLE_BENCHMARKS_BASIC=OFF"

    if ($WithTests) {
        # ---------- Test-specific overrides ----------
        Write-Host "🧪 Enabling production build with tests..." -ForegroundColor Yellow
        $cmakeOptions += "-DVELOX_BUILD_TESTING=ON"
        $cmakeOptions += "-DVELOX_ENABLE_TPCH_CONNECTOR=ON"
        $cmakeOptions += "-DVELOX_ENABLE_TPCDS_CONNECTOR=ON"
        # Ensure test executables can find MSVC runtime and Windows SDK system libs.
        # MSBuild overrides $env:LIB so we pass NuGet lib dirs via linker flags.
        # Also link dbghelp.lib (needed by glog symbolization on Windows).
        $msvcLib = "$vsToolsetPath\lib\$targetLibDir"
        $sdkUmLib = "$nugetPackages\Microsoft.Windows.Sdk.Cpp.$sdkPkgPlatform\$WindowsSDKPkgVersion\c\um\$sdkPlatform"
        $sdkUcrtLib = "$nugetPackages\Microsoft.Windows.Sdk.Cpp.$sdkPkgPlatform\$WindowsSDKPkgVersion\c\ucrt\$sdkPlatform"
        $linkerLibPaths = @()
        if (Test-Path $msvcLib) { $linkerLibPaths += "/LIBPATH:`"$msvcLib`"" }
        if (Test-Path $sdkUmLib) { $linkerLibPaths += "/LIBPATH:`"$sdkUmLib`"" }
        if (Test-Path $sdkUcrtLib) { $linkerLibPaths += "/LIBPATH:`"$sdkUcrtLib`"" }
        $linkerLibPaths += "dbghelp.lib"
        if ($linkerLibPaths.Count -gt 0) {
            $linkerFlags = $linkerLibPaths -join ' '
            $cmakeOptions += "-DCMAKE_EXE_LINKER_FLAGS=$linkerFlags"
            $cmakeOptions += "-DCMAKE_SHARED_LINKER_FLAGS=$linkerFlags"
            $cmakeOptions += "-DCMAKE_MODULE_LINKER_FLAGS=$linkerFlags"
        }
    } else {
        # ---------- Production-only overrides ----------
        Write-Host "🚀 Configuring Release production build (MONO_LIBRARY - Optimized)..." -ForegroundColor Yellow
        $cmakeOptions += "-DVELOX_BUILD_TESTING=OFF"
        $cmakeOptions += "-DVELOX_BUILD_TEST_UTILS=OFF"
        $cmakeOptions += "-DVELOX_ENABLE_TPCH_CONNECTOR=OFF"
        $cmakeOptions += "-DVELOX_ENABLE_TPCDS_CONNECTOR=OFF"
    }
}

# Configure
Write-Host "⚙️  Configuring CMake..." -ForegroundColor Cyan
Write-Host "Running: cmake -B $BuildDir $(($cmakeOptions -join ' '))" -ForegroundColor DarkGray

# Temporarily unset env vars that redirect MSBuild to the NuGet toolset.
# This is only needed for the Visual Studio generator (not Ninja).
# cmake's VS generator test-compiles a tiny program via MSBuild to detect the
# compiler.  MSBuild uses VCToolsInstallDir_170 as a BASE path and looks for
# cl.exe at <base>\bin\HostX64\x64\cl.exe.  The NuGet VisualCppTools package
# uses a different layout (bin\amd64\cl.exe), so MSBuild can't find the compiler
# and cmake reports "No CMAKE_C_COMPILER could be found".
# Also: setting WindowsSdkDir to the NuGet path causes MSBuild's _CheckWindowsSDKInstalled
# target (VCTargetsPath.vcxproj) to fail with MSB8037, because the NuGet SDK is not
# registered in the Windows registry.
# Solution: unset ALL NuGet-toolset env vars during cmake configure so MSBuild uses
# the system VS + registry SDK (which succeeds).  Restore them before cmake --build.
# For Ninja builds (ARM64), skip this - Ninja doesn't use MSBuild for detection.
if (-not $isArm64) {
$savedVCToolsInstallDir    = $env:VCToolsInstallDir
$savedVCToolsInstallDir170 = $env:VCToolsInstallDir_170
$savedVCInstallDir         = $env:VCInstallDir
$savedDisableRegistryUse   = $env:DisableRegistryUse
$savedCheckMSVC            = $env:CheckMSVCComponents
$savedWindowsSdkDir        = $env:WindowsSdkDir
$savedWindowsSdkDir10      = $env:WindowsSdkDir_10
$savedWindowsSDKVersion    = $env:WindowsSDKVersion
$savedWindowsSDKLibVersion = $env:WindowsSDKLibVersion
$savedUCRTVersion          = $env:UCRTVersion
$savedWindowsSdkBinPath    = $env:WindowsSdkBinPath
$savedWindowsSdkVerBinPath = $env:WindowsSdkVerBinPath
$savedWindowsSDK_x64       = $env:WindowsSDK_ExecutablePath_x64
$savedWindowsSDK_x86       = $env:WindowsSDK_ExecutablePath_x86
$savedWindowsLibPath       = $env:WindowsLibPath
$savedWindowsSdkLibDir     = $env:WindowsSdkLibDir
Remove-Item env:VCToolsInstallDir     -ErrorAction SilentlyContinue
Remove-Item env:VCToolsInstallDir_170 -ErrorAction SilentlyContinue
Remove-Item env:VCInstallDir          -ErrorAction SilentlyContinue
Remove-Item env:DisableRegistryUse    -ErrorAction SilentlyContinue
Remove-Item env:CheckMSVCComponents   -ErrorAction SilentlyContinue
Remove-Item env:WindowsSdkDir         -ErrorAction SilentlyContinue
Remove-Item env:WindowsSdkDir_10      -ErrorAction SilentlyContinue
Remove-Item env:WindowsSDKVersion     -ErrorAction SilentlyContinue
Remove-Item env:WindowsSDKLibVersion  -ErrorAction SilentlyContinue
Remove-Item env:UCRTVersion           -ErrorAction SilentlyContinue
Remove-Item env:WindowsSdkBinPath     -ErrorAction SilentlyContinue
Remove-Item env:WindowsSdkVerBinPath  -ErrorAction SilentlyContinue
Remove-Item env:WindowsSDK_ExecutablePath_x64 -ErrorAction SilentlyContinue
Remove-Item env:WindowsSDK_ExecutablePath_x86 -ErrorAction SilentlyContinue
Remove-Item env:WindowsLibPath        -ErrorAction SilentlyContinue
Remove-Item env:WindowsSdkLibDir      -ErrorAction SilentlyContinue
}

# Run CMake from project root
$cmakeArgs = @("-B", $BuildDir) + $cmakeOptions
& cmake @cmakeArgs
$cmakeExit = $LASTEXITCODE

# Restore NuGet toolset env vars for the MSBuild build step.
if (-not $isArm64) {
$env:VCToolsInstallDir     = $savedVCToolsInstallDir
$env:VCToolsInstallDir_170 = $savedVCToolsInstallDir170
$env:VCInstallDir          = $savedVCInstallDir
$env:DisableRegistryUse    = $savedDisableRegistryUse
$env:CheckMSVCComponents   = $savedCheckMSVC
$env:WindowsSdkDir         = $savedWindowsSdkDir
$env:WindowsSdkDir_10      = $savedWindowsSdkDir10
$env:WindowsSDKVersion     = $savedWindowsSDKVersion
$env:WindowsSDKLibVersion  = $savedWindowsSDKLibVersion
$env:UCRTVersion           = $savedUCRTVersion
$env:WindowsSdkBinPath     = $savedWindowsSdkBinPath
$env:WindowsSdkVerBinPath  = $savedWindowsSdkVerBinPath
$env:WindowsSDK_ExecutablePath_x64 = $savedWindowsSDK_x64
$env:WindowsSDK_ExecutablePath_x86 = $savedWindowsSDK_x86
$env:WindowsLibPath        = $savedWindowsLibPath
$env:WindowsSdkLibDir      = $savedWindowsSdkLibDir
}

if ($cmakeExit -ne 0) {
    Write-Host "❌ CMake configuration failed!" -ForegroundColor Red
    exit $cmakeExit
}

Write-Host "✓ Configuration complete" -ForegroundColor Green
Write-Host ""

# Fix LanguageStandard in generated vcxproj files
# CMake 3.31.6-msvc6 (Visual Studio bundled) generates stdcpp17 despite CMAKE_CXX_STANDARD=20
# This is a workaround to force C++20 in all generated vcxproj files
$vcxprojFiles = Get-ChildItem $BuildDir -Recurse -Filter "*.vcxproj" | Where-Object {
    $content = Get-Content $_.FullName -Raw
    $content -match '<LanguageStandard>stdcpp17</LanguageStandard>'
}
if ($vcxprojFiles) {
    Write-Host "⚙️  Fixing C++ standard in $($vcxprojFiles.Count) vcxproj files (stdcpp17 → stdcpp20)..." -ForegroundColor Yellow
    foreach ($file in $vcxprojFiles) {
        $content = Get-Content $file.FullName -Raw
        $content = $content -replace '<LanguageStandard>stdcpp17</LanguageStandard>', '<LanguageStandard>stdcpp20</LanguageStandard>'
        Set-Content $file.FullName $content -NoNewline
    }
    Write-Host "✓ Fixed C++ standard to C++20" -ForegroundColor Green
}

# Strip debug information from Debug builds to reduce .obj/.lib size
# CMake always sets DebugInformationFormat=ProgramDatabase (/Zi) for Debug configs,
# which embeds CodeView records in .obj files adding ~5-10% size overhead.
# User confirmed PDB files are not needed. This saves significant space in the mono library.
if ($BuildType -eq "Debug") {
    $vcxprojFilesDebugInfo = Get-ChildItem $BuildDir -Recurse -Filter "*.vcxproj" | Where-Object {
        $content = Get-Content $_.FullName -Raw
        $content -match '<DebugInformationFormat>ProgramDatabase</DebugInformationFormat>'
    }
    if ($vcxprojFilesDebugInfo) {
        Write-Host "⚙️  Stripping debug info from $($vcxprojFilesDebugInfo.Count) vcxproj files (Debug builds don't need PDB)..." -ForegroundColor Yellow
        foreach ($file in $vcxprojFilesDebugInfo) {
            $content = Get-Content $file.FullName -Raw
            $content = $content -replace '<DebugInformationFormat>ProgramDatabase</DebugInformationFormat>', '<DebugInformationFormat></DebugInformationFormat>'
            Set-Content $file.FullName $content -NoNewline
        }
        Write-Host "✓ Stripped debug information format from Debug builds" -ForegroundColor Green
    }
}

# Fix WinFlexBison generated Scanner.cpp missing commas (WinFlexBison 2.6.4 bug)
$scannerFile = Join-Path $BuildDir "velox\type\parser\Scanner.cpp"
if (Test-Path $scannerFile) {
    $lines = Get-Content $scannerFile
    $changed = $false
    for ($i = 0; $i -lt $lines.Count; $i++) {
        if ($lines[$i] -match '^\s+(-?\d+,?\s+)*-?\d+\s+-?\d+') {
            # Line has adjacent numbers without comma separator - fix it
            $lines[$i] = $lines[$i] -replace '(\d)\s{2,}(-)', '$1,  $2'
            $lines[$i] = $lines[$i] -replace '(\d)\s{2,}(\d)', '$1,  $2'
            $changed = $true
        }
    }
    if ($changed) {
        $lines | Set-Content $scannerFile
        Write-Host "Fixed WinFlexBison Scanner.cpp missing commas" -ForegroundColor Green
    }
}

# Fix: prevent parallel win_flex race condition
# win_flex 2.6.4 collides on %TEMP%\~1_flex_N files when run in parallel (MSBuild /m:N).
# Strategy:
#   1. Validate each Scanner.cpp was generated from its correct .ll source (detect corruption).
#   2. Regenerate any corrupted/missing Scanner.cpp files using win_flex directly (sequential).
#   3. Touch all Scanner.cpp timestamps so MSBuild sees them as newer than .ll sources
#      and skips the custom build rule entirely.
#   4. Delete stale .tlog files from any prior failed build that would force re-run of flex.
#
# Known race: parallel MSBuild (e.g. /m:12) can have multiple win_flex instances write to each
# other's output path via a shared %TEMP% file, corrupting Scanner.cpp with the wrong content.
Write-Host " Validating and ensuring flex-generated scanners are up-to-date..." -ForegroundColor Yellow

# Map of expected .ll source fragment → (build Scanner.cpp path, source .ll path, flex flags)
$sourceRoot = (Resolve-Path "$PSScriptRoot\..").Path
$scannerMap = @(
    @{ ExpectedFragment = "TypeCalculation.ll";     OutDir = "velox\expression\type_calculation";            LlFile = "$sourceRoot\velox\expression\type_calculation\TypeCalculation.ll";             FlexFlags = "--prefix=veloxtc" },
    @{ ExpectedFragment = "SignatureParser.ll";     OutDir = "velox\expression\signature_parser";            LlFile = "$sourceRoot\velox\expression\signature_parser\SignatureParser.ll";             FlexFlags = "--prefix=veloxsp" },
    @{ ExpectedFragment = "TypeParser.ll";          OutDir = "velox\functions\prestosql\types\parser";       LlFile = "$sourceRoot\velox\functions\prestosql\types\parser\TypeParser.ll";             FlexFlags = "--prefix=veloxprestotp" },
    @{ ExpectedFragment = "TypeParser.ll";          OutDir = "velox\type\parser";                            LlFile = "$sourceRoot\velox\type\parser\TypeParser.ll";                                  FlexFlags = "--prefix=veloxtpdeprecated" }
)

$generatedScanners = @()
foreach ($entry in $scannerMap) {
    $scannerPath = Join-Path $BuildDir "$($entry.OutDir)\Scanner.cpp"
    if (-not (Test-Path $scannerPath)) {
        Write-Host "  ⚠ Missing: $scannerPath — will generate" -ForegroundColor Yellow
    } else {
        # Check first few lines for correct .ll source reference
        $firstLines = Get-Content $scannerPath -TotalCount 5 -ErrorAction SilentlyContinue
        $correct = ($firstLines -join "`n") -match [regex]::Escape($entry.ExpectedFragment)
        if (-not $correct) {
            Write-Host "  ⚠ Corrupted: $scannerPath (wrong source) — regenerating from $($entry.ExpectedFragment)" -ForegroundColor Yellow
        } else {
            # File is correct — just collect for timestamp touch
            $generatedScanners += (Get-Item $scannerPath)
            continue
        }
    }
    # Regenerate using win_flex
    if ($flexExe -and (Test-Path $flexExe)) {
        $outDir = Split-Path $scannerPath -Parent
        if (-not (Test-Path $outDir)) { New-Item -ItemType Directory -Path $outDir | Out-Null }
        & $flexExe $entry.FlexFlags -o $scannerPath $entry.LlFile 2>&1 | Out-Null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  ✓ Regenerated: $scannerPath" -ForegroundColor Green
            $generatedScanners += (Get-Item $scannerPath)
        } else {
            Write-Host "  ❌ Failed to regenerate: $scannerPath (win_flex exit $LASTEXITCODE)" -ForegroundColor Red
        }
    } else {
        Write-Host "  ❌ win_flex not found — cannot regenerate $scannerPath" -ForegroundColor Red
    }
}

# Touch all valid Scanner.cpp timestamps to prevent MSBuild from re-running flex
$now = Get-Date
foreach ($sf in $generatedScanners) {
    $sf.LastWriteTime = $now
    Write-Host "  Touched: $($sf.FullName)" -ForegroundColor DarkGray
}

# Pre-generate bison outputs (.yy.cc/.yy.h) to prevent parallel race conditions.
# Same issue as flex: multiple win_bison instances run in parallel with /m:N and corrupt
# each other via shared m4 temp files, producing "undefined macro b4_symbol" errors.
# We run bison sequentially here, then neutralize the MSBuild custom build rules.
$bisonMap = @(
    @{ YyFile = "velox\expression\signature_parser\SignatureParser.yy";          OutDir = "velox\expression\signature_parser";            BaseName = "SignatureParser" },
    @{ YyFile = "velox\expression\type_calculation\TypeCalculation.yy";          OutDir = "velox\expression\type_calculation";            BaseName = "TypeCalculation" },
    @{ YyFile = "velox\functions\prestosql\types\parser\TypeParser.yy";          OutDir = "velox\functions\prestosql\types\parser";       BaseName = "TypeParser" },
    @{ YyFile = "velox\type\parser\TypeParser.yy";                               OutDir = "velox\type\parser";                            BaseName = "TypeParser" }
)
$bisonOutputCount = 0
foreach ($entry in $bisonMap) {
    $yySource = Join-Path $sourceRoot $entry.YyFile
    $outDir = Join-Path $BuildDir $entry.OutDir
    $outCc = Join-Path $outDir "$($entry.BaseName).yy.cc"
    $outH  = Join-Path $outDir "$($entry.BaseName).yy.h"
    if (-not (Test-Path $outDir)) { New-Item -ItemType Directory -Path $outDir -Force | Out-Null }
    $needRegen = (-not (Test-Path $outCc)) -or (-not (Test-Path $outH))
    if (-not $needRegen) {
        # Check if source is newer than output
        $srcTime = (Get-Item $yySource -ErrorAction SilentlyContinue).LastWriteTime
        $outTime = (Get-Item $outCc -ErrorAction SilentlyContinue).LastWriteTime
        if ($srcTime -gt $outTime) { $needRegen = $true }
    }
    if ($needRegen -and $bisonExe -and (Test-Path $bisonExe)) {
        $bisonResult = & $bisonExe -Wno-deprecated "--defines=$outH" -o $outCc $yySource 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  ✓ Generated bison: $outCc" -ForegroundColor Green
            $bisonOutputCount++
        } else {
            Write-Host "  ❌ Bison failed for $yySource (exit $LASTEXITCODE)" -ForegroundColor Red
            $bisonResult | ForEach-Object { Write-Host "    $_" -ForegroundColor DarkGray }
        }
    } elseif (-not $needRegen) {
        # Touch existing files to prevent MSBuild re-run
        if (Test-Path $outCc) { (Get-Item $outCc).LastWriteTime = $now }
        if (Test-Path $outH)  { (Get-Item $outH).LastWriteTime = $now }
        $bisonOutputCount++
    }
}

# Remove 'unsuccessfulbuild' marker files left by failed MSBuild custom builds.
# When MSBuild sees this marker, it unconditionally re-runs the custom build step
# regardless of timestamps or tlog state. Deleting the marker allows MSBuild to
# use the existing tlog to check whether inputs changed; since .ll/.yy sources
# haven't changed and outputs are now freshly touched, MSBuild will skip flex/bison.
# NOTE: Do NOT delete the full CustomBuild.*.tlog files - that removes all tracking
# and forces MSBuild to re-run every time (no-tlog = unconditional re-run).
$failedMarkers = @()
Get-ChildItem $BuildDir -Recurse -Filter "unsuccessfulbuild" -ErrorAction SilentlyContinue | Where-Object {
    $_.FullName -notmatch "\\vcpkg_installed\\"
} | ForEach-Object { $failedMarkers += $_ }
foreach ($marker in $failedMarkers) {
    Remove-Item $marker.FullName -Force -ErrorAction SilentlyContinue
    Write-Host "  Cleared failed-build marker: $($marker.DirectoryName)" -ForegroundColor DarkGray
}
if ($failedMarkers.Count -gt 0) {
    Write-Host "✓ Cleared $($failedMarkers.Count) 'unsuccessfulbuild' marker(s) - MSBuild will use tlog for incremental check" -ForegroundColor Green
}
if ($generatedScanners.Count -gt 0) {
    Write-Host "✓ Pre-generated $($generatedScanners.Count) flex scanner(s)" -ForegroundColor Green
}
if ($bisonOutputCount -gt 0) {
    Write-Host "✓ Pre-generated/validated $bisonOutputCount bison output(s)" -ForegroundColor Green
}

# Neutralize Flex and Bison generator vcxproj files to prevent MSBuild from re-running them.
# On fresh builds there are no .tlog files, so MSBuild ignores timestamps and ALWAYS re-runs
# the custom build steps. With /m:N, multiple win_flex/win_bison instances run in parallel and
# corrupt each other's output via shared temp files — causing build failures.
# The flex/bison rules live in separate *_gen_src.vcxproj projects (not velox.vcxproj).
# Since we pre-generate all Scanner.cpp and .yy.cc/.yy.h files above, we replace the
# commands with no-ops so MSBuild skips flex/bison entirely.
Write-Host "⚙️  Neutralizing Flex/Bison custom build rules in generator vcxproj files..." -ForegroundColor Yellow
$genSrcProjects = Get-ChildItem $BuildDir -Recurse -Filter "*_gen_src.vcxproj" -ErrorAction SilentlyContinue | Where-Object { $_.FullName -notmatch "vcpkg" }
$neutralizedCount = 0
foreach ($proj in $genSrcProjects) {
    $vcxContent = Get-Content $proj.FullName -Raw
    if ($vcxContent -match "win_flex|win_bison") {
        # Neutralize both flex (win_flex.exe) and bison (win_bison.exe) commands.
        # Both are pre-generated above. Must match .exe to avoid false positives from
        # the WinFlexBison directory path (which contains win_flex_bison as a substring).
        $vcxContent = [regex]::Replace($vcxContent,
            '(<Command[^>]*>)([^<]*)(</Command>)',
            { param($m)
                if ($m.Groups[2].Value -match 'win_flex\.exe') {
                    "$($m.Groups[1].Value)echo Skipped flex (pre-generated)$($m.Groups[3].Value)"
                } elseif ($m.Groups[2].Value -match 'win_bison\.exe') {
                    "$($m.Groups[1].Value)echo Skipped bison (pre-generated)$($m.Groups[3].Value)"
                } else {
                    $m.Value
                }
            })
        Set-Content $proj.FullName $vcxContent -NoNewline
        $neutralizedCount++
        Write-Host "  ✓ Neutralized flex/bison in: $($proj.Name)" -ForegroundColor Green
    }
}
if ($neutralizedCount -eq 0) {
    Write-Host "  No Flex/Bison generator projects found" -ForegroundColor DarkGray
} else {
    Write-Host "✓ Neutralized flex/bison commands in $neutralizedCount generator project(s)" -ForegroundColor Green
}

# Build
Write-Host "🔨 Building Velox..." -ForegroundColor Cyan
Write-Host "Running: cmake --build $BuildDir --config $BuildType --parallel $Parallelism" -ForegroundColor DarkGray
cmake --build $BuildDir --config $BuildType --parallel $Parallelism

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Build failed!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Troubleshooting tips:" -ForegroundColor Yellow
    Write-Host "  - Check the error messages above" -ForegroundColor White
    Write-Host "  - Try reducing parallelism: -Parallelism 1" -ForegroundColor White
    Write-Host "  - Try a clean build: -CleanBuild" -ForegroundColor White
    Write-Host "  - Check build log: $BuildDir\CMakeFiles\CMakeError.log" -ForegroundColor White
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "✓ Build complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Build output:" -ForegroundColor Cyan
Write-Host "  Libraries: $BuildDir\velox\**\$BuildType\*.lib" -ForegroundColor White
Write-Host "  Headers: velox\**\*.h" -ForegroundColor White
Write-Host ""

if ($Minimal) {
    Write-Host "Next steps for minimal build:" -ForegroundColor Yellow
    Write-Host "  1. Create NuGet package: .\build-nuget.ps1" -ForegroundColor White
} else {
    Write-Host "Next steps for production build:" -ForegroundColor Yellow
    Write-Host "  1. Create NuGet package: .\build-nuget.ps1" -ForegroundColor White
    Write-Host "  2. Package will include: Arrow, ABFS, Parquet, Hive connector" -ForegroundColor White
}
Write-Host ""
