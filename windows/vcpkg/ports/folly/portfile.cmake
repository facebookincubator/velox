if(VCPKG_TARGET_IS_WINDOWS)
    vcpkg_check_linkage(ONLY_STATIC_LIBRARY)
endif()

vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO facebook/folly
    REF "v${VERSION}"
    SHA512 6d377c48cf1c0796da6fad34b930e9608f3cd765a675414eaad45ff46e9d0b9bb5f027b187ec135e88bb60a83cb91c07d266a6673621caf3f9961942b55276e2
    HEAD_REF main
    PATCHES
        fix-deps.patch
        disable-uninitialized-resize-on-new-stl.patch
        fix-unistd-include.patch
        fix-absolute-dir.patch
        fix-arm64-builtins.patch
)

# Fix for multi-config generators: wrap pkg-config generation in conditional
# This prevents "Evaluation file to be written multiple times" error with Visual Studio
vcpkg_replace_string("${SOURCE_PATH}/CMakeLists.txt"
[[# Generate a pkg-config file so that downstream projects that don't use
# CMake can depend on folly using pkg-config.
configure_file(]]
[[# Skip pkg-config for multi-config generators (causes file(GENERATE) conflict)
get_property(is_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(NOT is_multi_config)
# Generate a pkg-config file so that downstream projects that don't use
# CMake can depend on folly using pkg-config.
configure_file(]]
)

vcpkg_replace_string("${SOURCE_PATH}/CMakeLists.txt"
[[install(
  FILES ${CMAKE_CURRENT_BINARY_DIR}/libfolly.pc
  DESTINATION ${LIB_INSTALL_DIR}/pkgconfig
  COMPONENT dev
)

option(BUILD_TESTS]]
[[install(
    FILES ${CMAKE_CURRENT_BINARY_DIR}/libfolly.pc
    DESTINATION ${LIB_INSTALL_DIR}/pkgconfig
    COMPONENT dev
  )
endif()

option(BUILD_TESTS]]
)

file(REMOVE "${SOURCE_PATH}/CMake/FindFastFloat.cmake")
file(REMOVE "${SOURCE_PATH}/CMake/FindFmt.cmake")
file(REMOVE "${SOURCE_PATH}/CMake/FindLibsodium.cmake")
file(REMOVE "${SOURCE_PATH}/CMake/FindZstd.cmake")
file(REMOVE "${SOURCE_PATH}/CMake/FindSnappy.cmake")
file(REMOVE "${SOURCE_PATH}/CMake/FindLZ4.cmake")
file(REMOVE "${SOURCE_PATH}/build/fbcode_builder/CMake/FindDoubleConversion.cmake")
file(REMOVE "${SOURCE_PATH}/build/fbcode_builder/CMake/FindGMock.cmake")
file(REMOVE "${SOURCE_PATH}/build/fbcode_builder/CMake/FindGflags.cmake")
file(REMOVE "${SOURCE_PATH}/build/fbcode_builder/CMake/FindGlog.cmake")
file(REMOVE "${SOURCE_PATH}/build/fbcode_builder/CMake/FindLibEvent.cmake")
file(REMOVE "${SOURCE_PATH}/build/fbcode_builder/CMake/FindSodium.cmake")
file(REMOVE "${SOURCE_PATH}/build/fbcode_builder/CMake/FindZstd.cmake")

string(COMPARE EQUAL "${VCPKG_CRT_LINKAGE}" "static" MSVC_USE_STATIC_RUNTIME)

vcpkg_check_features(OUT_FEATURE_OPTIONS FEATURE_OPTIONS
    FEATURES
        "bzip2"      VCPKG_LOCK_FIND_PACKAGE_BZip2
        "libaio"     VCPKG_LOCK_FIND_PACKAGE_LibAIO
        "libsodium"  VCPKG_LOCK_FIND_PACKAGE_LIBSODIUM
        "liburing"   VCPKG_LOCK_FIND_PACKAGE_LibUring
        "lz4"        VCPKG_LOCK_FIND_PACKAGE_LZ4
        "snappy"     VCPKG_LOCK_FIND_PACKAGE_SNAPPY
        "zstd"       VCPKG_LOCK_FIND_PACKAGE_ZSTD
)

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    DISABLE_PARALLEL_CONFIGURE
    OPTIONS
        -DMSVC_USE_STATIC_RUNTIME=${MSVC_USE_STATIC_RUNTIME}
        -DCMAKE_INSTALL_DIR=share/folly
        -DCMAKE_POLICY_DEFAULT_CMP0167=NEW
        -DVCPKG_LOCK_FIND_PACKAGE_fmt=ON
        -DVCPKG_LOCK_FIND_PACKAGE_LibDwarf=OFF
        -DVCPKG_LOCK_FIND_PACKAGE_Libiberty=OFF
        -DVCPKG_LOCK_FIND_PACKAGE_LibUnwind=${VCPKG_TARGET_IS_LINUX}
        -DVCPKG_LOCK_FIND_PACKAGE_ZLIB=ON
        # Pre-set try_run() results for cross-compilation (ARM64 on x64 host)
        -DFOLLY_HAVE_UNALIGNED_ACCESS_EXITCODE=0
        -DFOLLY_HAVE_UNALIGNED_ACCESS_EXITCODE__TRYRUN_OUTPUT=""
        -DFOLLY_HAVE_WCHAR_SUPPORT_EXITCODE=0
        -DFOLLY_HAVE_WCHAR_SUPPORT_EXITCODE__TRYRUN_OUTPUT=""
        -DHAVE_VSNPRINTF_ERRORS_EXITCODE=0
        -DHAVE_VSNPRINTF_ERRORS_EXITCODE__TRYRUN_OUTPUT=""
        ${FEATURE_OPTIONS}
    MAYBE_UNUSED_VARIABLES
        MSVC_USE_STATIC_RUNTIME
        FOLLY_HAVE_UNALIGNED_ACCESS_EXITCODE__TRYRUN_OUTPUT
        FOLLY_HAVE_WCHAR_SUPPORT_EXITCODE__TRYRUN_OUTPUT
        HAVE_VSNPRINTF_ERRORS_EXITCODE__TRYRUN_OUTPUT
)

vcpkg_cmake_install()
vcpkg_copy_pdbs()
vcpkg_fixup_pkgconfig()
vcpkg_cmake_config_fixup()

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")

vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")
