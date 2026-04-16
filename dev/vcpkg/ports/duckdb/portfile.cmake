vcpkg_download_distfile(ARCHIVE
    URLS "https://github.com/duckdb/duckdb/archive/refs/tags/v0.8.1.tar.gz"
    FILENAME "v0.8.1.tar.gz"
    SHA512 06e480aeefdb73bdc1b3d977d5b6f56b40d8893ac4de8702ca54e47ee14a0fa2532e3f9b1aaa49000d6b0b82b93e46e9e2bbaa7a1a6761542f91443a903ddb5d
)

vcpkg_extract_source_archive_ex(
    OUT_SOURCE_PATH SOURCE_PATH
    ARCHIVE "${ARCHIVE}"
)

# Add Clang-specific warning suppression flag
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(DUCKDB_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-missing-template-arg-list-after-template-kw")
else()
    set(DUCKDB_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()

vcpkg_configure_cmake(
    SOURCE_PATH "${SOURCE_PATH}"
    PREFER_NINJA
    OPTIONS
        "-DCMAKE_CXX_FLAGS=${DUCKDB_CXX_FLAGS}"
        -DBUILD_UNITTESTS=OFF
        -DENABLE_SANITIZER=OFF
        -DENABLE_UBSAN=OFF
        -DBUILD_SHELL=OFF
        -DEXPORT_DLL_SYMBOLS=OFF
        -DCMAKE_BUILD_TYPE=Release
)
vcpkg_install_cmake(ADD_BIN_TO_PATH)

vcpkg_fixup_cmake_targets(CONFIG_PATH lib/cmake/DuckDB)

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/lib/cmake")
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/lib/cmake")

vcpkg_cmake_config_fixup(PACKAGE_NAME duckdb)

vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")
