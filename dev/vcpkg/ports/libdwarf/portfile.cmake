vcpkg_download_distfile(ARCHIVE
    URLS
        "https://github.com/davea42/libdwarf-code/archive/refs/tags/20210528.tar.gz"
    FILENAME "libdwarf-20210528.tar.xz"
    SHA512 99f39e34d4ad9a658a4c181a3e0211b4362bebe758b81297426c37b262b8480619da03e2db2472610febe8da67edf6636e04f77632792534d05d3e1edd4c89a5
)

vcpkg_extract_source_archive_ex(
    OUT_SOURCE_PATH SOURCE_PATH
    ARCHIVE "${ARCHIVE}"
)

vcpkg_configure_make(SOURCE_PATH ${SOURCE_PATH} AUTOCONFIG)
vcpkg_install_make()

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/share")
vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/COPYING")
vcpkg_fixup_pkgconfig()