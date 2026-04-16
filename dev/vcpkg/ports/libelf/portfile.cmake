vcpkg_download_distfile(
  ARCHIVE
  URLS
  "https://mirror.clarkson.edu/gentoo/distfiles/80/libelf-0.8.13.tar.gz"
  "https://mirror.dotsrc.org/mirrors/pub/gsb/gsb64-2.30_slackware64-13.1/source/l/libelf/libelf-0.8.13.tar.gz"
  "https://fossies.org/linux/misc/old/libelf-0.8.13.tar.gz"
  FILENAME
  "libelf-0.8.13.tar.gz"
  SHA512
  d2a4ea8ccc0bbfecac38fa20fbd96aefa8e86f8af38691fb6991cd9c5a03f587475ecc2365fc89a4954c11a679d93460ee9a5890693112f6133719af3e6582fe
)

vcpkg_extract_source_archive(SOURCE_PATH ARCHIVE "${ARCHIVE}" PATCHES
                             install.patch)

# Update config.guess and config.sub
file(COPY ${CURRENT_PORT_DIR}/config.guess DESTINATION ${SOURCE_PATH})
file(COPY ${CURRENT_PORT_DIR}/config.sub DESTINATION ${SOURCE_PATH})

vcpkg_configure_make(SOURCE_PATH ${SOURCE_PATH} AUTOCONFIG)
vcpkg_install_make()
vcpkg_fixup_pkgconfig()

# file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/share") file(REMOVE_RECURSE
# "${CURRENT_PACKAGES_DIR}/share") file(REMOVE_RECURSE
# "${CURRENT_PACKAGES_DIR}/tools")

vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/COPYING.LIB")
