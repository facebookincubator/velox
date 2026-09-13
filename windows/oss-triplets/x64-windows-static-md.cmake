set(VCPKG_TARGET_ARCHITECTURE x64)
set(VCPKG_CRT_LINKAGE dynamic)
set(VCPKG_LIBRARY_LINKAGE static)
set(VCPKG_PLATFORM_TOOLSET v143)
# The OSS Windows build is Release-only, so build the vcpkg dependencies
# release-only by default (roughly halves dependency build time). Set
# VELOX_BUILD_TYPE=debug to also build debug variants. Release-only is the
# robust default because vcpkg does not forward arbitrary environment
# variables to triplet evaluation in every setup.
if(NOT "$ENV{VELOX_BUILD_TYPE}" STREQUAL "debug")
    set(VCPKG_BUILD_TYPE release)
endif()
