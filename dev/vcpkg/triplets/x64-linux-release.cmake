set(VCPKG_TARGET_ARCHITECTURE x64)
set(VCPKG_CRT_LINKAGE dynamic)
set(VCPKG_LIBRARY_LINKAGE static)

set(VCPKG_CMAKE_SYSTEM_NAME Linux)

set(VCPKG_BUILD_TYPE release)
set(VCPKG_POLICY_ONLY_RELEASE_CRT enabled)

set(VCPKG_C_FLAGS "")
set(VCPKG_CXX_FLAGS "-std=c++20")

# Build glog and gflags as shared libraries to avoid dual flag registration
# when .so plugins are dlopen'd
if(PORT MATCHES "glog")
    set(VCPKG_LIBRARY_LINKAGE dynamic)
endif()

if(PORT MATCHES "gflags")
    set(VCPKG_LIBRARY_LINKAGE dynamic)
endif()
