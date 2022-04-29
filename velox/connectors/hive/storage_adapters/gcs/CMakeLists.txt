# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# for generated headers

add_library(velox_gcs GCSFileSystem.cpp GCSUtil.cpp)
#TODO script should set it
link_directories("/home/eman_copty_intel_com/data2/vcpkg/packages/google-cloud-cpp_x64-linux//lib")
include_directories("/home/eman_copty_intel_com/data2/google-cloud-cpp/cmake-out/vcpkg_installed/x64-linux/include/")

set(GCSSDK_INCLUDE_DIRS "/home/eman_copty_intel_com/data2/vcpkg/packages/google-cloud-cpp_x64-linux/include")
set(GCSSDK_LIBRARIES "google_cloud_cpp_common google_cloud_cpp_rest_internal google_cloud_cpp_storage")


target_include_directories(velox_gcs PUBLIC ${GCSSDK_INCLUDE_DIRS})
target_link_libraries(velox_gcs ${GCSSDK_LIBRARIES})
#GCSSDK_INCLUDE_DIRS
if(${VELOX_BUILD_TESTING})
  #add_subdirectory(tests) #TODO add tests
  #add_subdirectory(benchmark)
endif()
