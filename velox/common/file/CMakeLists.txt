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
include_directories(.)
add_library(file File.cpp File.h S3File.h)
target_link_libraries(file velox_exception ${GTEST_BOTH_LIBRARIES} ${GLOG}
                      ${FOLLY_WITH_DEPENDENCIES} ${FMT})
target_include_directories(file PRIVATE ${AWSSDK_INCLUDE_DIRS})

add_executable(file_test FileTest.cpp)
add_test(file_test file_test)
target_link_libraries(file_test file)
if (VELOX_ENABLE_S3)
    add_dependencies(file aws-sdk)
    target_link_libraries(file aws-sdk-s3 aws-sdk-core)
    add_executable(s3file_test S3FileTest.cpp)
    add_test(s3file_test s3file_test)
    target_link_libraries(s3file_test file velox_core)
    target_include_directories(s3file_test PRIVATE ${AWSSDK_INCLUDE_DIRS})
endif()