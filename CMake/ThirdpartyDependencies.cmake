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

include(ExternalProject)
macro(build_awssdk)
    message("Configured to download and build AWS-SDK-CPP version " ${AWS_SDK_VERSION})
    externalproject_add(aws-sdk
            PREFIX aws-sdk-cpp
            GIT_REPOSITORY "https://github.com/aws/aws-sdk-cpp.git"
            GIT_TAG ${AWS_SDK_VERSION}
            SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/aws-sdk-cpp/src/aws-sdk-cpp"
            CMAKE_ARGS
                        -DBUILD_ONLY:STRING=s3
                        -DBUILD_SHARED_LIBS:BOOL=OFF
                        -DMINIMIZE_SIZE:BOOL=ON
                        -DENABLE_TESTING:BOOL=OFF
                        -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
                        -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_CURRENT_BINARY_DIR}/aws-sdk-cpp/install
            BUILD_ALWAYS      TRUE
            TEST_COMMAND      ""
            )
    ExternalProject_Get_Property(aws-sdk INSTALL_DIR)
    add_library(aws-sdk-core STATIC IMPORTED)
    add_library(aws-sdk-s3 STATIC IMPORTED)
    set_target_properties(aws-sdk-core PROPERTIES IMPORTED_LOCATION ${INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}aws-cpp-sdk-core${CMAKE_STATIC_LIBRARY_SUFFIX})
    set_target_properties(aws-sdk-s3   PROPERTIES IMPORTED_LOCATION ${INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}aws-cpp-sdk-s3${CMAKE_STATIC_LIBRARY_SUFFIX})
    set(AWS_INCLUDE_DIRS ${INSTALL_DIR}/include)
    set(AWS_LIBS         aws-sdk-s3 aws-sdk-core)
    include_directories(SYSTEM ${AWS_INCLUDE_DIRS})
endmacro()

if (VELOX_ENABLE_S3)
    # S3 Reference  https://aws.amazon.com/blogs/developer/developer-experience-of-the-aws-sdk-for-c-now-simplified-by-cmake/
    # AWS S3 SDK provides the needed AWSSDKConfig.cmake file
    # Quietly check if AWS SDK is already installed
    if (NOT DEFINED AWS_INSTALL_DIR AND EXISTS ${CMAKE_CURRENT_BINARY_DIR}/aws-sdk-cpp/install)
        set(AWS_INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}/aws-sdk-cpp/install)
    endif()

    if (DEFINED AWS_INSTALL_DIR)
        set(CMAKE_PREFIX_PATH "${AWS_INSTALL_DIR}/lib/cmake/AWSSDK" ${CMAKE_PREFIX_PATH})
        set(CMAKE_PREFIX_PATH "${AWS_INSTALL_DIR}/lib/cmake" ${CMAKE_PREFIX_PATH})
        set(CMAKE_PREFIX_PATH "${AWS_INSTALL_DIR}" ${CMAKE_PREFIX_PATH})
    endif()
    find_package(AWSSDK QUIET COMPONENTS s3)
    if(NOT AWSSDK_FOUND)
        build_awssdk()
    endif()
endif()


