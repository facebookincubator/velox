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

# This file either finds an exisiting installation of AWS SDK CPP or builds one.
# Set AWS_SDK_CPP_INSTALL_DIR if you have a custom install location of AWS SDK CPP.

macro(build_awssdk)
    message("Configured to download and build AWS-SDK-CPP version " ${AWS_SDK_VERSION})
    ExternalProject_Add(aws-sdk
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
                        -DCMAKE_INSTALL_PREFIX:PATH=${VELOX_DEPENDENCY_INSTALL_DIR}
            BUILD_ALWAYS      TRUE
            TEST_COMMAND      ""
            )
    ExternalProject_Get_Property(aws-sdk INSTALL_DIR)
    add_library(aws-sdk-core STATIC IMPORTED)
    add_library(aws-sdk-s3 STATIC IMPORTED)
    set_target_properties(aws-sdk-core PROPERTIES IMPORTED_LOCATION
            ${INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}aws-cpp-sdk-core${CMAKE_STATIC_LIBRARY_SUFFIX})
    set_target_properties(aws-sdk-s3   PROPERTIES IMPORTED_LOCATION
            ${INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}aws-cpp-sdk-s3${CMAKE_STATIC_LIBRARY_SUFFIX})
    target_include_directories(aws-sdk-s3 PUBLIC ${AWS_INCLUDE_DIRS})
endmacro()

# S3 Reference  https://aws.amazon.com/blogs/developer/developer-experience-of-the-aws-sdk-for-c-now-simplified-by-cmake/
# AWS S3 SDK provides the needed AWSSDKConfig.cmake file
# If the user did not provide an installation path, default to Velox dependency build directory
if (NOT DEFINED AWS_SDK_CPP_INSTALL_DIR AND EXISTS ${VELOX_DEPENDENCY_INSTALL_DIR})
    set(AWS_SDK_CPP_INSTALL_DIR ${VELOX_DEPENDENCY_INSTALL_DIR})
endif ()

if (DEFINED AWS_SDK_CPP_INSTALL_DIR)
    set(CMAKE_PREFIX_PATH "${AWS_SDK_CPP_INSTALL_DIR}/lib/cmake/AWSSDK" ${CMAKE_PREFIX_PATH})
    set(CMAKE_PREFIX_PATH "${AWS_SDK_CPP_INSTALL_DIR}/lib/cmake" ${CMAKE_PREFIX_PATH})
    set(CMAKE_PREFIX_PATH "${AWS_SDK_CPP_INSTALL_DIR}" ${CMAKE_PREFIX_PATH})
endif ()
# Quietly check if AWS SDK is already installed
find_package(AWSSDK QUIET COMPONENTS s3)
if (NOT AWSSDK_FOUND)
    build_awssdk()
endif ()