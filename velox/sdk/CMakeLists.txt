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
find_library(RE2 re2 REQUIRED)
#set(ENV{JAVA_HOME} "/Users/xuyiming/Library/Java/JavaVirtualMachines/openjdk-18.0.2/Contents/Home")
#set(ENV{JAVA_HOME} "/Library/Java/JavaVirtualMachines/jdk1.8.0_361.jdk/Contents/Home")

include(${PROJECT_SOURCE_DIR}/CMake/kudu_cmake_fns.txt)


# Adds a third-party library with name ${NAME}. If BUILD_SHARED_LIBS is true, the new
# library refers to ${SHARED_LIB}; otherwise it refers to ${STATIC_LIB}. If only one
# library (static or shared) is provided, it is used regardless of BUILD_SHARED_LIBS. The
# library's headers are added to the system include path.
function(IMPALA_ADD_THIRDPARTY_LIB NAME HEADER STATIC_LIB SHARED_LIB)
    message(STATUS "----------> Adding thirdparty library ${NAME}. <----------")
    if (HEADER)
        include_directories(SYSTEM ${HEADER})
        message(STATUS "Header files: ${HEADER}")
    endif ()
    if (NOT STATIC_LIB AND NOT SHARED_LIB)
        message(FATAL_ERROR "Library '${NAME}' has neither shared nor static library files")
        return()
    endif ()

    if ((BUILD_SHARED_LIBS AND SHARED_LIB) OR NOT STATIC_LIB)
        ADD_THIRDPARTY_LIB(${NAME} SHARED_LIB ${SHARED_LIB})
    else ()
        ADD_THIRDPARTY_LIB(${NAME} STATIC_LIB ${STATIC_LIB})
    endif ()
endfunction()

find_package(JNI REQUIRED)

if (VELOX_ENABLE_SDK_DEBUG)
    add_definitions(-DVELOX_ENABLE_SDK_DEBUG)
endif ()

IMPALA_ADD_THIRDPARTY_LIB(java_jvm "${JNI_INCLUDE_DIRS}" "" ${JAVA_JVM_LIBRARY})
include_directories(${JNI_INCLUDE_DIRS})

set(SDK_INCLUDE_LIBRARIES
        velox_type
        velox_vector
        velox_exec
        velox_exec_test_lib
        velox_parse_expression
        velox_hive_connector
        velox_memory
        velox_dwio_parquet_reader
        velox_functions_prestosql_impl
        velox_functions_spark
        velox_aggregates
        velox_functions_spark_aggregates
        velox_window
        velox_vector
        velox_functions_spark_window
        velox_exec_test_lib
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations -Wno-attributes")
# Build Velox backend.
set(SDK_SRCS
        cpp/jni/VeloxJniWrapper.cc
        cpp/jni/JniErrors.cc
        cpp/jni/JniUtil.cc
        cpp/native/NativePlanBuilder.cc
        cpp/native/NativeClass.cc
        cpp/memory/MemoryManager.cpp
        cpp/memory/NativeMemoryManger.cc
        cpp/common/Global.cc
        cpp/funcitons/RegistrationAllFunctions.cc
        
)

add_library(sdk SHARED ${SDK_SRCS})
target_link_libraries(sdk ${SDK_INCLUDE_LIBRARIES})