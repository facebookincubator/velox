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
# Variables.

# The output library.
add_library(
  velox4j SHARED
  velox4j/init/Init.cpp
  velox4j/jni/JniCommon.cpp
  velox4j/jni/JniError.cpp
  velox4j/jni/JniLoader.cpp
  velox4j/jni/JniWrapper.cpp
  velox4j/jni/StaticJniWrapper.cpp
  velox4j/iterator/DownIterator.cpp
  velox4j/lifecycle/ObjectStore.cpp
  velox4j/arrow/Arrow.cpp
  velox4j/query/Query.cpp
  velox4j/query/QueryExecutor.cpp
  velox4j/connector/ExternalStream.cpp
  velox4j/memory/AllocationListener.cpp
  velox4j/memory/JavaAllocationListener.cpp
  velox4j/memory/MemoryManager.cpp
  velox4j/memory/ArrowMemoryPool.cpp
  velox4j/conf/Config.cpp
  velox4j/init/Config.cpp
  velox4j/eval/Evaluation.cpp
  velox4j/eval/Evaluator.cpp
  velox4j/iterator/BlockingQueue.cpp)
target_include_directories(
  velox4j PUBLIC ${CMAKE_CURRENT_LIST_DIR} ${velox_SOURCE_DIR}
                 ${JniHelpersLib_SOURCE_DIR})
target_link_libraries(
  velox4j
  PUBLIC velox JniHelpers JNI::JNI)
set_target_properties(
  velox4j
  PROPERTIES
    OUTPUT_NAME velox4j
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib
    ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)

set(VELOX4J_3RD_EXCLUSIONS
    "libc\\\\.so\\\\.[0-9]+"
    "libstdc\\\\+\\\\+\\\\.so\\\\.[0-9]+"
    "libgcc_s\\\\.so\\\\.[0-9]+"
    "libm\\\\.so\\\\.[0-9]+"
    "linux-vdso\\\\.so\\\\.[0-9]+"
    "ld-linux-x86-64\\\\.so\\\\.[0-9]+"
    "libdl\\\\.so\\\\.[0-9]+"
    "libpthread\\\\.so\\\\.[0-9]+")

install(
  CODE "
    file(GET_RUNTIME_DEPENDENCIES
            RESOLVED_DEPENDENCIES_VAR 3rd_deps
            LIBRARIES $<TARGET_FILE:velox4j>
            PRE_EXCLUDE_REGEXES
            ${VELOX4J_3RD_EXCLUSIONS}
    )
    message(\"Found all shared 3rd dependencies of velox4j: \${3rd_deps}\")
    foreach(dep IN LISTS 3rd_deps)
        get_filename_component(link_name \${dep} NAME)
        get_filename_component(link_target \${dep} REALPATH)
        message(\"Installing 3rd dependency of velox4j: \${link_name}->\${link_target} to destination: ${VELOX4J_INSTALL_DESTINATION}...\")
        file(INSTALL DESTINATION ${VELOX4J_INSTALL_DESTINATION} TYPE FILE FILES \${link_target} RENAME \${link_name})
    endforeach()
    message(\"Installing main velox4j library...\")
    file(INSTALL DESTINATION ${VELOX4J_INSTALL_DESTINATION} TYPE FILE FILES $<TARGET_FILE:velox4j>)
"
  COMPONENT velox4j)
