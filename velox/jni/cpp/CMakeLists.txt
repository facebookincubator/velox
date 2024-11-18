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


find_package(JNI REQUIRED)

if (VELOX_ENABLE_JNI_BINDINGS_REF_DEBUG)
    add_definitions(-DVELOX_ENABLE_JNI_BINDINGS_REF_DEBUG)
endif ()

include_directories(${JNI_INCLUDE_DIRS})

velox_add_library(jni SHARED
        jni/VeloxJniWrapper.cc
        jni/JniErrors.cc
        jni/JniUtil.cc
        native/NativePlanBuilder.cc
        native/NativeClass.cc
        memory/MemoryManager.cpp
        memory/NativeMemoryManger.cc
        common/Global.cc
        functions/RegistrationAllFunctions.cc
)

velox_link_libraries(jni 
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
        velox_exec_test_lib)


