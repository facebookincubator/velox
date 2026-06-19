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

# On Windows with MSVC and vcpkg, use Arrow's CMake config
if(MSVC)
  # Avoid recursive calls
  if(NOT _VELOX_FINDING_ARROW)
    set(_VELOX_FINDING_ARROW TRUE)
    
    find_package(Arrow CONFIG QUIET)
    if(Arrow_FOUND)
      find_package(Parquet CONFIG QUIET)
      find_package(Thrift CONFIG QUIET)
      
      # Create aliases for compatibility
      if(NOT TARGET arrow)
        if(TARGET Arrow::arrow_shared)
          add_library(arrow ALIAS Arrow::arrow_shared)
        elseif(TARGET Arrow::arrow_static)
          add_library(arrow ALIAS Arrow::arrow_static)
        endif()
      endif()
      
      if(NOT TARGET parquet)
        if(TARGET Parquet::parquet_shared)
          add_library(parquet ALIAS Parquet::parquet_shared)
        elseif(TARGET Parquet::parquet_static)
          add_library(parquet ALIAS Parquet::parquet_static)
        endif()
      endif()
      
      if(NOT TARGET arrow_testing)
        # Arrow testing library may not be available in vcpkg build
        add_library(arrow_testing INTERFACE)
      endif()
      
      if(NOT TARGET thrift AND TARGET thrift::thrift)
        add_library(thrift ALIAS thrift::thrift)
      endif()
      
      unset(_VELOX_FINDING_ARROW)
      return()
    endif()
    
    unset(_VELOX_FINDING_ARROW)
  else()
    # We're being called recursively, just return
    return()
  endif()
endif()

include(FindPackageHandleStandardArgs)

find_library(ARROW_LIB libarrow.a)
find_library(ARROW_TESTING_LIB libarrow_testing.a)
find_path(ARROW_INCLUDE_PATH arrow/api.h)

find_package_handle_standard_args(
  Arrow
  DEFAULT_MSG
  ARROW_LIB
  ARROW_TESTING_LIB
  ARROW_INCLUDE_PATH
)

# Only add the libraries once.
if(Arrow_FOUND AND NOT TARGET arrow)
  add_library(arrow STATIC IMPORTED GLOBAL)
  add_library(arrow_testing STATIC IMPORTED GLOBAL)

  set_target_properties(
    arrow
    arrow_testing
    PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${ARROW_INCLUDE_PATH}
  )
  set_target_properties(arrow PROPERTIES IMPORTED_LOCATION ${ARROW_LIB})

  # arrow_testing's gtest_util.cc.o references arrow::ipc::internal::json::*
  # and arrow::Initialize symbols defined in libarrow.a. We need libarrow.a
  # to appear AFTER libarrow_testing.a in the link line so ld --as-needed
  # has the unresolved refs in its set when it sees libarrow.a.
  #
  # WORKAROUND for an ODR violation: the velox-dev:adapters image installs
  # both Apache Thrift (at /usr/local/include/thrift/transport/) and FBThrift
  # (at /usr/local/include/thrift/lib/cpp/transport/). Both inject
  # apache::thrift::transport::TMemoryBuffer into the same C++ namespace
  # with different inline bodies (different `write_virt` overflow logic,
  # null-pointer guards, etc.). When MONO=ON + static linking, the linker
  # dedupes weak template instantiations across libvelox.a's translation
  # units and picks ONE definition. Apache's wins → vendored arrow-parquet
  # code is happy. FBThrift's wins → vendored code calls FBThrift's
  # TMemoryBuffer::write_virt at runtime and tests fail with "Insufficient
  # space in external MemoryBuffer".
  #
  # Declaring the dep via the literal ${ARROW_LIB} path (instead of the
  # `arrow` target name) appends libarrow.a to consumers' link lines as a
  # terminal node, without going through CMake's transitive-dep graph. This
  # preserves the established link order under which Apache's TMemoryBuffer
  # definition wins. Using the `arrow` target here would reorder CMake's
  # dep graph (because `arrow` carries INTERFACE_LINK_LIBRARIES thrift) and
  # flip the winner to FBThrift — breaking velox_dwio_arrow_parquet_writer_test.
  #
  # Proper fix is upstream: FBThrift should not share the apache::thrift::*
  # namespace, or velox should split libvelox so FBThrift-using code and
  # vendored arrow-parquet code never land in the same link line.
  set_target_properties(
    arrow_testing
    PROPERTIES IMPORTED_LOCATION ${ARROW_TESTING_LIB} INTERFACE_LINK_LIBRARIES ${ARROW_LIB}
  )
endif()
