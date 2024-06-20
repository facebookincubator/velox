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

  if(DEFINED ENV{ARROW_EP_PATH})
    message(
      STATUS
        "Arrow path is set: $ENV{ARROW_EP_PATH}, trying to use existing arrow libraries."
    )
    #find_package(Arrow REQUIRED PATHS $ENV{ARROW_EP_PATH}/install/lib/cmake/Arrow/)
    set(ARROW_LIB_PATH $ENV{ARROW_EP_PATH}/install/lib/)
    find_library(ARROW_LIB libarrow.a PATHS ${ARROW_LIB_PATH})
    find_library(PARQUET_LIB libparquet.a PATHS ${ARROW_LIB_PATH})
    find_library(ARROW_TESTING_LIB libarrow_testing.a PATHS ${ARROW_LIB_PATH})
    if("${ARROW_LIB}" STREQUAL "ARROW_LIB-NOTFOUND"
       #OR "${PARQUET_LIB}" STREQUAL "PARQUET_LIB-NOTFOUND"
       OR "${ARROW_TESTING_LIB}" STREQUAL "ARROW_TESTING_LIB-NOTFOUND")
      message(FATAL_ERROR "Arrow libraries not found in ${ARROW_LIB_PATH}")
    endif()
    set(Arrow_FOUND true)

    add_library(thrift STATIC IMPORTED GLOBAL)
    if(NOT Thrift_FOUND)
      set(THRIFT_ROOT $ENV{ARROW_EP_PATH}/src/arrow_ep-build/thrift_ep-install)
      find_library(THRIFT_LIB thrift PATHS ${THRIFT_ROOT}/lib)
      if("${THRIFT_LIB}" STREQUAL "THRIFT_LIB-NOTFOUND")
        message(FATAL_ERROR "Thrift library not found in ${THRIFT_ROOT}/lib")
      endif()
      set(THRIFT_INCLUDE_DIR ${THRIFT_ROOT}/include)
    endif()
    set_property(TARGET thrift PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                                        ${THRIFT_INCLUDE_DIR})
    set_property(TARGET thrift PROPERTY IMPORTED_LOCATION ${THRIFT_LIB})

    add_library(arrow STATIC IMPORTED GLOBAL)
    add_library(parquet STATIC IMPORTED GLOBAL)
    add_library(arrow_testing STATIC IMPORTED GLOBAL)

    set_target_properties(
      arrow arrow_testing parquet
      PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
                 $ENV{ARROW_EP_PATH}/install/include)
    set_target_properties(arrow PROPERTIES IMPORTED_LOCATION ${ARROW_LIB})
    set_property(TARGET arrow PROPERTY INTERFACE_LINK_LIBRARIES ${RE2} thrift)
    set_target_properties(parquet PROPERTIES IMPORTED_LOCATION ${PARQUET_LIB})
    set_target_properties(arrow_testing PROPERTIES IMPORTED_LOCATION
                                                   ${ARROW_TESTING_LIB})
  endif()
