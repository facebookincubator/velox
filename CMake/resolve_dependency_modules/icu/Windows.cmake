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

if(NOT CMAKE_VS_MSBUILD_COMMAND OR NOT CMAKE_VS_PLATFORM_NAME STREQUAL "x64")
  message(FATAL_ERROR "Bundled ICU on Windows currently requires a Visual Studio x64 generator.")
endif()
if(CMAKE_MSVC_RUNTIME_LIBRARY AND NOT CMAKE_MSVC_RUNTIME_LIBRARY MATCHES "DLL$")
  message(FATAL_ERROR "ICU's native Windows shared build requires the dynamic MSVC runtime.")
endif()

set(ICU_DIR ${CMAKE_CURRENT_BINARY_DIR}/_deps/icu)
set(ICU_INCLUDE_DIRS ${ICU_DIR}/$<CONFIG>/include)
set(icu_native_source ${CMAKE_CURRENT_BINARY_DIR}/icu-src-$<CONFIG>)
set(icu_native_configuration "$<IF:$<CONFIG:Debug>,Debug,Release>")
set(ICU_BUILD_SCRIPT ${CMAKE_CURRENT_BINARY_DIR}/icu-build-windows.cmd)
configure_file(
  ${CMAKE_CURRENT_LIST_DIR}/build-windows.cmd.in
  ${ICU_BUILD_SCRIPT}
  @ONLY
  NEWLINE_STYLE DOS
)

set(icu_components data i18n io uc)
set(icu_library_data icudt)
set(icu_library_i18n icuin)
set(icu_library_io icuio)
set(icu_library_uc icuuc)
set(ICU_LIBRARIES)
set(source_libraries)
set(source_dlls)
set(installed_outputs)
foreach(component IN LISTS icu_components)
  add_library(ICU::${component} SHARED IMPORTED)
  list(APPEND ICU_LIBRARIES ICU::${component})
  set(suffix "")
  if(NOT component STREQUAL "data")
    set(suffix "$<$<CONFIG:Debug>:d>")
  endif()
  set(library ${icu_library_${component}}${suffix}.lib)
  set(dll ${icu_library_${component}}${VELOX_ICU4C_BUILD_VERSION}${suffix}.dll)
  list(APPEND source_libraries ${icu_native_source}/lib64/${library})
  list(APPEND source_dlls ${icu_native_source}/bin64/${dll})
  list(APPEND installed_outputs ${ICU_DIR}/$<CONFIG>/lib/${library} ${ICU_DIR}/$<CONFIG>/bin/${dll})
endforeach()

# Native Debug and Release builds share data filenames, so keep their trees separate.
foreach(configuration IN LISTS CMAKE_CONFIGURATION_TYPES)
  string(TOUPPER ${configuration} configuration_upper)
  set(prefix ${ICU_DIR}/${configuration})
  file(MAKE_DIRECTORY ${prefix}/include)
  foreach(component IN LISTS icu_components)
    set(suffix "")
    if(configuration STREQUAL "Debug" AND NOT component STREQUAL "data")
      set(suffix d)
    endif()
    set(library ${icu_library_${component}}${suffix}.lib)
    set(dll ${icu_library_${component}}${VELOX_ICU4C_BUILD_VERSION}${suffix}.dll)
    set_property(
      TARGET ICU::${component}
      APPEND
      PROPERTY IMPORTED_CONFIGURATIONS ${configuration_upper}
    )
    set_target_properties(
      ICU::${component}
      PROPERTIES
        IMPORTED_IMPLIB_${configuration_upper} ${prefix}/lib/${library}
        IMPORTED_LOCATION_${configuration_upper} ${prefix}/bin/${dll}
    )
  endforeach()
endforeach()

ExternalProject_Add(
  ICU
  URL ${VELOX_ICU4C_SOURCE_URL}
  URL_HASH ${VELOX_ICU4C_BUILD_SHA256_CHECKSUM}
  PATCH_COMMAND git apply ${CMAKE_CURRENT_LIST_DIR}/windows-data-rebuild.patch
  SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/icu-source
  BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/icu-build
  INSTALL_DIR ${ICU_DIR}
  EXCLUDE_FROM_ALL TRUE
  CONFIGURE_COMMAND
    ${CMAKE_COMMAND} -E copy_directory_if_different "<SOURCE_DIR>" "${icu_native_source}"
  BUILD_COMMAND
    "$ENV{COMSPEC}" /d /c call "${ICU_BUILD_SCRIPT}" "${icu_native_source}"
    ${icu_native_configuration}
  BUILD_BYPRODUCTS ${source_libraries} ${source_dlls}
  INSTALL_COMMAND
    ${CMAKE_COMMAND} -E make_directory "<INSTALL_DIR>/$<CONFIG>/lib" "<INSTALL_DIR>/$<CONFIG>/bin"
  COMMAND
    ${CMAKE_COMMAND} -E copy_directory "${icu_native_source}/include"
    "<INSTALL_DIR>/$<CONFIG>/include"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${source_libraries} "<INSTALL_DIR>/$<CONFIG>/lib"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${source_dlls} "<INSTALL_DIR>/$<CONFIG>/bin"
  INSTALL_BYPRODUCTS ${installed_outputs}
)
ExternalProject_Add_StepDependencies(ICU build ${ICU_BUILD_SCRIPT})

foreach(component IN LISTS icu_components)
  add_dependencies(ICU::${component} ICU)
  target_include_directories(ICU::${component} SYSTEM INTERFACE ${ICU_INCLUDE_DIRS})
endforeach()
target_link_libraries(ICU::uc INTERFACE ICU::data)
target_link_libraries(ICU::i18n INTERFACE ICU::uc)
target_link_libraries(ICU::io INTERFACE ICU::i18n ICU::uc)
add_library(ICU::ICU INTERFACE IMPORTED)
target_link_libraries(ICU::ICU INTERFACE ${ICU_LIBRARIES})
list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR})
