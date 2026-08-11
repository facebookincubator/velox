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

# Only entered when VELOX_ENABLE_REGEX_COMPAT_TESTS=ON.

set(_REGEX_COMPAT_SRC Re2Regex.cpp Pcre2Regex.cpp)
set(_REGEX_COMPAT_LIBS re2::re2 pcre2-8::pcre2-8)

if(VELOX_ENABLE_REGEX_COMPAT_JAVA_BACKEND)
  list(APPEND _REGEX_COMPAT_SRC JvmFixture.cpp JavaRegex.cpp)
  list(APPEND _REGEX_COMPAT_LIBS ${JNI_LIBRARIES})
endif()

velox_add_library(velox_regex_compat ${_REGEX_COMPAT_SRC})

velox_link_libraries(velox_regex_compat
  PUBLIC ${_REGEX_COMPAT_LIBS}
  PRIVATE velox_functions_lib velox_java_pcre2_translator)

if(VELOX_ENABLE_REGEX_COMPAT_JAVA_BACKEND)
  velox_include_directories(velox_regex_compat PUBLIC ${JNI_INCLUDE_DIRS})
  velox_compile_definitions(velox_regex_compat
    PUBLIC VELOX_REGEX_COMPAT_HAS_JAVA=1)
else()
  velox_compile_definitions(velox_regex_compat
    PUBLIC VELOX_REGEX_COMPAT_HAS_JAVA=0)
endif()

if(${VELOX_BUILD_TESTING})
  add_subdirectory(tests)
endif()
