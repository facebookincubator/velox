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

include(CheckCXXSourceCompiles)

set(TEST_FLAG "-mavx512f -mavx512bw -mavx512vl -mavx512vbmi -mavx512vbmi2")
set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} ${TEST_FLAG}")
check_cxx_source_compiles(
  "
    #include <immintrin.h>
    int main() {
        auto a = _mm512_multishift_epi64_epi8(__m512i(), __m512i());
        (void)a;
        return 0;
    }
"
  SUPPORT_AVX512_VBMI)

# Set AVX-512 flag for specific source build
set(VELOX_AVX512_FLAG "")
if(SUPPORT_AVX512_VBMI)
  set(VELOX_AVX512_FLAG "${VELOX_AVX512_FLAG} ${TEST_FLAG}")
  add_definitions(-DVELOX_ENABLE_AVX512)
  message("Use ${VELOX_AVX512_FLAG} to compile")
else()
  message(FATAL_ERROR "AVX-512 not supported")
endif()
