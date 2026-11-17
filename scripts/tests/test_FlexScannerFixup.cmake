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

if(NOT DEFINED FLEX_FIXUP_SCRIPT)
  set(FLEX_FIXUP_SCRIPT "${CMAKE_CURRENT_LIST_DIR}/../../CMake/FlexScannerFixup.cmake")
endif()

function(check_fixup name generated expected)
  set(FLEX_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/test_FlexScannerFixup_${name}.cpp")
  file(WRITE "${FLEX_OUTPUT}" "${generated}")
  include("${FLEX_FIXUP_SCRIPT}")
  file(READ "${FLEX_OUTPUT}" actual)
  if(NOT actual STREQUAL expected)
    file(REMOVE "${FLEX_OUTPUT}")
    message(FATAL_ERROR "${name}: unexpected scanner fixup:\n${actual}")
  endif()
  include("${FLEX_FIXUP_SCRIPT}")
  file(READ "${FLEX_OUTPUT}" repeated)
  file(REMOVE "${FLEX_OUTPUT}")
  if(NOT repeated STREQUAL expected)
    message(FATAL_ERROR "${name}: scanner fixup is not idempotent")
  endif()
endfunction()

check_fixup(
  legacy
  [=[/*FALLTHROUGH*/
YY_FATAL_ERROR("flex scanner jammed");
LexerError("input in flex scanner failed");
]=]
  [=[[[fallthrough]];
YY_FATAL_ERROR("flex scanner jammed");
LexerError("input in flex scanner failed");
]=]
)

check_fixup(
  panic
  [=[/*FALLTHROUGH*/
yypanic("flex scanner jammed" );
yypanic ("flex scanner jammed");
yypanic("a different error");
]=]
  [=[[[fallthrough]];
YY_FATAL_ERROR("flex scanner jammed" );
YY_FATAL_ERROR("flex scanner jammed");
yypanic("a different error");
]=]
)
