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

# Marks the syntax error count in a bison-generated C++ parser as
# [[maybe_unused]]. The parser only increments it, so Clang's
# -Wunused-but-set-variable reports it. Bison releases up to 3.8.2 generate
# it unmarked; the unreleased upstream fix marks it the same way.
#
# Usage: cmake -DBISON_OUTPUT=<generated parser> -P BisonMarkErrorCountUnused.cmake
file(READ "${BISON_OUTPUT}" content)
string(REPLACE "int yynerrs_ = 0;" "[[maybe_unused]] int yynerrs_ = 0;" content "${content}")
file(WRITE "${BISON_OUTPUT}" "${content}")
