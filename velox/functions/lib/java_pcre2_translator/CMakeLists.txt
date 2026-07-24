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

velox_add_library(
  velox_java_pcre2_translator
  ClassBodyParser.cpp
  ClassRenderer.cpp
  Evaluator.cpp
  JavaRegexTranslator.cpp
  JdkPropertyExpander.cpp
  PropertyMap.cpp
  RangeSet.cpp
  HEADERS
  ClassBodyParser.h
  ClassNode.h
  ClassRenderer.h
  EvaluationFailedException.h
  Evaluator.h
  JavaRegexTranslator.h
  JdkPropertyExpander.h
  PropertyMap.h
  RangeSet.h
)

velox_link_libraries(velox_java_pcre2_translator PRIVATE ICU::uc)

if(${VELOX_BUILD_TESTING})
  add_subdirectory(tests)
endif()
