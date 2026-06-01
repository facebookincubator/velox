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
# Copyright (c) Facebook, Inc. and its affiliates.
# (Apache-2.0)

add_executable(
  velox_java_pcre2_translator_test
  ClassBodyParserTest.cpp
  ClassRendererTest.cpp
  EvaluatorTest.cpp
  JavaRegexTranslatorTest.cpp
  JdkPropertyExpanderTest.cpp
  PropertyMapTest.cpp
  RangeSetTest.cpp
)

target_link_libraries(
  velox_java_pcre2_translator_test
  PRIVATE velox_java_pcre2_translator GTest::gtest GTest::gtest_main
)

add_test(NAME velox_java_pcre2_translator_test COMMAND velox_java_pcre2_translator_test)
