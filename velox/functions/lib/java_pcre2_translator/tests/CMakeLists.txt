# Copyright (c) Facebook, Inc. and its affiliates.
# (Apache-2.0)

add_executable(velox_java_pcre2_translator_test
  JavaRegexTranslatorTest.cpp)

target_link_libraries(velox_java_pcre2_translator_test
  PRIVATE velox_java_pcre2_translator GTest::gtest GTest::gtest_main)

add_test(NAME velox_java_pcre2_translator_test
         COMMAND velox_java_pcre2_translator_test)
