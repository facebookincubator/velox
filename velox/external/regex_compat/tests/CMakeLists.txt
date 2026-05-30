# Copyright (c) Facebook, Inc. and its affiliates.
# (Apache-2.0)

add_executable(velox_regex_compat_test
  TestMain.cpp
  Re2RegexTest.cpp
  Pcre2RegexTest.cpp
  JavaRegexTest.cpp
  BackendTypedTest.cpp)

target_link_libraries(velox_regex_compat_test
  PRIVATE velox_regex_compat GTest::gtest GTest::gmock)

add_test(NAME velox_regex_compat_test COMMAND velox_regex_compat_test)
