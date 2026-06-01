# Copyright (c) Facebook, Inc. and its affiliates.
# (Apache-2.0)

# Fetch OpenJDK 17 regex TestCases.txt corpus at configure time.  Pinned
# to jdk-17.0.13-ga for reproducibility (SHA256 verified).
set(_openjdk_corpus_url
  "https://raw.githubusercontent.com/openjdk/jdk17u/jdk-17.0.13-ga/test/jdk/java/util/regex/TestCases.txt")
set(_openjdk_corpus_sha256
  "1bf5c8a2a4fba557ff4e4a5d69d86bbd2a9e0c720b9a6455aa001526375ba946")
set(_openjdk_corpus_path
  "${CMAKE_CURRENT_BINARY_DIR}/openjdk_corpus_TestCases.txt")
if(NOT EXISTS "${_openjdk_corpus_path}")
  message(STATUS "Fetching OpenJDK regex TestCases.txt corpus...")
  file(DOWNLOAD
    "${_openjdk_corpus_url}"
    "${_openjdk_corpus_path}"
    EXPECTED_HASH SHA256=${_openjdk_corpus_sha256}
    SHOW_PROGRESS
    STATUS _dl_status)
  list(GET _dl_status 0 _dl_code)
  if(NOT _dl_code EQUAL 0)
    message(WARNING "Failed to download OpenJDK corpus: ${_dl_status}")
  endif()
endif()

add_executable(velox_regex_compat_test
  TestMain.cpp
  Re2RegexTest.cpp
  Pcre2RegexTest.cpp
  JavaRegexTest.cpp
  BackendTypedTest.cpp
  PatternPortedTest.cpp
  MatcherMatchingPortedTest.cpp
  MatcherReplacementPortedTest.cpp
  MatcherResultsPortedTest.cpp
  MatcherMatchResultPortedTest.cpp
  PatternSplitPortedTest.cpp
  MatcherUnicodePortedTest.cpp
  OpenJdkCorpusDiffTest.cpp)

target_compile_definitions(velox_regex_compat_test
  PRIVATE OPENJDK_CORPUS_PATH="${_openjdk_corpus_path}")

target_link_libraries(velox_regex_compat_test
  PRIVATE velox_regex_compat GTest::gtest GTest::gmock)

add_test(NAME velox_regex_compat_test COMMAND velox_regex_compat_test)
