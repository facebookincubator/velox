# Copyright (c) Facebook, Inc. and its affiliates.
# (Apache-2.0)

# Fetch OpenJDK 17 regex corpus files at configure time.  Pinned to
# jdk-17.0.13-ga (SHA256 verified).  All three are processed by
# OpenJDK's own RegExTest.processFile.
set(_openjdk_corpus_base
  "https://raw.githubusercontent.com/openjdk/jdk17u/jdk-17.0.13-ga/test/jdk/java/util/regex")
set(_openjdk_corpus_dir "${CMAKE_CURRENT_BINARY_DIR}/openjdk_corpus")
file(MAKE_DIRECTORY "${_openjdk_corpus_dir}")

function(_fetch_corpus_file _fname _fsha)
  set(_path "${_openjdk_corpus_dir}/${_fname}")
  if(NOT EXISTS "${_path}")
    message(STATUS "Fetching OpenJDK regex corpus: ${_fname}")
    file(DOWNLOAD
      "${_openjdk_corpus_base}/${_fname}"
      "${_path}"
      EXPECTED_HASH SHA256=${_fsha}
      SHOW_PROGRESS
      STATUS _dl_status)
    list(GET _dl_status 0 _dl_code)
    if(NOT _dl_code EQUAL 0)
      message(WARNING "Failed to download ${_fname}: ${_dl_status}")
    endif()
  endif()
endfunction()

_fetch_corpus_file(TestCases.txt
  1bf5c8a2a4fba557ff4e4a5d69d86bbd2a9e0c720b9a6455aa001526375ba946)
_fetch_corpus_file(BMPTestCases.txt
  6dbdfc4c64797831b798ad5d4b546f8cbfb2e76036018fe11013f168fc4f11f2)
_fetch_corpus_file(SupplementaryTestCases.txt
  96a56b7e3d0732f6cb30d307c9025517fdf515732d2ce1ae9e5496e30367a019)

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
  RegExTestPortedTest.cpp
  OpenJdkCorpusDiffTest.cpp)

target_compile_definitions(velox_regex_compat_test
  PRIVATE OPENJDK_CORPUS_DIR="${_openjdk_corpus_dir}")

target_link_libraries(velox_regex_compat_test
  PRIVATE velox_regex_compat GTest::gtest GTest::gmock)

add_test(NAME velox_regex_compat_test COMMAND velox_regex_compat_test)
