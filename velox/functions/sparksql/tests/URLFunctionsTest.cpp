/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <gmock/gmock.h>
#include <optional>

#include "velox/functions/sparksql/tests/SparkFunctionBaseTest.h"

namespace facebook::velox::functions::sparksql::test {

namespace {
std::optional<std::string> operator+(
    const std::optional<std::string>& opt1,
    const std::optional<std::string>& opt2) {
  if (!opt1.has_value()) {
    return opt2;
  }
  if (!opt2.has_value()) {
    return opt1;
  }
  return opt1.value() + opt2.value();
}

std::optional<std::string> operator&(
    const std::optional<std::string>& opt1,
    const std::optional<std::string>& opt2) {
  if (!opt1.has_value() || !opt2.has_value()) {
    return std::nullopt;
  }
  return opt1.value() + opt2.value();
}

class URLFunctionsTest
    : public functions::sparksql::test::SparkFunctionBaseTest {
 protected:
  void validate(
      const std::optional<std::string>& url,
      const std::optional<std::string>& expectedProtocol,
      const std::optional<std::string>& expectedUserinfo,
      const std::optional<std::string>& expectedHost,
      const std::optional<std::string>& expectedPort,
      const std::optional<std::string>& expectedPath,
      const std::optional<std::string>& expectedQuery,
      const std::optional<std::string>& expectedRef) {
    const auto parseUrl = [&](const std::optional<std::string>& partToExtract)
        -> std::optional<std::string> {
      std::cout << "TEST:" << url.value() << "-> " << partToExtract.value()
                << std::endl;
      return evaluateOnce<std::string>("parse_url(c0, c1)", url, partToExtract);
    };

    EXPECT_EQ(parseUrl("PROTOCOL"), expectedProtocol);

    EXPECT_EQ(parseUrl("USERINFO"), expectedUserinfo);
    EXPECT_EQ(parseUrl("HOST"), expectedHost);
    auto expectedAuthority =
        (expectedUserinfo & "@") + expectedHost + (":" & expectedPort);
    EXPECT_EQ(parseUrl("AUTHORITY"), expectedAuthority);

    EXPECT_EQ(parseUrl("PATH"), expectedPath);
    EXPECT_EQ(parseUrl("QUERY"), expectedQuery);

    auto expectedFile = expectedPath + ("?" & expectedQuery);
    EXPECT_EQ(parseUrl("FILE"), expectedFile);

    EXPECT_EQ(parseUrl("REF"), expectedRef);
  }
};

TEST_F(URLFunctionsTest, validateURL) {
  validate(
      "http://user:pass@example.com:8080/path1/p.php?k1=v1&k2=v2#Ref1",
      "http",
      "user:pass",
      "example.com",
      "8080",
      "/path1/p.php",
      "k1=v1&k2=v2",
      "Ref1");
  validate(
      "HTTP://example.com/path1/p.php",
      "HTTP",
      std::nullopt,
      "example.com",
      std::nullopt,
      "/path1/p.php",
      std::nullopt,
      std::nullopt);
  validate(
      "http://example.com:8080/path1/p.php?k1=v1&k2=v2#Ref1",
      "http",
      std::nullopt,
      "example.com",
      "8080",
      "/path1/p.php",
      "k1=v1&k2=v2",
      "Ref1");
  validate(
      "https://username@example.com",
      "https",
      "username",
      "example.com",
      std::nullopt,
      "",
      std::nullopt,
      std::nullopt);
  validate(
      "https:/auth/login.html",
      "https",
      std::nullopt,
      std::nullopt,
      std::nullopt,
      "/auth/login.html",
      std::nullopt,
      std::nullopt);
  validate(
      "foo",
      std::nullopt,
      std::nullopt,
      std::nullopt,
      std::nullopt,
      "foo",
      std::nullopt,
      std::nullopt);
}

TEST_F(URLFunctionsTest, validateParameter) {
  const auto checkParseUrlWithKey =
      [&](const std::optional<std::string>& expected,
          const std::optional<std::string>& url,
          const std::optional<std::string>& key) {
        const std::optional<std::string>& partToExtract = "QUERY";
        EXPECT_EQ(
            evaluateOnce<std::string>(
                "parse_url(c0, c1, c2)", url, partToExtract, key),
            expected);
      };

  checkParseUrlWithKey(
      "v2", "http://example.com/path1/p.php?k1=v1&k2=v2#Ref1", "k2");
  checkParseUrlWithKey(
      "v1", "http://example.com/path1/p.php?k1=v1&k2=v2&k3&k4#Ref1", "k1");
  checkParseUrlWithKey(
      "", "http://example.com/path1/p.php?k1=v1&k2=v2&k3&k4#Ref1", "k3");
  checkParseUrlWithKey(
      std::nullopt,
      "http://example.com/path1/p.php?k1=v1&k2=v2&k3&k4#Ref1",
      "k6");
  checkParseUrlWithKey(std::nullopt, "foo", "");
}

TEST_F(URLFunctionsTest, sparkUT) {
  const auto checkParseUrl =
      [&](const std::optional<std::string>& expected,
          const std::optional<std::string>& url,
          const std::optional<std::string>& partToExtract) {
        EXPECT_EQ(
            evaluateOnce<std::string>("parse_url(c0, c1)", url, partToExtract),
            expected);
      };
  const auto checkParseUrlWithKey =
      [&](const std::optional<std::string>& expected,
          const std::optional<std::string>& url,
          const std::optional<std::string>& partToExtract,
          const std::optional<std::string>& key) {
        EXPECT_EQ(
            evaluateOnce<std::string>(
                "parse_url(c0, c1, c2)", url, partToExtract, key),
            expected);
      };

  checkParseUrl(
      "spark.apache.org", "http://spark.apache.org/path?query=1", "HOST");
  checkParseUrl("/path", "http://spark.apache.org/path?query=1", "PATH");
  checkParseUrl("query=1", "http://spark.apache.org/path?query=1", "QUERY");
  checkParseUrl("Ref", "http://spark.apache.org/path?query=1#Ref", "REF");
  checkParseUrl("http", "http://spark.apache.org/path?query=1", "PROTOCOL");
  checkParseUrl(
      "/path?query=1", "http://spark.apache.org/path?query=1", "FILE");
  checkParseUrl(
      "spark.apache.org:8080",
      "http://spark.apache.org:8080/path?query=1",
      "AUTHORITY");
  checkParseUrl(
      "userinfo", "http://userinfo@spark.apache.org/path?query=1", "USERINFO");
  checkParseUrlWithKey(
      "1", "http://spark.apache.org/path?query=1", "QUERY", "query");

  // Null checking.
  checkParseUrl(std::nullopt, std::nullopt, "HOST");
  checkParseUrl(
      std::nullopt, "http://spark.apache.org/path?query=1", std::nullopt);
  checkParseUrl(std::nullopt, std::nullopt, std::nullopt);
  checkParseUrl(std::nullopt, "test", "HOST");
  checkParseUrl(std::nullopt, "http://spark.apache.org/path?query=1", "NO");
  checkParseUrl(
      std::nullopt, "http://spark.apache.org/path?query=1", "USERINFO");
  checkParseUrlWithKey(
      std::nullopt, "http://spark.apache.org/path?query=1", "HOST", "query");
  checkParseUrlWithKey(
      std::nullopt, "http://spark.apache.org/path?query=1", "QUERY", "quer");
  checkParseUrlWithKey(
      std::nullopt,
      "http://spark.apache.org/path?query=1",
      "QUERY",
      std::nullopt);
  checkParseUrlWithKey(
      std::nullopt, "http://spark.apache.org/path?query=1", "QUERY", "");
}

} // namespace
} // namespace facebook::velox::functions::sparksql::test
