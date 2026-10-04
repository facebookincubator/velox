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
#include <atomic>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <libxml/parser.h>
#include <libxml/xmlerror.h>

#include "velox/common/base/VeloxException.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/testutil/TempDirectoryPath.h"
#include "velox/functions/sparksql/XPathUtil.h"
#include "velox/functions/sparksql/tests/SparkFunctionBaseTest.h"

namespace facebook::velox::functions::sparksql::test {
namespace {

std::string writeFile(
    const std::string& directory,
    const std::string& name,
    const std::string& content) {
  const auto file = std::filesystem::path(directory) / name;
  {
    std::ofstream out(file, std::ios::binary);
    out << content;
  }
  VELOX_CHECK(std::filesystem::exists(file), "Failed to write {}", name);
  return "file://" + file.string();
}

// Returns the message of the user error that 'fn' throws. Any other outcome
// fails the test and returns an empty string.
std::string userErrorMessage(const std::function<void()>& fn) {
  try {
    fn();
  } catch (const VeloxUserError& error) {
    return error.message();
  } catch (const std::exception& error) {
    ADD_FAILURE() << "Expected a user error, got: " << error.what();
    return "";
  }
  ADD_FAILURE() << "Expected a user error";
  return "";
}

// True if 'message' is 'prefix' followed by a decimal libxml2 code and a
// closing parenthesis, with nothing else after it.
bool hasCodeSuffix(const std::string& message, const std::string& prefix) {
  if (message.size() < prefix.size() + 2 || message.rfind(prefix, 0) != 0 ||
      message.back() != ')') {
    return false;
  }
  const auto code =
      message.substr(prefix.size(), message.size() - prefix.size() - 1);
  return code.find_first_not_of("0123456789") == std::string::npos;
}

std::optional<bool> evalBooleanDirect(
    std::string_view xml,
    std::string_view path) {
  auto evaluated = xpath::evalBoolean(xml, path);
  if (evaluated.hasError()) {
    VELOX_USER_FAIL("{}", evaluated.error().message());
  }
  return evaluated.value();
}

std::optional<std::string> evalStringDirect(
    std::string_view xml,
    std::string_view path) {
  auto evaluated = xpath::evalString(xml, path);
  if (evaluated.hasError()) {
    VELOX_USER_FAIL("{}", evaluated.error().message());
  }
  if (!evaluated.value().has_value()) {
    return std::nullopt;
  }
  return std::string(evaluated.value()->view());
}

class XPathFunctionsTest : public SparkFunctionBaseTest {
 protected:
  // Helper for xpath_boolean.
  std::optional<bool> xpathBoolean(
      const std::string& xml,
      const std::string& path) {
    return evaluateOnce<bool>(
        "xpath_boolean(c0, c1)",
        std::optional<std::string>(xml),
        std::optional<std::string>(path));
  }

  // Helper for xpath_string.
  std::optional<std::string> xpathString(
      const std::string& xml,
      const std::string& path) {
    return evaluateOnce<std::string>(
        "xpath_string(c0, c1)",
        std::optional<std::string>(xml),
        std::optional<std::string>(path));
  }

  // Checks both functions on the same input. xpath_boolean(P) is expected to
  // equal boolean(P) and xpath_string(P) to equal string(P).
  void expectBoth(
      const std::string& xml,
      const std::string& path,
      const std::optional<std::string>& expectedString,
      const std::optional<bool>& expectedBoolean) {
    SCOPED_TRACE(path);
    EXPECT_EQ(xpathString(xml, path), expectedString);
    EXPECT_EQ(xpathBoolean(xml, path), expectedBoolean);
  }

  // Checks that both functions raise a user error containing 'message'.
  void expectUserErrorBoth(
      const std::string& xml,
      const std::string& path,
      const std::string& message) {
    SCOPED_TRACE(path);
    VELOX_ASSERT_USER_THROW(xpathString(xml, path), message);
    VELOX_ASSERT_USER_THROW(xpathBoolean(xml, path), message);
  }
};

TEST_F(XPathFunctionsTest, xpathBooleanBasic) {
  {
    SCOPED_TRACE("Node-set truthiness: true when a matching node exists.");
    EXPECT_EQ(xpathBoolean("<a><b>1</b></a>", "a/b"), true);
    EXPECT_EQ(xpathBoolean("<a><b>1</b></a>", "a/c"), false);
    EXPECT_EQ(xpathBoolean("<a><b>true</b></a>", "a/b"), true);
  }
  {
    // A node-set is true when non-empty regardless of the node's text, so
    // falsy-looking text ("false", "0") must still yield true - a guard
    // against ever testing the node text instead of its existence.
    SCOPED_TRACE("Node existence, not node text, decides the result.");
    EXPECT_EQ(xpathBoolean("<a><b>false</b></a>", "a/b"), true);
    EXPECT_EQ(xpathBoolean("<a><b>0</b></a>", "a/b"), true);
  }
  {
    // A boolean XPath expression returns its own evaluated value.
    SCOPED_TRACE("Boolean expression returns its own value.");
    EXPECT_EQ(xpathBoolean("<a><b>1</b></a>", "a/b = 1"), true);
    EXPECT_EQ(xpathBoolean("<a><b>1</b></a>", "a/b = 2"), false);
  }
}

TEST_F(XPathFunctionsTest, xpathBooleanScalarConversions) {
  for (const auto& path :
       {"false()", "0", "-0", "0 div 0", "number('not a number')", "''"}) {
    SCOPED_TRACE(path);
    EXPECT_EQ(xpathBoolean("<a/>", path), false);
  }
  for (const auto& path :
       {"true()", "1", "-1", "1 div 0", "-1 div 0", "'false'", "'0'"}) {
    SCOPED_TRACE(path);
    EXPECT_EQ(xpathBoolean("<a/>", path), true);
  }
}

TEST_F(XPathFunctionsTest, xpathStringBasic) {
  EXPECT_EQ(xpathString("<a><b>b</b><c>cc</c></a>", "a/c"), "cc");
  EXPECT_EQ(xpathString("<a><b>b1</b><b>b2</b></a>", "a/b"), "b1");
  // No match returns an empty string, not NULL.
  EXPECT_EQ(xpathString("<a><b>b</b></a>", "a/c"), "");
}

TEST_F(XPathFunctionsTest, mixedRowNullsAndErrors) {
  const auto input = makeRowVector({
      makeNullableFlatVector<std::string>(
          {"<a>first long output string</a>",
           "<a/>",
           "",
           std::nullopt,
           "<a/>",
           "not xml",
           "<a/>",
           "<a/>",
           "<a>tail</a>",
           "<a/>",
           "<a/>"}),
      makeNullableFlatVector<std::string>(
          {"a",
           "false()",
           "a",
           "a",
           std::nullopt,
           "a",
           "p:a",
           "''",
           "a",
           "a[",
           "1 div 10000000"}),
  });
  velox::test::assertEqualVectors(
      makeNullableFlatVector<std::string>(
          {"first long output string",
           "false",
           std::nullopt,
           std::nullopt,
           std::nullopt,
           std::nullopt,
           std::nullopt,
           "",
           "tail",
           std::nullopt,
           "0.0000001"}),
      evaluate("try(xpath_string(c0, c1))", input));
  velox::test::assertEqualVectors(
      makeNullableFlatVector<bool>(
          {true,
           false,
           std::nullopt,
           std::nullopt,
           std::nullopt,
           std::nullopt,
           std::nullopt,
           false,
           true,
           std::nullopt,
           true}),
      evaluate("try(xpath_boolean(c0, c1))", input));
}

TEST_F(XPathFunctionsTest, xpathNumberStringConversions) {
  const std::vector<std::pair<std::string, std::string>> cases = {
      {"0", "0"},
      {"-0", "0"},
      {"0 div 0", "NaN"},
      {"1 div 0", "Infinity"},
      {"-1 div 0", "-Infinity"},
      {"1 div 3", "0.3333333333333333"},
      {"1 div 10000000", "0.0000001"},
      {"-1 div 10000000", "-0.0000001"},
      {"2147483648", "2147483648"},
      {"100000000000000000000", "100000000000000000000"},
      {"9007199254740991", "9007199254740991"},
  };
  // The UDF converts a numeric result itself, so this holds for every
  // supported libxml2 version.
  for (const auto& [path, expected] : cases) {
    SCOPED_TRACE(path);
    EXPECT_EQ(xpathString("<a/>", path), expected);
  }
  EXPECT_EQ(xpathString("<a/>", "translate('120', 12, 34)"), "340");
  EXPECT_EQ(
      xpathString(
          "<!DOCTYPE a [<!ATTLIST b id ID #REQUIRED>]>"
          "<a><b id='n1'>value</b></a>",
          "id('n1')"),
      "value");
  EXPECT_EQ(xpathBoolean("<a xml:lang='en'><b/></a>", "a/b[lang('en')]"), true);
  // Conversions inside XPath go through the string-function wrappers.
  // Repetition also checks that converting a cached numeric constant does not
  // mutate the compiled expression's copy of that constant.
  for (int pass = 0; pass < 2; ++pass) {
    for (const auto& [path, expected] : cases) {
      SCOPED_TRACE(path);
      EXPECT_EQ(xpathString("<a/>", "string(" + path + ")"), expected);
      EXPECT_EQ(xpathString("<a/>", "concat('', " + path + ")"), expected);
    }
  }
  EXPECT_EQ(xpathBoolean("<a/>", "contains(1 div 10000000, 'e')"), false);
  EXPECT_EQ(xpathBoolean("<a/>", "starts-with(1 div 10000000, '0.')"), true);
  EXPECT_EQ(xpathString("<a/>", "string-length(1 div 10000000)"), "9");
  EXPECT_EQ(xpathString("<a/>", "substring(1 div 10000000, 3, 3)"), "000");
  EXPECT_EQ(
      xpathString("<a/>", "substring-before(1 div 10000000, '1')"), "0.000000");
  EXPECT_EQ(
      xpathString("<a/>", "substring-after(1 div 10000000, '0.')"), "0000001");
  EXPECT_EQ(
      xpathString("<a/>", "normalize-space(1 div 10000000)"), "0.0000001");
  EXPECT_EQ(
      xpathString("<a/>", "concat(substring(12345, 2, 2), 1 div 10000000)"),
      "230.0000001");
  EXPECT_EQ(
      xpathString("<a>100000000000000000000</a>", "string(number(a))"),
      "100000000000000000000");
  EXPECT_EQ(xpathString("<a>0.0000001</a>", "string(number(a))"), "0.0000001");
}

TEST_F(XPathFunctionsTest, invalidXmlThrows) {
  // Spark's UDFXPathUtil throws on malformed XML, so these raise user errors
  // instead of returning NULL.
  VELOX_ASSERT_USER_THROW(
      xpathBoolean("not xml", "a/b"), "Invalid XML document");
  VELOX_ASSERT_USER_THROW(
      xpathString("<a><b></a>", "a/b"), "Invalid XML document");
  VELOX_ASSERT_USER_THROW(xpathString("<a/><b/>", "/"), "Invalid XML document");
}

TEST_F(XPathFunctionsTest, invalidXPathThrows) {
  // An XPath syntax error throws (Spark's xpath.compile throws
  // RuntimeException), so these raise user errors.
  VELOX_ASSERT_USER_THROW(
      xpathBoolean("<a><b>1</b></a>", "///[invalid"), "Invalid XPath");
  VELOX_ASSERT_USER_THROW(
      xpathString("<a><b>1</b></a>", "a/b["), "Invalid XPath");
}

TEST_F(XPathFunctionsTest, xpathEvaluationErrorsThrow) {
  for (const auto& path :
       {"missing()",
        "$missing",
        "contains('x')",
        "1 | 2",
        "string(1, 2)",
        "concat(1)",
        "substring(123)"}) {
    SCOPED_TRACE(path);
    VELOX_ASSERT_USER_THROW(xpathBoolean("<a/>", path), "Invalid XPath");
    VELOX_ASSERT_USER_THROW(xpathString("<a/>", path), "Invalid XPath");
  }
}

TEST_F(XPathFunctionsTest, namespacePrefixedReturnsNull) {
  // A prefixed step references an unregistered prefix -> evaluation fails ->
  // NULL, as documented in the namespace compatibility notes.
  EXPECT_EQ(
      xpathString(
          "<x:a xmlns:x=\"http://e\"><x:b>v</x:b></x:a>", "x:a/x:b/text()"),
      std::nullopt);
  // Undeclared prefix in the input parses but still fails evaluation.
  EXPECT_EQ(
      xpathString("<ns:a><ns:b>v</ns:b></ns:a>", "ns:a/ns:b/text()"),
      std::nullopt);
  EXPECT_EQ(xpathBoolean("<a xmlns=\"urn:test\"><b>v</b></a>", "a/b"), false);
  EXPECT_EQ(xpathString("<a xmlns=\"urn:test\"><b>v</b></a>", "a/b"), "");
  EXPECT_EQ(xpathString("<a xml:lang='en'/>", "a/@xml:lang"), "en");
  EXPECT_EQ(xpathBoolean("<a/>", "false() and ns:a"), false);
}

TEST_F(XPathFunctionsTest, doctypeHandling) {
  {
    SCOPED_TRACE("DOCTYPE without internal subset is accepted.");
    EXPECT_EQ(
        xpathString("<!DOCTYPE foo><foo><b>val</b></foo>", "foo/b/text()"),
        "val");
  }
  {
    SCOPED_TRACE("Unused internal entity declarations are accepted.");
    EXPECT_EQ(
        xpathString(
            "<!DOCTYPE foo [<!ENTITY x \"hello\">]><foo><b>val</b></foo>",
            "foo/b/text()"),
        "val");
  }
  {
    SCOPED_TRACE("Brackets inside quoted values in the internal subset.");
    EXPECT_EQ(
        xpathString(
            "<!DOCTYPE foo [<!ENTITY x SYSTEM \"file://[test]\">]>"
            "<foo><b>42</b></foo>",
            "foo/b/text()"),
        "42");
  }
  {
    SCOPED_TRACE("Comment containing ']' and '>' inside the internal subset.");
    EXPECT_EQ(
        xpathString(
            "<!DOCTYPE foo [<!-- ] --><!ENTITY y \"z\">]>"
            "<foo><b>val</b></foo>",
            "foo/b/text()"),
        "val");
  }
  {
    SCOPED_TRACE("PI containing ']' and '>' inside the internal subset.");
    EXPECT_EQ(
        xpathString(
            "<!DOCTYPE foo [<?pi ]> ?><!ENTITY y \"z\">]>"
            "<foo><b>val</b></foo>",
            "foo/b/text()"),
        "val");
  }
  {
    // Retained user-defined entity references produce NULL rather than being
    // expanded for XPath evaluation. Spark expands internal entities.
    SCOPED_TRACE("Internal entity reference returns NULL.");
    EXPECT_EQ(
        xpathString(
            "<!DOCTYPE foo [<!ENTITY x \"hi\">]><foo><b>&x;</b></foo>",
            "foo/b/text()"),
        std::nullopt);
    EXPECT_EQ(
        xpathString(
            "<!DOCTYPE foo [<!ENTITY x \"hi\">]><foo b=\"&x;\"/>", "foo/@b"),
        std::nullopt);
  }
  {
    // Real files are used by externalResourcesAreNeverLoaded; a nonexistent
    // path would pass even if loading were enabled.
    SCOPED_TRACE("A SYSTEM entity reference is retained, not expanded.");
    const std::string xml =
        "<!DOCTYPE foo [<!ENTITY x SYSTEM \"file:///xpath-absent\">]>"
        "<foo>&x;</foo>";
    EXPECT_EQ(xpathBoolean(xml, "foo"), std::nullopt);
    EXPECT_EQ(xpathString(xml, "foo"), std::nullopt);
  }
  {
    SCOPED_TRACE("A DOCTYPE that names an external subset is accepted.");
    EXPECT_EQ(
        xpathString(
            "<!DOCTYPE foo SYSTEM \"file:///xpath-absent\">"
            "<foo>ok</foo>",
            "foo"),
        "ok");
    // An undeclared reference is retained when the external subset is not
    // loaded, even though the document has no internal subset.
    EXPECT_EQ(
        xpathString(
            "<!DOCTYPE foo SYSTEM \"file:///xpath-absent\">"
            "<foo>&bar;</foo>",
            "foo"),
        std::nullopt);
    EXPECT_EQ(
        xpathString(
            "<!DOCTYPE foo SYSTEM \"file:///xpath-absent\">"
            "<foo b=\"&bar;\"/>",
            "foo/@b"),
        std::nullopt);
    VELOX_ASSERT_USER_THROW(
        xpathString(
            "<!DOCTYPE foo SYSTEM \"file:///xpath-absent\">"
            "<foo>&bar;</foo>",
            "///["),
        "Invalid XPath");
    VELOX_ASSERT_USER_THROW(
        xpathString(
            "<?xml version=\"1.0\" standalone=\"yes\"?>"
            "<!DOCTYPE foo SYSTEM \"file:///xpath-absent\">"
            "<foo>&bar;</foo>",
            "foo"),
        "Invalid XML document");
  }
  {
    SCOPED_TRACE("An external parameter entity declaration is accepted.");
    EXPECT_EQ(
        xpathString(
            "<!DOCTYPE foo [<!ENTITY % remote SYSTEM "
            "\"file:///xpath-absent\"> %remote;]>"
            "<foo>ok</foo>",
            "foo"),
        "ok");
  }
  {
    SCOPED_TRACE("A retained internal reference is not expanded.");
    EXPECT_EQ(
        xpathString(
            "<!DOCTYPE lolz [<!ENTITY lol \"lol\">"
            "<!ENTITY lol2 \"&lol;&lol;&lol;&lol;&lol;\">]>"
            "<lolz>&lol2;</lolz>",
            "lolz"),
        std::nullopt);
  }
}

TEST_F(XPathFunctionsTest, prologHandling) {
  {
    SCOPED_TRACE("An xml-stylesheet processing instruction is accepted.");
    EXPECT_EQ(
        xpathString(
            "<?xml-stylesheet type=\"text/xsl\" href=\"s.xsl\"?>"
            "<a><b>val</b></a>",
            "a/b/text()"),
        "val");
  }
  {
    SCOPED_TRACE("An XML declaration is accepted.");
    EXPECT_EQ(
        xpathString("<?xml version=\"1.0\"?><a><b>val</b></a>", "a/b/text()"),
        "val");
  }
  {
    SCOPED_TRACE("Comment before DOCTYPE is accepted.");
    EXPECT_EQ(
        xpathString(
            "<!-- lead --><!DOCTYPE foo><foo><b>val</b></foo>", "foo/b/text()"),
        "val");
  }
  {
    SCOPED_TRACE("PI before DOCTYPE is accepted.");
    EXPECT_EQ(
        xpathString(
            "<?pi data?><!DOCTYPE foo><foo><b>val</b></foo>", "foo/b/text()"),
        "val");
  }
  {
    // Spark parses a character Reader, not bytes in the declared encoding.
    // Velox's StringType inputs likewise remain UTF-8 regardless of the
    // declaration, including encodings not supported by the bundled parser.
    SCOPED_TRACE("Encoding declarations do not reinterpret StringType.");
    EXPECT_EQ(
        xpathString(
            "<?xml version=\"1.0\" encoding=\"ISO-8859-1\"?>"
            "<a><b>\xc3\xa9</b></a>",
            "a/b/text()"),
        "\xc3\xa9");
    EXPECT_EQ(
        xpathString(
            "<?xml version=\"1.0\" encoding=\"UTF-16\"?>"
            "<a><b>val</b></a>",
            "a/b/text()"),
        "val");
    EXPECT_EQ(
        xpathString(
            "<?xml version=\"1.0\" encoding=\"not-a-real-encoding\"?>"
            "<a><b>val</b></a>",
            "a/b/text()"),
        "val");
    VELOX_ASSERT_USER_THROW(
        xpathString(
            "<?xml version=\"1.0\" encoding=\"ISO-8859-1\"?>"
            "<a><b>\xe9</b></a>",
            "a/b/text()"),
        "Invalid XML document");
  }
  {
    SCOPED_TRACE("A character Reader rejects a leading byte-order mark.");
    VELOX_ASSERT_USER_THROW(
        xpathString("\xef\xbb\xbf<a><b>val</b></a>", "a/b"),
        "Invalid XML document");
    VELOX_ASSERT_USER_THROW(
        xpathBoolean("\xef\xbb\xbf<?xml version=\"1.0\"?><a/>", "a"),
        "Invalid XML document");
    EXPECT_EQ(xpathString("<a>\xef\xbb\xbf</a>", "a"), "\xef\xbb\xbf");
    VELOX_ASSERT_USER_THROW(
        xpathString("\xff\xfe<a/>", "a"), "Invalid XML document");
    VELOX_ASSERT_USER_THROW(
        xpathString("<a>\xc3</a>", "a"), "Invalid XML document");
  }
}

TEST_F(XPathFunctionsTest, documentContext) {
  const std::string xml = "<a><b>v</b></a>";
  EXPECT_EQ(xpathBoolean(xml, "/"), true);
  EXPECT_EQ(xpathString(xml, "/"), "v");
  EXPECT_EQ(xpathString(xml, "name(.)"), "");
  EXPECT_EQ(xpathString(xml, "name(/*)"), "a");
  EXPECT_EQ(xpathString(xml, "name(a/..)"), "");
  EXPECT_EQ(xpathString(xml, "count(//*)"), "2");
  // No wrapper element is visible above the document element.
  EXPECT_EQ(xpathBoolean(xml, "//_r"), false);
  EXPECT_EQ(xpathString(xml, "position()"), "1");
  EXPECT_EQ(xpathString(xml, "last()"), "1");
  EXPECT_EQ(xpathString(" \n<!--lead--><a>v</a> \n", "/node()[1]"), "lead");
  // A union is returned in document order, not operand order.
  EXPECT_EQ(xpathString("<a><b>1</b><c>2</c></a>", "//c | //b"), "1");
}

TEST_F(XPathFunctionsTest, absolutePath) {
  EXPECT_EQ(xpathString("<a><b>abs</b></a>", "/a/b/text()"), "abs");
  EXPECT_EQ(xpathString("<a><b>2</b></a>", "3 * /a/b"), "6");
  EXPECT_EQ(
      xpathString("<a><b>v</b></a>", "/a/b[count(/a/b) = 1]/text()"), "v");
  EXPECT_EQ(xpathString("<a><b>v</b></a>", "concat(/a/b, '/a/b')"), "v/a/b");
}

TEST_F(XPathFunctionsTest, pathSyntax) {
  {
    SCOPED_TRACE("Descendant axis.");
    EXPECT_EQ(xpathString("<a><x><b>deep</b></x></a>", "//b/text()"), "deep");
  }
  {
    SCOPED_TRACE("Wildcard step.");
    EXPECT_EQ(xpathString("<a><x><b>w</b></x></a>", "a/*/b/text()"), "w");
    EXPECT_EQ(xpathString("<a><b>w</b></a>", "*/b/text()"), "w");
  }
  {
    SCOPED_TRACE("Element names that are also XPath operators.");
    EXPECT_EQ(xpathString("<div><b>v</b></div>", "div/b/text()"), "v");
    EXPECT_EQ(xpathString("<a><div><b>v</b></div></a>", "a/div/b/text()"), "v");
    EXPECT_EQ(xpathString("<or><b>v</b></or>", "or/b/text()"), "v");
    EXPECT_EQ(xpathString("<a><mod><b>v</b></mod></a>", "a/mod/b/text()"), "v");
    EXPECT_EQ(
        xpathString("<a><div><b>v</b></div></a>", "a / div / b/text()"), "v");
    EXPECT_EQ(xpathString("<a-><b>v</b></a->", "a-/b/text()"), "v");
    EXPECT_EQ(xpathBoolean("<a><b>1</b></a>", "true() and /a/b = 1"), true);
  }
  {
    SCOPED_TRACE("Slash inside a string literal.");
    EXPECT_EQ(xpathString("<a><b>(/c)</b></a>", "a/b[text()='(/c)']"), "(/c)");
    EXPECT_EQ(
        xpathString("<a><b>x/y</b></a>", "a/b[text()=\"x/y\"]/text()"), "x/y");
  }
}

TEST_F(XPathFunctionsTest, compiledPathOwnership) {
  // A cached expression is compiled against one document's context and then
  // evaluated against others. Results must follow the current document, and a
  // parse failure must not poison the cached path.
  EXPECT_EQ(xpathString("<a><b>first</b></a>", "a/b"), "first");
  EXPECT_EQ(xpathString("<a><b>second</b></a>", "a/b"), "second");
  VELOX_ASSERT_USER_THROW(
      xpathString("not xml", "a/b"), "Invalid XML document");
  EXPECT_EQ(xpathString("<a><b>third</b></a>", "a/b"), "third");
  for (int i = 0; i < 2; ++i) {
    VELOX_ASSERT_USER_THROW(xpathString("<a/>", "missing()"), "Invalid XPath");
  }
}

// ==================== Node types ====================

TEST_F(XPathFunctionsTest, nodeStringValues) {
  EXPECT_EQ(xpathString("<a id=\"x123\">val</a>", "a/@id"), "x123");
  EXPECT_EQ(xpathString("<a b=\"\"/>", "a/@b"), "");
  EXPECT_EQ(xpathString("<a><?pi?></a>", "a/processing-instruction()"), "");
  EXPECT_EQ(xpathString("<?pi?><a/>", "/processing-instruction()"), "");
  EXPECT_EQ(xpathString("<a><!----></a>", "a/comment()"), "");

  // XPath coalesces adjacent text and CDATA, even though the DOM exposes
  // separate nodes. Comments and PIs still separate text nodes.
  EXPECT_EQ(xpathString("<a><![CDATA[x]]>y</a>", "a"), "xy");
  EXPECT_EQ(xpathString("<a><![CDATA[x]]>y</a>", "a/text()"), "xy");
  EXPECT_EQ(xpathString("<a><![CDATA[x]]>y</a>", "a/text()[2]"), "");
  EXPECT_EQ(
      xpathString("<a>x<![CDATA[y]]>z<![CDATA[w]]></a>", "a/text()"), "xyzw");
  EXPECT_EQ(
      xpathString("<a><![CDATA[x]]><![CDATA[y]]></a>", "count(a/text())"), "1");
  EXPECT_EQ(xpathBoolean("<a><![CDATA[x]]>y</a>", "a[text() = 'xy']"), true);
  EXPECT_EQ(xpathBoolean("<a><![CDATA[x]]>y</a>", "a/text()[2]"), false);
  EXPECT_EQ(
      xpathString("<a><![CDATA[x]]><!--c--><![CDATA[y]]></a>", "a/text()[2]"),
      "y");
  EXPECT_EQ(xpathString("<a><![CDATA[x]]><?pi z?>y</a>", "a/text()[2]"), "y");
  EXPECT_EQ(xpathString("<a>x<!--c-->y<?pi z?></a>", "a"), "xy");
  EXPECT_EQ(xpathString("<a>x<!--c-->y<?pi z?></a>", "a/comment()"), "c");
  EXPECT_EQ(
      xpathString("<a>x<!--c-->y<?pi z?></a>", "a/processing-instruction()"),
      "z");
  EXPECT_EQ(xpathString("<a> x </a>", "a"), " x ");
}

TEST_F(XPathFunctionsTest, nestingLimit) {
  const auto nested = [](int depth) {
    std::string xml;
    xml.reserve(depth * 7 + 1);
    for (int i = 0; i < depth; ++i) {
      xml += "<a>";
    }
    xml += "v";
    for (int i = 0; i < depth; ++i) {
      xml += "</a>";
    }
    return xml;
  };

  EXPECT_EQ(xpathString(nested(32), "//a[not(a)]"), "v");
  // libxml2 accepts 256 nested elements by default and rejects more, as
  // documented. The limit is not configurable here because HUGE is unset.
  EXPECT_EQ(xpathString(nested(256), "//a[not(a)]"), "v");
  VELOX_ASSERT_USER_THROW(
      xpathString(nested(300), "a"), "Invalid XML document");
  VELOX_ASSERT_USER_THROW(
      xpathBoolean(nested(300), "a"), "Invalid XML document");
}

struct XmlErrorCounts {
  int structured{0};
  int generic{0};
};

#if LIBXML_VERSION >= 21200
void countStructuredXmlError(void* context, const xmlError*) {
#else
void countStructuredXmlError(void* context, xmlError*) {
#endif
  ++static_cast<XmlErrorCounts*>(context)->structured;
}

void countGenericXmlError(void* context, const char*, ...) {
  ++static_cast<XmlErrorCounts*>(context)->generic;
}

TEST_F(XPathFunctionsTest, libxmlErrorHandlersAreNotInvoked) {
  // Use a fresh thread so no earlier evaluation in this binary can mask
  // changes to libxml2's thread-local handlers.
  bool isolated = false;
  std::thread worker([&]() {
    xmlInitParser();
    XmlErrorCounts counts;
    const auto oldStructured = xmlStructuredError;
    void* oldStructuredContext = xmlStructuredErrorContext;
    const auto oldGeneric = xmlGenericError;
    void* oldGenericContext = xmlGenericErrorContext;
    xmlSetStructuredErrorFunc(&counts, countStructuredXmlError);
    xmlSetGenericErrorFunc(&counts, countGenericXmlError);
    try {
      evalStringDirect("<a/>", "a");
      try {
        evalStringDirect("not xml", "a");
      } catch (const VeloxUserError&) {
      }
      try {
        evalStringDirect("<a/>", "///[");
      } catch (const VeloxUserError&) {
      }
      isolated = counts.structured == 0 && counts.generic == 0 &&
          xmlStructuredError == countStructuredXmlError &&
          xmlStructuredErrorContext == &counts &&
          xmlGenericError == countGenericXmlError &&
          xmlGenericErrorContext == &counts;
    } catch (...) {
      isolated = false;
    }
    xmlSetStructuredErrorFunc(oldStructuredContext, oldStructured);
    xmlSetGenericErrorFunc(oldGenericContext, oldGeneric);
  });
  worker.join();
  EXPECT_TRUE(isolated);
}

TEST_F(XPathFunctionsTest, libxmlLastErrorIsRestored) {
  // Use a fresh thread so the thread-local last error starts clean.
  bool restored = false;
  std::thread worker([&]() {
    xmlInitParser();
    xmlResetLastError();
    try {
      if (evalStringDirect("<a>v</a>", "a") != "v" ||
          xmlGetLastError() != nullptr) {
        return;
      }
      xmlReadMemory(
          "<", 1, nullptr, "UTF-8", XML_PARSE_NOERROR | XML_PARSE_NOWARNING);
      const xmlError* seeded = xmlGetLastError();
      if (seeded == nullptr) {
        return;
      }
      const xmlError before = *seeded;
      const std::string text =
          before.message == nullptr ? "" : std::string(before.message);
      // The caller's error is restored in place, not copied: copying can
      // truncate the message or fail to allocate.
      const auto unchanged = [&]() {
        const xmlError* after = xmlGetLastError();
        return after == seeded && after->domain == before.domain &&
            after->code == before.code && after->level == before.level &&
            after->line == before.line && after->message == before.message &&
            (after->message == nullptr || text == after->message);
      };
      restored = evalStringDirect("<b>w</b>", "b") == "w" && unchanged();
      for (const auto& [xml, path] :
           std::vector<std::pair<std::string, std::string>>{
               {"not xml", "a"}, {"<a/>", "a["}, {"<a/>", "missing()"}}) {
        bool threw = false;
        try {
          evalStringDirect(xml, path);
        } catch (const VeloxUserError&) {
          threw = true;
        }
        restored = restored && threw && unchanged();
      }
    } catch (...) {
      restored = false;
    }
    xmlResetLastError();
  });
  worker.join();
  EXPECT_TRUE(restored);
}

TEST_F(XPathFunctionsTest, numberArgumentsAreStringified) {
  // Each expression puts a number whose string form differs between libxml2
  // ("1e-07", 15 significant digits) and XPath decimal notation
  // ("0.0000001", 16 digits) where the wrapper has to convert it: first,
  // middle, and last argument positions of all twelve wrapped functions.
  const std::string tiny = "1 div 10000000";
  const std::string third = "1 div 3";
  const std::vector<std::pair<std::string, std::string>> cases = {
      {"string(" + third + ")", "0.3333333333333333"},
      {"concat('a', " + tiny + ", 'b')", "a0.0000001b"},
      {"concat('a', " + third + ", 'b')", "a0.3333333333333333b"},
      {"concat(" + tiny + ", 'b', " + tiny + ")", "0.0000001b0.0000001"},
      {"contains('a0.0000001b', " + tiny + ")", "true"},
      {"contains(" + third + ", '0.3333333333333333')", "true"},
      {"starts-with('0.0000001x', " + tiny + ")", "true"},
      {"starts-with(" + third + ", '0.3333333333333333')", "true"},
      {"string-length(" + third + ")", "18"},
      {"substring(" + tiny + ", 3, 3)", "000"},
      // Only the first argument of substring is a string. Stringifying the
      // length or the start would turn these into NaN.
      {"substring('12345', 2, 1 div 0)", "2345"},
      {"substring('12345', -1 div 0)", "12345"},
      {"substring-before('ab0.0000001', " + tiny + ")", "ab"},
      {"substring-before(" + third + ", '3')", "0."},
      {"substring-after('0.0000001tail', " + tiny + ")", "tail"},
      {"substring-after('0.3333333333333333 tail', " + third + ")", " tail"},
      {"normalize-space(" + third + ")", "0.3333333333333333"},
      {"translate('0.0000001', '0.', " + tiny + ")", "0.0000001"},
      {"translate('0.0000001', " + tiny + ", '')", ""},
      {"translate(" + tiny + ", '0', 'Z')", "Z.ZZZZZZ1"},
  };
  // The second pass evaluates cached compiled expressions, which must not
  // have been changed by the first.
  for (int pass = 0; pass < 2; ++pass) {
    for (const auto& [path, expected] : cases) {
      SCOPED_TRACE(path);
      EXPECT_EQ(xpathString("<a/>", path), expected);
    }
  }

  const std::string idXml =
      "<!DOCTYPE a [<!ATTLIST b id ID #REQUIRED>]>"
      "<a><b id=\"Infinity\">value</b></a>";
  EXPECT_EQ(xpathString(idXml, "id(1 div 0)"), "value");
  EXPECT_EQ(xpathString(idXml, "id(-1 div 0)"), "");

  // libxml2 would look up "1e-07" here.
  const std::string tinyIdXml =
      "<!DOCTYPE a [<!ATTLIST b id ID #REQUIRED>]>"
      "<a><b id=\"0.0000001\">value</b></a>";
  EXPECT_EQ(xpathString(tinyIdXml, "id(" + tiny + ")"), "value");
  EXPECT_EQ(xpathString(tinyIdXml, "id(" + third + ")"), "");

  const std::string langXml = "<a xml:lang='0.0000001'><b>hit</b></a>";
  EXPECT_EQ(xpathString(langXml, "a/b[lang(" + tiny + ")]"), "hit");
  EXPECT_EQ(xpathBoolean(langXml, "a/b[lang(" + tiny + ")]"), true);
  EXPECT_EQ(xpathBoolean(langXml, "a/b[lang(" + third + ")]"), false);
}

TEST_F(XPathFunctionsTest, functionsWithoutArgumentsUseTheContextNode) {
  // With no arguments nothing is converted and the context node is used.
  EXPECT_EQ(xpathString("<a>ab</a>", "string()"), "ab");
  EXPECT_EQ(xpathString("<a>ab</a>", "string-length()"), "2");
  EXPECT_EQ(xpathString("<a> x  y </a>", "normalize-space()"), "x y");
}

TEST_F(XPathFunctionsTest, numbersStayNumbersOutsideStringContexts) {
  // The string-conversion wrappers must not turn genuinely numeric operands
  // into strings.
  const std::string xml = "<a><b>1</b><b>2</b></a>";
  EXPECT_EQ(xpathString(xml, "sum(a/b)"), "3");
  EXPECT_EQ(xpathString(xml, "floor(7 div 2)"), "3");
  EXPECT_EQ(xpathString(xml, "ceiling(7 div 2)"), "4");
  EXPECT_EQ(xpathString(xml, "a/b[2] * 4"), "8");
  EXPECT_EQ(xpathString(xml, "1 div 10000000 > 0"), "true");
  EXPECT_EQ(xpathString(xml, "boolean(1 div 10000000)"), "true");
  EXPECT_EQ(xpathString(xml, "string(a/b[2] + a/b[1]) + 1"), "4");
}

// ==================== Conversion semantics ====================

TEST_F(XPathFunctionsTest, adapterMatchesXPathConversionFunctions) {
  // xpath_string(P) must equal string(P) and xpath_boolean(P) must equal
  // boolean(P) for every result type: node-sets of each node kind, numbers,
  // strings, and booleans. Each expectation is also spelled out literally so
  // the test cannot pass merely because both sides are wrong together.
  struct Case {
    std::string xml;
    std::string path;
    std::string asString;
    bool asBoolean;
  };
  const std::vector<Case> cases = {
      {"<a><b>v</b></a>", "a/b", "v", true},
      {"<a><b>v</b></a>", "a/c", "", false},
      {"<a><b>v</b></a>", "/", "v", true},
      {"<a><b>v</b></a>", "//b", "v", true},
      {"<a><b>v</b></a>", "/a/b/text()", "v", true},
      {"<a id=\"i\"/>", "a/@id", "i", true},
      {"<a><!--c--></a>", "a/comment()", "c", true},
      {"<a/>", "1 div 4", "0.25", true},
      {"<a/>", "''", "", false},
      {"<a><b>false</b></a>", "a/b", "false", true},
      {"<a><b>0</b></a>", "a/b", "0", true},
      {"<a/>", "0 div 0", "NaN", false},
      {"<a/>", "-1 div 0", "-Infinity", true},
      {"<a><b>1</b></a>", "a/b = 1", "true", true},
      {"<a><b>1</b></a>", "a/b = 2", "false", false},
      {"<a><b/></a>", "a/b", "", true},
      {"<a><b>b1</b><b>b2</b></a>", "a/b", "b1", true},
      {"<a><b>b1</b><b>b2</b></a>", "a/b[2]", "b2", true},
      {"<a><b>b1</b></a>", "a/b[2]", "", false},
      {"<a>1</a>", "a + 1", "2", true},
      {"<a/>", "count(a)", "1", true},
      {"<a/>", "count(a/b)", "0", false},
      {"<a>0</a>", "a", "0", true},
      {"<a>0</a>", "number(a)", "0", false},
      {"<a/>", "not(a/b)", "true", true},
      {"<a><b/></a>", "not(a/b)", "false", false},
      {"<a>x<b>y</b></a>", "//*", "xy", true},
      {"<a><b>7</b></a>", "a/b * 2", "14", true},
  };
  for (const auto& c : cases) {
    SCOPED_TRACE(c.xml + " | " + c.path);
    expectBoth(c.xml, c.path, c.asString, c.asBoolean);
    EXPECT_EQ(xpathString(c.xml, "string(" + c.path + ")"), c.asString);
    EXPECT_EQ(xpathBoolean(c.xml, "boolean(" + c.path + ")"), c.asBoolean);
  }
}

// ==================== External resources and entities ====================

TEST_F(XPathFunctionsTest, externalResourcesAreNeverLoaded) {
  // Every reference below points at a real file. If any loading option were
  // enabled the content would be expanded (changing the result) or, for the
  // malformed file, fail to parse (raising an error).
  const auto directory = common::testutil::TempDirectoryPath::create();
  const std::string marker = "secret-file-content-marker";
  const auto textUri = writeFile(directory->getPath(), "secret.txt", marker);
  const auto malformedUri =
      writeFile(directory->getPath(), "malformed.xml", "<<<not xml");
  const auto dtdUri = writeFile(
      directory->getPath(), "ext.dtd", "<!ATTLIST b id ID #REQUIRED>");

  const std::string idBody = "<a><b id=\"n1\">value</b></a>";
  const auto general = [](const std::string& uri) {
    return "<!DOCTYPE a [<!ENTITY x SYSTEM \"" + uri + "\">]>";
  };
  {
    SCOPED_TRACE("External general entity in text is retained, not expanded.");
    for (const auto& uri : {textUri, malformedUri}) {
      const auto xml = general(uri) + "<a>&x;</a>";
      EXPECT_EQ(xpathString(xml, "a"), std::nullopt);
      EXPECT_EQ(xpathBoolean(xml, "a"), std::nullopt);
    }
  }
  {
    // An external entity in an attribute value is a well-formedness error.
    SCOPED_TRACE("External general entity in an attribute is rejected.");
    const auto xml = general(textUri) + "<a b=\"&x;\"/>";
    expectUserErrorBoth(xml, "a/@b", "Invalid XML document");
  }
  {
    SCOPED_TRACE("External parameter entity is not loaded.");
    const auto xml =
        "<!DOCTYPE a [<!ENTITY % pe SYSTEM \"" + dtdUri + "\"> %pe;]>" + idBody;
    EXPECT_EQ(xpathString(xml, "id('n1')"), "");
    EXPECT_EQ(xpathBoolean(xml, "id('n1')"), false);
    EXPECT_EQ(xpathString(xml, "a/b"), "value");
  }
  {
    SCOPED_TRACE("External DTD subset is not loaded.");
    const auto xml = "<!DOCTYPE a SYSTEM \"" + dtdUri + "\">" + idBody;
    EXPECT_EQ(xpathString(xml, "id('n1')"), "");
    EXPECT_EQ(xpathBoolean(xml, "id('n1')"), false);
    EXPECT_EQ(xpathString(xml, "a/b"), "value");
  }
  {
    // Control for the two cases above: the same declaration in the internal
    // subset is honoured, so an empty id() result really means "not loaded".
    SCOPED_TRACE("Internal ATTLIST makes id() work.");
    const std::string xml =
        "<!DOCTYPE a [<!ATTLIST b id ID #REQUIRED>]>" + idBody;
    EXPECT_EQ(xpathString(xml, "id('n1')"), "value");
    EXPECT_EQ(xpathBoolean(xml, "id('n1')"), true);
  }
  {
    SCOPED_TRACE("XInclude is not processed.");
    const std::string include =
        "<xi:include href=\"" + textUri + "\" parse=\"text\"/>";
    const auto xml =
        "<a xmlns:xi=\"http://www.w3.org/2001/XInclude\">" + include + "</a>";
    EXPECT_EQ(xpathString(xml, "string(a)"), "");
    EXPECT_EQ(xpathString(xml, "count(a/*)"), "1");
    EXPECT_EQ(xpathBoolean(xml, "contains(string(a), 'secret')"), false);
  }
}

TEST_F(XPathFunctionsTest, entityReferencesAnywhereInTheTreeReturnNull) {
  // A retained entity reference makes the result NULL wherever it sits: as
  // the only child, a later sibling, deep in the tree, or inside an attribute
  // value, including when it is followed by unrelated elements.
  const std::string dtd = "<!DOCTYPE a [<!ENTITY e \"v\">]>";
  for (const auto& body :
       {"<a>&e;</a>",
        "<a><b/>&e;</a>",
        "<a>&e;<b/></a>",
        "<a><b><c><d>&e;</d></c></b></a>",
        "<a><b/><c x=\"&e;\"/></a>",
        "<a><b><c x=\"1\" y=\"&e;\"/></b></a>",
        "<a><b><c/></b><d>&e;</d></a>",
        "<a><b/><c/>&e;</a>",
        "<a x=\"&e;\"/>",
        "<a x=\"p&e;q\"/>"}) {
    SCOPED_TRACE(body);
    for (const auto& path : {"a", "a/b", "count(a/*)", "true()", "a/@x"}) {
      SCOPED_TRACE(path);
      EXPECT_EQ(xpathString(dtd + body, path), std::nullopt);
      EXPECT_EQ(xpathBoolean(dtd + body, path), std::nullopt);
    }
  }
}

TEST_F(XPathFunctionsTest, entityLikeTextThatIsNotAReferenceIsEvaluated) {
  const std::string dtd = "<!DOCTYPE a [<!ENTITY e \"v\">]>";
  {
    SCOPED_TRACE("CDATA, comments and processing instructions keep &e; inert.");
    EXPECT_EQ(xpathString(dtd + "<a><![CDATA[&e;]]></a>", "a"), "&e;");
    EXPECT_EQ(xpathString(dtd + "<a><!--&e;-->ok</a>", "a"), "ok");
    EXPECT_EQ(xpathString(dtd + "<a><?p &e;?>ok</a>", "a"), "ok");
    EXPECT_EQ(xpathString(dtd + "<a><!--&e;--></a>", "a/comment()"), "&e;");
    EXPECT_EQ(xpathBoolean(dtd + "<a><![CDATA[&e;]]></a>", "a"), true);
  }
  {
    SCOPED_TRACE("An escaped ampersand is text, not a reference.");
    EXPECT_EQ(xpathString(dtd + "<a>&#38;e;</a>", "a"), "&e;");
    EXPECT_EQ(xpathString(dtd + "<a x=\"&#38;e;\"/>", "a/@x"), "&e;");
  }
  {
    SCOPED_TRACE("Predefined entities and character references expand.");
    EXPECT_EQ(xpathString("<a>&lt;&amp;&gt;&quot;&apos;</a>", "a"), "<&>\"'");
    EXPECT_EQ(xpathString("<a b=\"&lt;&amp;&quot;&#65;\"/>", "a/@b"), "<&\"A");
    EXPECT_EQ(
        xpathString("<a>&#65;&#x42;&#x1F600;</a>", "a"), "AB\xf0\x9f\x98\x80");
    EXPECT_EQ(xpathString(dtd + "<a>&lt;&#65;</a>", "a"), "<A");
  }
  {
    SCOPED_TRACE("An entity declaration that is never used is harmless.");
    EXPECT_EQ(xpathString(dtd + "<a>plain</a>", "a"), "plain");
    EXPECT_EQ(xpathBoolean(dtd + "<a/>", "a"), true);
  }
}

TEST_F(XPathFunctionsTest, entityAmplificationIsRejected) {
  // Exponential entity expansion must be refused rather than expanded.
  std::string dtd = "<!DOCTYPE a [<!ENTITY e0 \"xxxxxxxxxx\">";
  for (int i = 1; i <= 8; ++i) {
    const auto previous = "&e" + std::to_string(i - 1) + ";";
    std::string value;
    for (int n = 0; n < 10; ++n) {
      value += previous;
    }
    dtd += "<!ENTITY e" + std::to_string(i) + " \"" + value + "\">";
  }
  dtd += "]>";
  expectUserErrorBoth(dtd + "<a>&e8;</a>", "a", "Invalid XML document");
  // A shallow chain is within the limit and, being a retained reference,
  // yields NULL.
  EXPECT_EQ(xpathString(dtd + "<a>&e1;</a>", "a"), std::nullopt);
}

TEST_F(XPathFunctionsTest, dtdAttributeDefaultsAreNotApplied) {
  // DTDATTR is not set, so the DTD never adds attributes: a declared default,
  // including a #FIXED one, is invisible while an explicit value is kept.
  const std::string defaulted = "<!DOCTYPE a [<!ATTLIST a x CDATA \"def\">]>";
  EXPECT_EQ(xpathString(defaulted + "<a/>", "a/@x"), "");
  EXPECT_EQ(xpathBoolean(defaulted + "<a/>", "a/@x"), false);
  EXPECT_EQ(xpathString(defaulted + "<a/>", "count(a/@*)"), "0");
  EXPECT_EQ(xpathString(defaulted + "<a x=\"given\"/>", "a/@x"), "given");
  EXPECT_EQ(xpathString(defaulted + "<a x=\"given\"/>", "count(a/@*)"), "1");
  EXPECT_EQ(
      xpathBoolean(
          "<!DOCTYPE a [<!ATTLIST a x CDATA #FIXED \"fx\">]><a/>", "a/@x"),
      false);
}

// ==================== Size limits ====================

TEST_F(XPathFunctionsTest, hugeTextNodeIsRejected) {
  // libxml2 caps a single text node at 10 MB unless HUGE is set, and HUGE is
  // deliberately not set.
  const std::string xml = "<a>" + std::string(10'001'000, 'x') + "</a>";
#if LIBXML_VERSION >= 21300
  expectUserErrorBoth(xml, "a", "Invalid XML document");
#else
  // Older versions may report the limit as an allocation failure, which is
  // raised as an internal error.
  EXPECT_THROW(xpathString(xml, "a"), VeloxException);
  EXPECT_THROW(xpathBoolean(xml, "a"), VeloxException);
#endif
  // A large but permitted text node is handled.
  const std::string large = "<a>" + std::string(1'000'000, 'y') + "</a>";
  EXPECT_EQ(xpathString(large, "string-length(a)"), "1000000");
  EXPECT_EQ(xpathBoolean(large, "a"), true);
}

// ==================== Null, empty, and NUL inputs ====================

TEST_F(XPathFunctionsTest, nullAndEmptyInputsSkipEvaluation) {
  // A NULL or empty argument produces NULL before any parsing or compiling, so
  // invalid content in the other argument cannot raise an error.
  const auto input = makeRowVector({
      makeNullableFlatVector<std::string>(
          {std::nullopt,
           "",
           "not xml",
           "",
           std::nullopt,
           "<a/>",
           "<a/>",
           std::nullopt,
           "",
           std::string("<a/>\0x", 6)}),
      makeNullableFlatVector<std::string>(
          {"a[",
           "a[",
           "",
           std::nullopt,
           std::nullopt,
           "",
           std::nullopt,
           std::string("a\0b", 3),
           std::string("a\0b", 3),
           ""}),
  });
  velox::test::assertEqualVectors(
      makeAllNullFlatVector<std::string>(input->size()),
      evaluate("xpath_string(c0, c1)", input));
  velox::test::assertEqualVectors(
      makeAllNullFlatVector<bool>(input->size()),
      evaluate("xpath_boolean(c0, c1)", input));
}

TEST_F(XPathFunctionsTest, embeddedNulInDocument) {
  // libxml2 treats NUL as the end of its input, which would silently ignore
  // the rest of the document, so any NUL anywhere is rejected.
  const std::vector<std::string> documents = {
      std::string("<a/>\0", 5),
      std::string("<a>\0</a>", 8),
      std::string("<a\0/>", 5),
      std::string("\0<a/>", 5),
      std::string("<a/>\0ignored", 12),
      std::string("\0", 1),
  };
  for (size_t i = 0; i < documents.size(); ++i) {
    SCOPED_TRACE(i);
    expectUserErrorBoth(documents[i], "a", "Invalid XML document");
  }
}

TEST_F(XPathFunctionsTest, embeddedNulInPath) {
  // A NUL would truncate the path handed to libxml2, so the path is refused
  // and shown with the NUL replaced.
  const std::vector<std::pair<std::string, std::string>> cases = {
      {std::string("a\0b", 3), "Invalid XPath: a?b"},
      {std::string("a\0", 2), "Invalid XPath: a?"},
      {std::string("\0", 1), "Invalid XPath: ?"},
      {std::string("a/b\0[", 5), "Invalid XPath: a/b?["},
  };
  for (const auto& [path, expected] : cases) {
    SCOPED_TRACE(expected);
    EXPECT_EQ(
        userErrorMessage([&]() { evalStringDirect("<a><b/></a>", path); }),
        expected);
    EXPECT_EQ(
        userErrorMessage([&]() { evalBooleanDirect("<a><b/></a>", path); }),
        expected);
    expectUserErrorBoth("<a><b/></a>", path, "Invalid XPath");
  }
  // The truncated prefix is itself a valid path, so accepting the NUL would
  // have produced a result instead of an error.
  EXPECT_EQ(xpathString("<a>v</a>", "a"), "v");
  expectUserErrorBoth("<a>v</a>", std::string("a\0junk", 6), "Invalid XPath");
  // With both arguments bad, some user error is raised; which one is not
  // specified.
  expectUserErrorBoth("not xml", std::string("a\0", 2), "Invalid X");
}

TEST_F(XPathFunctionsTest, notWellFormedDocumentsAreRejected) {
  const std::vector<std::string> documents = {
      " ",
      "\n",
      "\t\r\n ",
      "text",
      "<",
      "<<a/>",
      "<a>",
      "</a>",
      "<a></b>",
      "<a/>x",
      "x<a/>",
      "<a/><b/>",
      "<a b=1/>",
      "<a b=\"1\" b=\"2\"/>",
      "<a><!-- -- --></a>",
      "<a>&undeclared-without-semicolon</a>",
      " <?xml version=\"1.0\"?><a/>",
      "<?xml version=\"1.0\"?><?xml version=\"1.0\"?><a/>",
  };
  for (size_t i = 0; i < documents.size(); ++i) {
    SCOPED_TRACE(documents[i]);
    expectUserErrorBoth(documents[i], "a", "Invalid XML document");
  }
  // Surrounding whitespace and trailing misc are fine.
  EXPECT_EQ(xpathBoolean("<a/>\n<!--t-->\n", "a"), true);
  EXPECT_EQ(xpathBoolean(" \n<a/> \n", "a"), true);
  EXPECT_EQ(xpathBoolean("<a/><?p d?>", "a"), true);
}

// ==================== Encodings and character data ====================

TEST_F(XPathFunctionsTest, invalidUtf8IsRejected) {
  const std::vector<std::string> sequences = {
      "\xc0\xaf",
      "\xc1\xbf",
      "\xe0\x80\x80",
      "\xed\xa0\x80",
      "\xf4\x90\x80\x80",
      "\xf8\x88\x80\x80\x80",
      "\x80",
      "\xe2\x82",
      "\xef\xbf\xbe",
      "\x01",
      "\x1f",
  };
  for (size_t i = 0; i < sequences.size(); ++i) {
    SCOPED_TRACE(i);
    expectUserErrorBoth(
        "<a>" + sequences[i] + "</a>", "a", "Invalid XML document");
  }
}

TEST_F(XPathFunctionsTest, nonUtf8DocumentsAreRejected) {
  // UTF-16 bytes, with or without a byte-order mark, are not UTF-8 text.
  expectUserErrorBoth("\xfe\xff<a/>", "a", "Invalid XML document");
  expectUserErrorBoth(
      std::string("<\0a\0/\0>\0", 8), "a", "Invalid XML document");
  expectUserErrorBoth(
      std::string("\xff\xfe<\0a\0/\0>\0", 10), "a", "Invalid XML document");
  // A byte-order mark is only tolerated as character data, never before the
  // first markup, even after whitespace.
  expectUserErrorBoth(" \xef\xbb\xbf<a/>", "a", "Invalid XML document");
  expectUserErrorBoth(
      "\xef\xbb\xbf<?xml version=\"1.0\"?><a/>", "a", "Invalid XML document");
}

TEST_F(XPathFunctionsTest, invalidCharacterReferencesAreRejected) {
  for (const auto& reference :
       {"&#0;", "&#xD800;", "&#xFFFE;", "&#1;", "&#x110000;", "&#x;", "&#;"}) {
    SCOPED_TRACE(reference);
    expectUserErrorBoth(
        std::string("<a>") + reference + "</a>", "a", "Invalid XML document");
  }
}

TEST_F(XPathFunctionsTest, unicodeContentRoundTrips) {
  const std::string e = "\xc3\xa9";
  const std::string euro = "\xe2\x82\xac";
  const std::string emoji = "\xf0\x9f\x98\x80";
  const std::string all = e + euro + emoji;
  {
    SCOPED_TRACE("Text, attribute, comment, and processing instruction.");
    EXPECT_EQ(xpathString("<a>" + all + "</a>", "a"), all);
    EXPECT_EQ(xpathString("<a b=\"" + all + "\"/>", "a/@b"), all);
    EXPECT_EQ(xpathString("<a><!--" + all + "--></a>", "a/comment()"), all);
    EXPECT_EQ(
        xpathString("<a><?p " + all + "?></a>", "a/processing-instruction()"),
        all);
    EXPECT_EQ(xpathString("<a><![CDATA[" + all + "]]></a>", "a"), all);
  }
  {
    SCOPED_TRACE("Non-ASCII names.");
    EXPECT_EQ(xpathString("<a><" + e + ">v</" + e + "></a>", "a/" + e), "v");
    EXPECT_EQ(xpathBoolean("<a><" + e + "/></a>", "a/" + e), true);
    EXPECT_EQ(xpathBoolean("<a><" + e + "/></a>", "a/b"), false);
    EXPECT_EQ(xpathString("<a " + e + "=\"x\"/>", "a/@" + e), "x");
  }
  {
    SCOPED_TRACE("Non-ASCII string literals and functions.");
    EXPECT_EQ(xpathString("<a>" + e + "</a>", "a[. = '" + e + "']"), e);
    EXPECT_EQ(xpathBoolean("<a>" + e + "</a>", "a = '" + e + "'"), true);
    EXPECT_EQ(xpathBoolean("<a>" + e + "</a>", "a = 'e'"), false);
    EXPECT_EQ(xpathString("<a/>", "'" + all + "'"), all);
    EXPECT_EQ(xpathString("<a>" + e + euro + "</a>", "string-length(a)"), "2");
    EXPECT_EQ(xpathString("<a>" + e + euro + "</a>", "substring(a, 2)"), euro);
    EXPECT_EQ(
        xpathBoolean("<a>" + all + "</a>", "contains(a, '" + emoji + "')"),
        true);
  }
  {
    SCOPED_TRACE("Character references decode to UTF-8.");
    EXPECT_EQ(xpathString("<a>&#xE9;&#x20AC;&#x1F600;</a>", "a"), all);
    EXPECT_EQ(xpathString("<a b=\"&#233;\"/>", "a/@b"), e);
  }
}

TEST_F(XPathFunctionsTest, whitespaceNormalization) {
  // Line ends in content are normalized to LF by the XML parser.
  EXPECT_EQ(xpathString("<a>x\r\ny\rz</a>", "a"), "x\ny\nz");
  // Attribute values turn literal whitespace into spaces but keep character
  // references to it.
  EXPECT_EQ(xpathString("<a b=\"x\ny\tz\"/>", "a/@b"), "x y z");
  EXPECT_EQ(xpathString("<a b=\"x&#10;y&#9;z\"/>", "a/@b"), "x\ny\tz");
  // Whitespace inside an element is data, including around child elements.
  EXPECT_EQ(xpathString("<a> <b/> </a>", "a"), "  ");
  EXPECT_EQ(xpathString("<a> <b/> </a>", "count(a/text())"), "2");
  EXPECT_EQ(xpathBoolean("<a> </a>", "a/text()"), true);
  EXPECT_EQ(xpathString("<a>  x \n y </a>", "normalize-space(a)"), "x y");
}

// ==================== Node kinds, root, and namespaces ====================

TEST_F(XPathFunctionsTest, nodeKindSelection) {
  EXPECT_EQ(xpathString("<a>x<!--c-->y</a>", "count(a/text())"), "2");
  EXPECT_EQ(xpathString("<a>x<!--c-->y</a>", "count(a/node())"), "3");
  EXPECT_EQ(
      xpathString("<a><?p d?><?q e?></a>", "count(a/processing-instruction())"),
      "2");
  EXPECT_EQ(
      xpathString("<a><?p d?><?q e?></a>", "a/processing-instruction('q')"),
      "e");
  expectBoth("<a><?p d?></a>", "a/processing-instruction('z')", "", false);
  // The XML declaration is not a processing instruction or a node.
  expectBoth(
      "<?xml version=\"1.0\"?><a/>", "/processing-instruction()", "", false);
  EXPECT_EQ(xpathString("<?xml version=\"1.0\"?><a/>", "count(/node())"), "1");
  // A DOCTYPE is not a node either.
  EXPECT_EQ(xpathString("<!DOCTYPE a><a/>", "count(/node())"), "1");
  EXPECT_EQ(xpathString("<!--c--><?p d?><a/>", "count(/node())"), "3");
  EXPECT_EQ(xpathString("<a/>\n<!--t-->", "count(/node())"), "2");
}

TEST_F(XPathFunctionsTest, rootAndDocumentNode) {
  const std::string xml = "<a>x<b>y</b></a>";
  expectBoth(xml, ".", "xy", true);
  expectBoth(xml, "/", "xy", true);
  expectBoth(xml, "/*", "xy", true);
  expectBoth(xml, "/a", "xy", true);
  expectBoth(xml, "/a/..", "xy", true);
  expectBoth(xml, "/..", "", false);
  expectBoth(xml, "/text()", "", false);
  expectBoth(xml, "//text()", "x", true);
  expectBoth(xml, "/a/b/ancestor::*", "xy", true);
  EXPECT_EQ(xpathString(xml, "count(/*)"), "1");
  EXPECT_EQ(xpathString(xml, "count(/a/b/ancestor::node())"), "2");
  expectBoth("<!--c--><a/>", "/comment()", "c", true);
  EXPECT_EQ(xpathString("<a>v</a><!--tail-->", "/node()[last()]"), "tail");
  EXPECT_EQ(xpathString("<a>v</a><!--tail-->", "count(/node())"), "2");
}

TEST_F(XPathFunctionsTest, defaultNamespaceAndPrefixedNames) {
  const std::string defaultNs = "<a xmlns=\"urn:x\"><b>v</b></a>";
  // An unprefixed name test never matches a namespaced element.
  expectBoth(defaultNs, "a/b", "", false);
  expectBoth(defaultNs, "/*/*", "v", true);
  EXPECT_EQ(xpathString(defaultNs, "local-name(/*)"), "a");
  EXPECT_EQ(xpathString(defaultNs, "namespace-uri(/*)"), "urn:x");
  EXPECT_EQ(xpathString(defaultNs, "name(/*)"), "a");

  const std::string prefixed = "<x:a xmlns:x=\"urn:x\"><x:b>v</x:b></x:a>";
  EXPECT_EQ(xpathString(prefixed, "name(/*)"), "x:a");
  EXPECT_EQ(xpathString(prefixed, "local-name(/*/*)"), "b");
  EXPECT_EQ(xpathString(prefixed, "namespace-uri(/*/*)"), "urn:x");
  expectBoth(prefixed, "/*/*[local-name()='b']", "v", true);
  expectBoth(prefixed, "/*/*[namespace-uri()='urn:x']", "v", true);
  // A prefix in the path is not bound, so evaluation yields NULL, not an error
  // and not a wrong match.
  expectBoth(prefixed, "x:a/x:b", std::nullopt, std::nullopt);
  expectBoth(prefixed, "/*/x:b", std::nullopt, std::nullopt);
  expectBoth(prefixed, "f:g()", std::nullopt, std::nullopt);
  expectBoth(prefixed, "$p:v", std::nullopt, std::nullopt);
  const auto input = makeRowVector({
      makeFlatVector<std::string>({prefixed, prefixed, "<a>ok</a>"}),
      makeFlatVector<std::string>({"f:g()", "$p:v", "a"}),
  });
  velox::test::assertEqualVectors(
      makeNullableFlatVector<std::string>({std::nullopt, std::nullopt, "ok"}),
      evaluate("xpath_string(c0, c1)", input));
  velox::test::assertEqualVectors(
      makeNullableFlatVector<bool>({std::nullopt, std::nullopt, true}),
      evaluate("xpath_boolean(c0, c1)", input));
}

TEST_F(XPathFunctionsTest, namespaceDeclarationsAreNotAttributes) {
  // Spark's DOM exposes xmlns declarations as attributes; libxml2's data model
  // does not. This is a documented difference.
  EXPECT_EQ(xpathString("<a xmlns:p=\"urn:p\" id=\"1\"/>", "count(a/@*)"), "1");
  EXPECT_EQ(xpathString("<a xmlns=\"urn:d\" id=\"1\"/>", "count(/*/@*)"), "1");
  EXPECT_EQ(xpathString("<a xmlns:p=\"urn:p\"/>", "count(a/@*)"), "0");
  EXPECT_EQ(xpathBoolean("<a xmlns:p=\"urn:p\"/>", "a/@xmlns:p"), std::nullopt);

  const std::string prefixedAttribute = "<a xmlns:p=\"urn:p\" p:id=\"1\"/>";
  EXPECT_EQ(xpathString(prefixedAttribute, "a/@p:id"), std::nullopt);
  EXPECT_EQ(xpathString(prefixedAttribute, "a/@*"), "1");
  EXPECT_EQ(xpathString(prefixedAttribute, "count(a/@*)"), "1");
  EXPECT_EQ(xpathString(prefixedAttribute, "local-name(a/@*)"), "id");
  EXPECT_EQ(xpathString(prefixedAttribute, "name(a/@*)"), "p:id");
  EXPECT_EQ(xpathString(prefixedAttribute, "a/@*[local-name()='id']"), "1");
}

// ==================== Diagnostics ====================

TEST_F(XPathFunctionsTest, invalidXPathMessageIsBounded) {
  // The path is caller-controlled and may be logged, so the message shows at
  // most 128 bytes of valid UTF-8 without control characters, never splits a
  // code point, and marks a cut with "...". Each path starts with '[' so that
  // compilation fails.
  const std::string e = "\xc3\xa9";
  const std::string euro = "\xe2\x82\xac";
  const std::string emoji = "\xf0\x9f\x98\x80";
  const auto brackets = [](size_t count) { return std::string(count, '['); };
  std::string twoByte = "[";
  std::string twoByteShown = "[";
  for (int i = 0; i < 100; ++i) {
    twoByte += e;
    if (i < 63) {
      twoByteShown += e;
    }
  }

  const std::vector<std::pair<std::string, std::string>> cases = {
      {"[a/b", "[a/b"},
      {brackets(128), brackets(128)},
      {brackets(129), brackets(128) + "..."},
      {brackets(4'000), brackets(128) + "..."},
      // A code point that ends exactly at the cap is kept; one that would
      // cross it is dropped whole.
      {brackets(124) + emoji, brackets(124) + emoji},
      {brackets(125) + emoji, brackets(125) + "..."},
      {brackets(126) + e, brackets(126) + e},
      {brackets(127) + e, brackets(127) + "..."},
      {brackets(126) + euro, brackets(126) + "..."},
      {twoByte, twoByteShown + "..."},
      {"[" + e + euro + emoji, "[" + e + euro + emoji},
      // Control characters and invalid or overlong sequences become '?'.
      {"[a\nb\tc\x7f"
       "d",
       "[a?b?c?d"},
      {"[a\xc2\x85"
       "b",
       "[a?b"},
      {"[a\xe2\x80\xa8"
       "b",
       "[a?b"},
      {"[a\xe2\x80\xa9"
       "b",
       "[a?b"},
      {"[a\xe2\x80\xae"
       "b",
       "[a?b"},
      {"[a\xe2\x80\x8b"
       "b",
       "[a?b"},
      {"[\xc0\x80", "[??"},
      {"[\xc1\xbf", "[??"},
      {"[\xed\xa0\x80", "[???"},
      {"[\xf4\x90\x80\x80", "[????"},
      {"[\xe2\x82", "[??"},
      {"[\xf0\x9f\x98", "[???"},
      {"[\x80", "[?"},
      {"[" + std::string(200, '\xff'), "[" + std::string(127, '?') + "..."},
  };
  for (const auto& [path, shown] : cases) {
    SCOPED_TRACE(shown);
    const auto message =
        userErrorMessage([&]() { evalStringDirect("<a/>", path); });
    EXPECT_TRUE(
        hasCodeSuffix(message, "Invalid XPath: " + shown + " (libxml2 code "))
        << message;
  }
}

TEST_F(XPathFunctionsTest, userErrorsCarryTheLibxmlCode) {
  // Parser text never reaches the message: it can repeat document bytes,
  // names, or a system identifier from the input.
  const std::string invalidXmlPrefix = "Invalid XML document (libxml2 code ";
  const std::string secret = "secret-document-marker";
  for (const auto& xml :
       {std::string("not xml"),
        std::string("<a>"),
        std::string("<a>\xc3</a>"),
        "<" + secret,
        "<a " + secret + "=\"1\"/><",
        "<a>" + secret + "</b>",
        "<!DOCTYPE a SYSTEM \"file:///" + secret + "\"><"}) {
    SCOPED_TRACE(xml);
    const auto message =
        userErrorMessage([&]() { evalStringDirect(xml, "a"); });
    EXPECT_TRUE(hasCodeSuffix(message, invalidXmlPrefix)) << message;
    EXPECT_EQ(message.find(secret), std::string::npos) << message;
  }
  // Failures detected before parsing have no libxml2 code to report.
  for (const auto& xml :
       {"<a " + secret + "=\"x\"/>" + std::string(1, '\0') + "tail",
        std::string("\xef\xbb\xbf<a/>")}) {
    SCOPED_TRACE(xml);
    EXPECT_EQ(
        userErrorMessage([&]() { evalBooleanDirect(xml, "a"); }),
        "Invalid XML document");
  }

  // Evaluation errors name the path and the libxml2 error class, so a caller
  // can tell an unknown function from a bad argument count or an unbound
  // variable.
  const auto pathMessage = [](const std::string& path, int code) {
    return "Invalid XPath: " + path + " (libxml2 code " + std::to_string(code) +
        ")";
  };
  const auto actual = [](const std::string& path) {
    return userErrorMessage([&]() { evalStringDirect("<a/>", path); });
  };
  EXPECT_EQ(
      actual("missing()"),
      pathMessage("missing()", XML_XPATH_UNKNOWN_FUNC_ERROR));
  EXPECT_EQ(
      actual("$missing"),
      pathMessage("$missing", XML_XPATH_UNDEF_VARIABLE_ERROR));
  EXPECT_EQ(
      actual("contains('x')"),
      pathMessage("contains('x')", XML_XPATH_INVALID_ARITY));
  EXPECT_EQ(actual("1 | 2"), pathMessage("1 | 2", XML_XPATH_INVALID_TYPE));
}

// ==================== TRY, vectors, and registration ====================

TEST_F(XPathFunctionsTest, tryNullsOnlyTheFailingRows) {
  const auto input = makeRowVector({
      makeNullableFlatVector<std::string>(
          {"<a>ok</a>",
           std::string("<a/>\0", 5),
           "\xef\xbb\xbf<a/>",
           "<a/>",
           "<a/>",
           "<a/>",
           "not xml",
           "<a>ok2</a>"}),
      makeNullableFlatVector<std::string>(
          {"a", "a", "a", "missing()", "a[", std::string("a\0b", 3), "a", "a"}),
  });
  velox::test::assertEqualVectors(
      makeNullableFlatVector<std::string>(
          {"ok",
           std::nullopt,
           std::nullopt,
           std::nullopt,
           std::nullopt,
           std::nullopt,
           std::nullopt,
           "ok2"}),
      evaluate("try(xpath_string(c0, c1))", input));
  velox::test::assertEqualVectors(
      makeNullableFlatVector<bool>(
          {true,
           std::nullopt,
           std::nullopt,
           std::nullopt,
           std::nullopt,
           std::nullopt,
           std::nullopt,
           true}),
      evaluate("try(xpath_boolean(c0, c1))", input));

  // Without TRY the same bad rows fail the batch, and the failure does not
  // leave state behind that affects later evaluations.
  const auto bad = makeRowVector({
      makeFlatVector<std::string>({"<a>ok</a>", std::string("<a/>\0", 5)}),
      makeFlatVector<std::string>({"a", "a"}),
  });
  VELOX_ASSERT_USER_THROW(
      evaluate("xpath_string(c0, c1)", bad), "Invalid XML document");
  VELOX_ASSERT_USER_THROW(
      evaluate("xpath_boolean(c0, c1)", bad), "Invalid XML document");
  EXPECT_EQ(xpathString("<a>after</a>", "a"), "after");
}

TEST_F(XPathFunctionsTest, preservesResultsForPartiallySelectedRows) {
  const auto input = makeRowVector({
      makeFlatVector<std::string>(
          {"<a>first</a>", "not xml", "<a>third</a>", "not xml"}),
      makeFlatVector<std::string>({"a", "a[", "a", "a["}),
      makeFlatVector<bool>({true, false, true, false}),
  });
  velox::test::assertEqualVectors(
      makeFlatVector<std::string>({"first", "fallback", "third", "fallback"}),
      evaluate("if(c2, xpath_string(c0, c1), 'fallback')", input));
  velox::test::assertEqualVectors(
      makeFlatVector<bool>({true, false, true, false}),
      evaluate("if(c2, xpath_boolean(c0, c1), false)", input));
}

TEST_F(XPathFunctionsTest, encodingsAndVaryingOutputLengths) {
  // Output lengths straddle the inline-string threshold and include empty and
  // NULL results, so a stale or truncated writer buffer would show up.
  const auto xml = makeNullableFlatVector<std::string>(
      {"<a>a value that is longer than twelve bytes</a>",
       "<a>s</a>",
       "<a/>",
       std::nullopt,
       "<a>another quite long value, not inlined</a>",
       "<a>12345678901</a>",
       "<a>123456789012</a>",
       "<a>1234567890123</a>"});
  const auto path = makeNullableFlatVector<std::string>(
      {"a", "a", "a", "a", "a", "a", "a", "a"});
  const auto rowType = ROW({"c0", "c1"}, {VARCHAR(), VARCHAR()});
  testEncodings(
      makeTypedExpr("xpath_string(c0, c1)", rowType),
      {xml, path},
      makeNullableFlatVector<std::string>(
          {"a value that is longer than twelve bytes",
           "s",
           "",
           std::nullopt,
           "another quite long value, not inlined",
           "12345678901",
           "123456789012",
           "1234567890123"}));
  testEncodings(
      makeTypedExpr("xpath_boolean(c0, c1)", rowType),
      {xml, path},
      makeNullableFlatVector<bool>(
          {true, true, true, std::nullopt, true, true, true, true}));
}

TEST_F(XPathFunctionsTest, constantDocumentWithVaryingPaths) {
  const auto xml = BaseVector::wrapInConstant(
      5, 0, makeFlatVector<std::string>({"<a><b>1</b><c>2</c></a>"}));
  const auto paths = makeFlatVector<std::string>(
      {"a/b", "a/c", "a/d", "count(a/*)", "a/b = 1"});
  const auto input = makeRowVector({xml, paths});
  velox::test::assertEqualVectors(
      makeFlatVector<std::string>({"1", "2", "", "2", "true"}),
      evaluate("xpath_string(c0, c1)", input));
  velox::test::assertEqualVectors(
      makeFlatVector<bool>({true, true, false, true, true}),
      evaluate("xpath_boolean(c0, c1)", input));
}

TEST_F(XPathFunctionsTest, constantPathWithVaryingDocuments) {
  const auto xmls = makeNullableFlatVector<std::string>(
      {"<a><b>1</b></a>", "<a/>", "", std::nullopt, "<a><b>3</b></a>"});
  const auto path =
      BaseVector::wrapInConstant(5, 0, makeFlatVector<std::string>({"a/b"}));
  const auto input = makeRowVector({xmls, path});
  velox::test::assertEqualVectors(
      makeNullableFlatVector<std::string>(
          {"1", "", std::nullopt, std::nullopt, "3"}),
      evaluate("xpath_string(c0, c1)", input));
  velox::test::assertEqualVectors(
      makeNullableFlatVector<bool>(
          {true, false, std::nullopt, std::nullopt, true}),
      evaluate("xpath_boolean(c0, c1)", input));
}

TEST_F(XPathFunctionsTest, registeredSignatures) {
  const auto booleanSignatures = getSignatureStrings("xpath_boolean");
  ASSERT_EQ(1u, booleanSignatures.size());
  EXPECT_EQ(1u, booleanSignatures.count("(varchar,varchar) -> boolean"));

  const auto stringSignatures = getSignatureStrings("xpath_string");
  ASSERT_EQ(1u, stringSignatures.size());
  EXPECT_EQ(1u, stringSignatures.count("(varchar,varchar) -> varchar"));
}

TEST_F(XPathFunctionsTest, concurrentEvaluation) {
  // Every thread owns its compiled-path cache and its libxml2 error state. The
  // distinct literal paths exercise cache churn while other threads parse,
  // evaluate, and fail.
  constexpr int kThreads = 4;
  constexpr int kRounds = 3;
  constexpr int kPathsPerRound = 67;
  std::atomic<int> failures{0};
  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  for (int i = 0; i < kThreads; ++i) {
    threads.emplace_back([i, &failures]() {
      try {
        const std::string id = std::to_string(i);
        const std::string xml = "<a>" + id + "</a>";
        for (int round = 0; round < kRounds; ++round) {
          for (int n = 0; n < kPathsPerRound; ++n) {
            const std::string suffix = "-" + id + "-" + std::to_string(round) +
                "-" + std::to_string(n);
            const auto value =
                evalStringDirect(xml, "concat(a, '" + suffix + "')");
            if (!value || *value != id + suffix) {
              failures.fetch_add(1);
            }
          }
          try {
            evalStringDirect("<a>", "a");
            failures.fetch_add(1);
          } catch (const VeloxUserError&) {
          }
          try {
            evalStringDirect(xml, "a[");
            failures.fetch_add(1);
          } catch (const VeloxUserError& error) {
            if (error.message().find("Invalid XPath: a[") ==
                std::string::npos) {
              failures.fetch_add(1);
            }
          }
          if (evalBooleanDirect(xml, "a = " + id) != true) {
            failures.fetch_add(1);
          }
        }
      } catch (...) {
        failures.fetch_add(1);
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  EXPECT_EQ(failures.load(), 0);
}

} // namespace
} // namespace facebook::velox::functions::sparksql::test
