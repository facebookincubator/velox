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
#include <optional>
#include <string>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/functions/sparksql/tests/SparkFunctionBaseTest.h"

namespace facebook::velox::functions::sparksql::test {
namespace {

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

TEST_F(XPathFunctionsTest, xpathBooleanNullHandling) {
  // NULL inputs produce NULL output.
  EXPECT_EQ(
      evaluateOnce<bool>(
          "xpath_boolean(c0, c1)",
          std::optional<std::string>(std::nullopt),
          std::optional<std::string>("a/b")),
      std::nullopt);
  EXPECT_EQ(
      evaluateOnce<bool>(
          "xpath_boolean(c0, c1)",
          std::optional<std::string>("<a/>"),
          std::optional<std::string>(std::nullopt)),
      std::nullopt);
  // Empty string inputs produce NULL.
  EXPECT_EQ(xpathBoolean("", "a/b"), std::nullopt);
  EXPECT_EQ(xpathBoolean("<a/>", ""), std::nullopt);
}

TEST_F(XPathFunctionsTest, xpathStringBasic) {
  EXPECT_EQ(xpathString("<a><b>b</b><c>cc</c></a>", "a/c"), "cc");
  EXPECT_EQ(xpathString("<a><b>b1</b><b>b2</b></a>", "a/b"), "b1");
}

TEST_F(XPathFunctionsTest, xpathStringNullAndEdge) {
  EXPECT_EQ(xpathString("", "a/b"), std::nullopt);
  // No match returns empty string.
  EXPECT_EQ(xpathString("<a><b>b</b></a>", "a/c"), "");
}

TEST_F(XPathFunctionsTest, xpathStringNonNodeSetResult) {
  // A path that is a boolean/number XPath expression (not a node-set) is cast
  // to its string form via xmlXPathCastToString, matching Spark.
  EXPECT_EQ(xpathString("<a><b>1</b></a>", "a/b = 1"), "true");
  EXPECT_EQ(xpathString("<a><b>1</b></a>", "a/b = 2"), "false");
}

TEST_F(XPathFunctionsTest, invalidXmlThrows) {
  // Spark's UDFXPathUtil throws on malformed XML; the offloaded path matches
  // that by raising a user error instead of returning NULL.
  VELOX_ASSERT_USER_THROW(
      xpathBoolean("not xml", "a/b"), "Invalid XML document");
  VELOX_ASSERT_USER_THROW(
      xpathString("<a><b></a>", "a/b"), "Invalid XML document");
}

TEST_F(XPathFunctionsTest, invalidXPathThrows) {
  // An XPath syntax error throws (Spark's xpath.compile throws
  // RuntimeException), so the offloaded path raises a user error.
  VELOX_ASSERT_USER_THROW(
      xpathBoolean("<a><b>1</b></a>", "///[invalid"), "Invalid XPath");
  VELOX_ASSERT_USER_THROW(
      xpathString("<a><b>1</b></a>", "a/b["), "Invalid XPath");
}

TEST_F(XPathFunctionsTest, namespacePrefixedReturnsNull) {
  // A prefixed step references an unregistered prefix -> evaluation fails ->
  // NULL (Spark, namespace-unaware, would match and return "v").
  EXPECT_EQ(
      xpathString(
          "<x:a xmlns:x=\"http://e\"><x:b>v</x:b></x:a>", "x:a/x:b/text()"),
      std::nullopt);
  // Undeclared prefix in the input parses but still fails evaluation.
  EXPECT_EQ(
      xpathString("<ns:a><ns:b>v</ns:b></ns:a>", "ns:a/ns:b/text()"),
      std::nullopt);
}

TEST_F(XPathFunctionsTest, doctypeHandling) {
  {
    SCOPED_TRACE("DOCTYPE without internal subset is stripped.");
    EXPECT_EQ(
        xpathString("<!DOCTYPE foo><foo><b>val</b></foo>", "foo/b/text()"),
        "val");
  }
  {
    SCOPED_TRACE("DOCTYPE with internal subset (brackets) is stripped.");
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
    // A comment inside the internal subset whose text contains ']' and '>'
    // must be treated as opaque; otherwise the bracket depth is corrupted and
    // the DOCTYPE scan terminates early, mangling the document.
    SCOPED_TRACE("Comment containing ']' and '>' inside the internal subset.");
    EXPECT_EQ(
        xpathString(
            "<!DOCTYPE foo [<!-- ] --><!ENTITY y \"z\">]>"
            "<foo><b>val</b></foo>",
            "foo/b/text()"),
        "val");
  }
  {
    // A processing instruction inside the internal subset containing ']' and
    // '>' is likewise opaque to the DOCTYPE scan.
    SCOPED_TRACE("PI containing ']' and '>' inside the internal subset.");
    EXPECT_EQ(
        xpathString(
            "<!DOCTYPE foo [<?pi ]> ?><!ENTITY y \"z\">]>"
            "<foo><b>val</b></foo>",
            "foo/b/text()"),
        "val");
  }
  {
    // SPARK-DIVERGENCE: XP-BUG-2. The input defines an entity in the internal
    // subset AND references it in the body. Stripping the DOCTYPE removes the
    // definition, so the wrapped re-parse fails on the now-undefined entity and
    // the function returns NULL. Spark expands the internal entity (-> "hi").
    SCOPED_TRACE("Internal entity reference returns NULL (XP-BUG-2).");
    EXPECT_EQ(
        xpathString(
            "<!DOCTYPE foo [<!ENTITY x \"hi\">]><foo><b>&x;</b></foo>",
            "foo/b/text()"),
        std::nullopt);
  }
}

TEST_F(XPathFunctionsTest, prologHandling) {
  {
    // A leading "<?xml-stylesheet ...?>" PI must NOT be mistaken for the XML
    // declaration: only "<?xml" followed by whitespace is the declaration.
    SCOPED_TRACE("xml-stylesheet PI is not treated as the XML declaration.");
    EXPECT_EQ(
        xpathString(
            "<?xml-stylesheet type=\"text/xsl\" href=\"s.xsl\"?>"
            "<a><b>val</b></a>",
            "a/b/text()"),
        "val");
  }
  {
    SCOPED_TRACE("A real XML declaration is stripped before wrapping.");
    EXPECT_EQ(
        xpathString("<?xml version=\"1.0\"?><a><b>val</b></a>", "a/b/text()"),
        "val");
  }
  {
    // A comment in the prolog before the DOCTYPE must not prevent DOCTYPE
    // stripping; otherwise the DOCTYPE would survive into the wrapper and the
    // re-parse would spuriously fail.
    SCOPED_TRACE("Comment before DOCTYPE still allows DOCTYPE stripping.");
    EXPECT_EQ(
        xpathString(
            "<!-- lead --><!DOCTYPE foo><foo><b>val</b></foo>", "foo/b/text()"),
        "val");
  }
  {
    SCOPED_TRACE("PI before DOCTYPE still allows DOCTYPE stripping.");
    EXPECT_EQ(
        xpathString(
            "<?pi data?><!DOCTYPE foo><foo><b>val</b></foo>", "foo/b/text()"),
        "val");
  }
  {
    // SPARK-DIVERGENCE: XP-BUG-2. Stripping the declaration drops
    // encoding="...", so the wrapped re-parse assumes UTF-8. A
    // non-UTF-8-declared document whose bytes are invalid UTF-8 (here a lone
    // 0xE9 under ISO-8859-1) re-parses as malformed -> NULL, even though the
    // raw parse honored the declaration.
    SCOPED_TRACE("Non-UTF-8 declared encoding returns NULL (XP-BUG-2).");
    EXPECT_EQ(
        xpathString(
            "<?xml version=\"1.0\" encoding=\"ISO-8859-1\"?>"
            "<a><b>\xe9</b></a>",
            "a/b/text()"),
        std::nullopt);
  }
}
TEST_F(XPathFunctionsTest, absolutePath) {
  // Absolute paths get rewritten to go through synthetic root.
  EXPECT_EQ(xpathString("<a><b>abs</b></a>", "/a/b/text()"), "abs");
}

TEST_F(XPathFunctionsTest, descendantAxis) {
  // "//" should NOT be rewritten - descendant-or-self axis.
  EXPECT_EQ(xpathString("<a><x><b>deep</b></x></a>", "//b/text()"), "deep");
}

TEST_F(XPathFunctionsTest, wildcardStep) {
  // '*' is the wildcard name-test and must NOT be treated as an absolute-path
  // boundary; the '/' after '*' stays relative (no "*/_r/b" corruption).
  EXPECT_EQ(xpathString("<a><x><b>w</b></x></a>", "a/*/b/text()"), "w");
  EXPECT_EQ(xpathString("<a><b>w</b></a>", "*/b/text()"), "w");
}

TEST_F(XPathFunctionsTest, elementNamedLikeWordOperator) {
  // An element literally named like a word operator (or/and/div/mod) must be
  // treated as a location step, not as an operator. Otherwise the '/' after it
  // is misclassified as an absolute path and the wrapper root is spliced into
  // the middle of the path, silently returning empty/no-match. "div" is the
  // common case (XHTML). Covers the step at the start of the path and mid-path.
  EXPECT_EQ(xpathString("<div><b>v</b></div>", "div/b/text()"), "v");
  EXPECT_EQ(xpathString("<a><div><b>v</b></div></a>", "a/div/b/text()"), "v");
  EXPECT_EQ(xpathString("<or><b>v</b></or>", "or/b/text()"), "v");
  EXPECT_EQ(xpathString("<a><mod><b>v</b></mod></a>", "a/mod/b/text()"), "v");
  // A real binary operator with a left operand still rewrites the absolute
  // operand correctly (regression guard for the legitimate case).
  EXPECT_EQ(xpathBoolean("<a><b>1</b></a>", "true() and /a/b = 1"), true);
}

TEST_F(XPathFunctionsTest, stringLiteralWithSlash) {
  // A '/' inside an XPath string literal must NOT be rewritten as an absolute
  // path. Here the predicate value "(/c)" contains an operator char '(' before
  // '/', which would be mis-rewritten to "(/_r/c)" without quote-awareness,
  // breaking the predicate match.
  EXPECT_EQ(xpathString("<a><b>(/c)</b></a>", "a/b[text()='(/c)']"), "(/c)");
  // Double-quoted literal variant.
  EXPECT_EQ(
      xpathString("<a><b>x/y</b></a>", "a/b[text()=\"x/y\"]/text()"), "x/y");
}

// ==================== Node types ====================

TEST_F(XPathFunctionsTest, attributeNode) {
  EXPECT_EQ(xpathString("<a id=\"x123\">val</a>", "a/@id"), "x123");
}

} // namespace
} // namespace facebook::velox::functions::sparksql::test
