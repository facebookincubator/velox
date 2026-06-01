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
//
// Originally authored by Oleksii PELYKH for pcre4j; ported from
// org.pcre4j.regex.translate.ClassRenderer (Java) under Apache-2.0 by the
// same author for inclusion in Velox.
//
#include "velox/functions/lib/java_pcre2_translator/ClassRenderer.h"

#include "velox/functions/lib/java_pcre2_translator/EvaluationFailedException.h"
#include "velox/functions/lib/java_pcre2_translator/Evaluator.h"

#include <cstdio>
#include <stdexcept>
#include <string>

namespace facebook::velox::functions::java_pcre2_translator {
namespace {

constexpr const char* kEmptyClass = "[^\\x{0}-\\x{10FFFF}]";

template <class... Ts>
struct Overloaded : Ts... {
  using Ts::operator()...;
};
template <class... Ts>
Overloaded(Ts...) -> Overloaded<Ts...>;

void appendCodePointUtf8(std::int32_t cp, std::string& sb) {
  if (cp <= 0x7F) {
    sb.push_back(static_cast<char>(cp));
  } else if (cp <= 0x7FF) {
    sb.push_back(static_cast<char>(0xC0 | (cp >> 6)));
    sb.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else if (cp <= 0xFFFF) {
    sb.push_back(static_cast<char>(0xE0 | (cp >> 12)));
    sb.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    sb.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else {
    sb.push_back(static_cast<char>(0xF0 | (cp >> 18)));
    sb.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
    sb.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    sb.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  }
}

void emitFlat(const ClassNode& node, std::string& sb);
void emitOriginalStyle(const ClassNode& node, std::string& sb);

RangeSet tryEvaluateIntersectionRangeSet(const Intersection& inter, bool& ok) {
  RangeSet result = RangeSet::all();
  for (const auto& operand : inter.operands) {
    auto rs = Evaluator::tryToRangeSet(*operand);
    if (!rs.has_value()) {
      ok = false;
      return RangeSet::empty();
    }
    result = result.intersect(*rs);
  }
  ok = true;
  return result;
}

void emitIntersectionFallbackOriginal(const Intersection& inter, std::string& sb) {
  for (std::size_t i = 0; i < inter.operands.size(); ++i) {
    if (i > 0) {
      sb.append("&&");
    }
    emitOriginalStyle(*inter.operands[i], sb);
  }
}

void emitFlat(const ClassNode& node, std::string& sb) {
  std::visit(
      Overloaded{
          [&](const Literal& lit) { ClassRenderer::emitLiteralInClass(lit.cp, sb); },
          [&](const Range& r) {
            ClassRenderer::emitLiteralInClass(r.lo, sb);
            sb.push_back('-');
            ClassRenderer::emitLiteralInClass(r.hi, sb);
          },
          [&](const PropertyLeaf& leaf) { sb.append(leaf.pcre2Token); },
          [&](const Negated& neg) {
            try {
              sb.append(Evaluator::toRangeSet(*neg.child).complement().toPcre2ClassBody());
            } catch (const EvaluationFailedException& e) {
              throw EvaluationFailedException("Cannot flatten nested [^...]; caller must fall back");
            }
          },
          [&](const Union& u) {
            for (const auto& child : u.children) {
              emitFlat(*child, sb);
            }
          },
          [&](const Intersection&) {
            throw std::logic_error("emitFlat must not be called on Intersection nodes");
          }},
      node.value);
}

void emitOriginalStyle(const ClassNode& node, std::string& sb) {
  std::visit(
      Overloaded{
          [&](const Literal& lit) { ClassRenderer::emitLiteralInClass(lit.cp, sb); },
          [&](const Range& r) {
            ClassRenderer::emitLiteralInClass(r.lo, sb);
            sb.push_back('-');
            ClassRenderer::emitLiteralInClass(r.hi, sb);
          },
          [&](const PropertyLeaf& leaf) { sb.append(leaf.pcre2Token); },
          [&](const Negated& neg) {
            sb.append("[^");
            emitOriginalStyle(*neg.child, sb);
            sb.push_back(']');
          },
          [&](const Union& u) {
            for (const auto& child : u.children) {
              emitOriginalStyle(*child, sb);
            }
          },
          [&](const Intersection& inter) { emitIntersectionFallbackOriginal(inter, sb); }},
      node.value);
}

std::string renderWithIntersection(const ClassNode& inner, bool negated) {
  auto rs = Evaluator::tryToRangeSet(inner);
  if (rs.has_value()) {
    RangeSet effective = negated ? rs->complement() : *rs;
    if (effective.isEmpty()) {
      return kEmptyClass;
    }
    return "[" + effective.toPcre2ClassBody() + "]";
  }

  if (const auto* inter = inner.getIf<Intersection>()) {
    bool ok = false;
    RangeSet operandResult = tryEvaluateIntersectionRangeSet(*inter, ok);
    if (ok) {
      RangeSet effective = negated ? operandResult.complement() : operandResult;
      if (effective.isEmpty()) {
        return kEmptyClass;
      }
      return "[" + effective.toPcre2ClassBody() + "]";
    }
  }

  std::string sb;
  sb.push_back('[');
  if (negated) {
    sb.push_back('^');
  }
  emitOriginalStyle(inner, sb);
  sb.push_back(']');
  return sb;
}

} // namespace

std::string ClassRenderer::render(const ClassNode& node) {
  return renderWithSignal(node).text;
}

ClassRenderer::RenderResult ClassRenderer::renderWithSignal(const ClassNode& node) {
  const bool negated = node.is<Negated>();
  const ClassNode& inner = negated ? *node.getIf<Negated>()->child : node;

  if (containsIntersection(inner)) {
    auto rendered = renderWithIntersection(inner, negated);
    return {rendered, rendered.find("&&") != std::string::npos};
  }

  if (auto rs = Evaluator::tryToRangeSet(inner)) {
    RangeSet effective = negated ? rs->complement() : *rs;
    if (effective.isEmpty()) {
      return {kEmptyClass, false};
    }
  }

  std::string sb;
  sb.push_back('[');
  if (negated) {
    sb.push_back('^');
  }
  try {
    emitFlat(inner, sb);
  } catch (const EvaluationFailedException&) {
    std::string fallback;
    fallback.push_back('[');
    if (negated) {
      fallback.push_back('^');
    }
    emitOriginalStyle(inner, fallback);
    fallback.push_back(']');
    return {fallback, false};
  }
  sb.push_back(']');
  return {sb, false};
}

void ClassRenderer::emitLiteralInClass(std::int32_t cp, std::string& sb) {
  if (cp >= 0x20 && cp <= 0x7E) {
    switch (cp) {
      case '\\':
      case ']':
      case '^':
      case '-':
        sb.push_back('\\');
        sb.push_back(static_cast<char>(cp));
        return;
      default:
        sb.push_back(static_cast<char>(cp));
        return;
    }
  }
  if (cp >= 0xD800 && cp <= 0xDFFF) {
    char buf[96];
    std::snprintf(
        buf,
        sizeof(buf),
        "Lone surrogate U+%04X is not representable in PCRE2 UTF mode",
        static_cast<unsigned>(cp));
    throw std::invalid_argument(buf);
  }
  char buf[16];
  std::snprintf(buf, sizeof(buf), "\\x{%X}", static_cast<unsigned>(cp));
  sb.append(buf);
}

bool ClassRenderer::containsIntersection(const ClassNode& node) {
  return std::visit(
      Overloaded{
          [](const Intersection&) { return true; },
          [](const Negated& neg) { return ClassRenderer::containsIntersection(*neg.child); },
          [](const Union& u) {
            for (const auto& child : u.children) {
              if (ClassRenderer::containsIntersection(*child)) {
                return true;
              }
            }
            return false;
          },
          [](const auto&) { return false; }},
      node.value);
}

} // namespace facebook::velox::functions::java_pcre2_translator
