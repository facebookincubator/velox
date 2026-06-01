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
// org.pcre4j.regex.translate.Evaluator (Java) under Apache-2.0 by the same
// author for inclusion in Velox.
//
#include "velox/functions/lib/java_pcre2_translator/Evaluator.h"

#include "velox/functions/lib/java_pcre2_translator/EvaluationFailedException.h"
#include "velox/functions/lib/java_pcre2_translator/JdkPropertyExpander.h"

#include <string>

namespace facebook::velox::functions::java_pcre2_translator {
namespace {

template <class... Ts>
struct Overloaded : Ts... {
  using Ts::operator()...;
};
template <class... Ts>
Overloaded(Ts...) -> Overloaded<Ts...>;

const RangeSet& digit() {
  static const RangeSet k = RangeSet::range('0', '9');
  return k;
}

const RangeSet& word() {
  static const RangeSet k = RangeSet::range('A', 'Z')
                                .unionWith(RangeSet::range('a', 'z'))
                                .unionWith(RangeSet::range('0', '9'))
                                .unionWith(RangeSet::single('_'));
  return k;
}

const RangeSet& space() {
  static const RangeSet k = RangeSet::single('\t')
                                .unionWith(RangeSet::single('\n'))
                                .unionWith(RangeSet::single(0x0B))
                                .unionWith(RangeSet::single('\f'))
                                .unionWith(RangeSet::single('\r'))
                                .unionWith(RangeSet::single(' '));
  return k;
}

const RangeSet& ascii() {
  static const RangeSet k = RangeSet::range(0x00, 0x7F);
  return k;
}

const RangeSet& alpha() {
  static const RangeSet k =
      RangeSet::range('A', 'Z').unionWith(RangeSet::range('a', 'z'));
  return k;
}

const RangeSet& alnum() {
  static const RangeSet k = alpha().unionWith(digit());
  return k;
}

const RangeSet& lower() {
  static const RangeSet k = RangeSet::range('a', 'z');
  return k;
}

const RangeSet& upper() {
  static const RangeSet k = RangeSet::range('A', 'Z');
  return k;
}

const RangeSet& hexDigit() {
  static const RangeSet k = digit()
                                .unionWith(RangeSet::range('A', 'F'))
                                .unionWith(RangeSet::range('a', 'f'));
  return k;
}

const RangeSet& blank() {
  static const RangeSet k =
      RangeSet::single(' ').unionWith(RangeSet::single('\t'));
  return k;
}

const RangeSet& cntrl() {
  static const RangeSet k =
      RangeSet::range(0x00, 0x1F).unionWith(RangeSet::single(0x7F));
  return k;
}

const RangeSet& graph() {
  static const RangeSet k = RangeSet::range(0x21, 0x7E);
  return k;
}

const RangeSet& print() {
  static const RangeSet k = RangeSet::range(0x20, 0x7E);
  return k;
}

const RangeSet& punct() {
  static const RangeSet k =
      print().subtract(alnum()).subtract(RangeSet::single(' '));
  return k;
}

RangeSet expandProperty(const PropertyLeaf& leaf) {
  const auto& token = leaf.pcre2Token;
  if (token == "\\d") {
    return digit();
  }
  if (token == "\\D") {
    return digit().complement();
  }
  if (token == "\\w") {
    return word();
  }
  if (token == "\\W") {
    return word().complement();
  }
  if (token == "\\s") {
    return space();
  }
  if (token == "\\S") {
    return space().complement();
  }
  if (token == "\\p{ASCII}") {
    return ascii();
  }
  if (token == "\\p{Alpha}") {
    return alpha();
  }
  if (token == "\\p{Alnum}") {
    return alnum();
  }
  if (token == "\\p{Lower}") {
    return lower();
  }
  if (token == "\\p{Upper}") {
    return upper();
  }
  if (token == "\\p{Digit}") {
    return digit();
  }
  if (token == "\\p{XDigit}") {
    return hexDigit();
  }
  if (token == "\\p{Space}") {
    return space();
  }
  if (token == "\\p{Blank}") {
    return blank();
  }
  if (token == "\\p{Cntrl}") {
    return cntrl();
  }
  if (token == "\\p{Graph}") {
    return graph();
  }
  if (token == "\\p{Print}") {
    return print();
  }
  if (token == "\\p{Punct}") {
    return punct();
  }

  auto jdk = JdkPropertyExpander::expand(token);
  if (jdk.has_value()) {
    return *jdk;
  }
  throw EvaluationFailedException("Cannot expand property: " + token);
}

} // namespace

RangeSet Evaluator::toRangeSet(const ClassNode& node) {
  return std::visit(
      Overloaded{
          [](const Literal& lit) { return RangeSet::single(lit.cp); },
          [](const Range& r) { return RangeSet::range(r.lo, r.hi); },
          [](const Negated& neg) {
            return Evaluator::toRangeSet(*neg.child).complement();
          },
          [](const Union& u) {
            RangeSet result = RangeSet::empty();
            for (const auto& child : u.children) {
              result = result.unionWith(Evaluator::toRangeSet(*child));
            }
            return result;
          },
          [](const Intersection& inter) {
            RangeSet result = RangeSet::all();
            for (const auto& operand : inter.operands) {
              result = result.intersect(Evaluator::toRangeSet(*operand));
            }
            return result;
          },
          [](const PropertyLeaf& leaf) { return expandProperty(leaf); }},
      node.value);
}

std::optional<RangeSet> Evaluator::tryToRangeSet(const ClassNode& node) {
  try {
    return toRangeSet(node);
  } catch (const EvaluationFailedException&) {
    return std::nullopt;
  }
}

} // namespace facebook::velox::functions::java_pcre2_translator
