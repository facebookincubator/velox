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

#include "velox/functions/prestosql/MyanmarFunctions.h"
#include <vector>
#include "myanmartools/detector.h"
#include "myanmartools/translit_z2u.h"
#include "velox/functions/Macros.h"
#include "velox/functions/Registerer.h"

namespace facebook::velox::functions {

// Zawgyi probability threshold matching Java implementation
constexpr double ZAWGYI_PROBABILITY_THRESHOLD = 0.9;

template <typename T>
struct MyanmarFontEncodingFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE void call(
      out_type<Varchar>& result,
      const arg_type<Varchar>& input) {
    std::string text(input.data(), input.size());

    // Use myanmar-tools ZawgyiDetector (same as Java)
    static myanmartools::ZawgyiDetector detector;
    double probability = detector.GetZawgyiProbability(text);

    if (probability > ZAWGYI_PROBABILITY_THRESHOLD) {
      result.copy_from("zawgyi");
    } else {
      result.copy_from("unicode");
    }
  }
};

template <typename T>
struct MyanmarNormalizeUnicodeFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE void call(
      out_type<Varchar>& result,
      const arg_type<Varchar>& input) {
    std::string text(input.data(), input.size());

    // Match Java logic: split by newlines, transliterate Zawgyi pieces, rejoin
    std::vector<std::string> pieces;
    std::vector<std::string> normalizedPieces;

    size_t start = 0;
    size_t end = text.find('\n');

    while (end != std::string::npos) {
      pieces.push_back(text.substr(start, end - start));
      start = end + 1;
      end = text.find('\n', start);
    }
    pieces.push_back(text.substr(start));

    // Process each piece using myanmar-tools (same as Java)
    static myanmartools::ZawgyiDetector detector;
    static myanmartools::TranslitZ2U transliterator;

    for (const auto& piece : pieces) {
      double probability = detector.GetZawgyiProbability(piece);
      if (probability > ZAWGYI_PROBABILITY_THRESHOLD) {
        normalizedPieces.push_back(transliterator.Transliterate(piece));
      } else {
        normalizedPieces.push_back(piece);
      }
    }

    // Join with newlines
    std::string normalized;
    for (size_t i = 0; i < normalizedPieces.size(); ++i) {
      if (i > 0) {
        normalized += '\n';
      }
      normalized += normalizedPieces[i];
    }

    result.copy_from(normalized);
  }
};

void registerMyanmarFontEncoding(const std::string& name) {
  registerFunction<MyanmarFontEncodingFunction, Varchar, Varchar>({name});
}

void registerMyanmarNormalizeUnicode(const std::string& name) {
  registerFunction<MyanmarNormalizeUnicodeFunction, Varchar, Varchar>({name});
}

} // namespace facebook::velox::functions
