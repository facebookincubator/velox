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

#include "velox/functions/prestosql/json/SimdJsonExtractor.h"

#include <cctype>
#include <unordered_map>
#include <vector>

#include "boost/algorithm/string/trim.hpp"
#include "folly/String.h"
#include "simdjson.h"
#include "velox/common/base/Exceptions.h"
#include "velox/functions/prestosql/json/JsonPathTokenizer.h"

namespace facebook::velox::functions {

namespace {

class SimdJsonExtractor {
 public:
  // Use this method to get an instance of SimdJsonExtractor given a json path.
  static SimdJsonExtractor& getInstance(folly::StringPiece path) {
    // Pre-process
    auto trimedPath = folly::trimWhitespace(path).str();

    std::shared_ptr<SimdJsonExtractor> op;
    if (kExtractorCache.count(trimedPath)) {
      op = kExtractorCache.at(trimedPath);
    } else {
      if (kExtractorCache.size() == kMaxCacheNum) {
        // TODO: Blindly evict the first one, use better policy
        kExtractorCache.erase(kExtractorCache.begin());
      }
      op = std::make_shared<SimdJsonExtractor>(trimedPath);
      kExtractorCache[trimedPath] = op;
    }
    return *op;
  }

  std::optional<std::string> extract(const std::string& json);
  std::optional<std::string> extractScalar(const std::string& json);
  std::optional<std::string> extractFromObject(
      int pathIndex,
      simdjson::dom::object obj);
  std::optional<std::string> extractFromArray(
      int pathIndex,
      simdjson::dom::array arr);
  std::optional<std::string> extractOndemand(const std::string& json);
  std::optional<std::string> extractFromObjectOndemand(
      int pathIndex,
      simdjson::ondemand::object obj);
  std::optional<std::string> extractFromArrayOndemand(
      int pathIndex,
      simdjson::ondemand::array arr);
  std::optional<int64_t> getJsonSize(const std::string& json);
  bool isDocBasicType(simdjson::ondemand::document& doc);
  bool isValueBasicType(
      simdjson::simdjson_result<simdjson::ondemand::value> rlt);

  // Shouldn't instantiate directly - use getInstance().
  explicit SimdJsonExtractor(const std::string& path) {
    if (!tokenize(path)) {
      VELOX_USER_FAIL("Invalid JSON path: {}", path);
    }
  }

 private:
  bool tokenize(const std::string& path) {
    if (path.empty()) {
      return false;
    }
    if (!kTokenizer.reset(path)) {
      return false;
    }

    while (kTokenizer.hasNext()) {
      if (auto token = kTokenizer.getNext()) {
        tokens_.push_back(token.value());
      } else {
        tokens_.clear();
        return false;
      }
    }
    return true;
  }

  // Cache tokenize operations in SimdJsonExtractor across invocations in the
  // same thread for the same JsonPath.
  thread_local static std::
      unordered_map<std::string, std::shared_ptr<SimdJsonExtractor>>
          kExtractorCache;
  thread_local static JsonPathTokenizer kTokenizer;

  // Max extractor number in extractor cache
  static const uint32_t kMaxCacheNum{32};

  std::vector<std::string> tokens_;
};

thread_local std::unordered_map<std::string, std::shared_ptr<SimdJsonExtractor>>
    SimdJsonExtractor::kExtractorCache;
thread_local JsonPathTokenizer SimdJsonExtractor::kTokenizer;

bool SimdJsonExtractor::isDocBasicType(simdjson::ondemand::document& doc) {
  return (doc.type() == simdjson::ondemand::json_type::number) ||
      (doc.type() == simdjson::ondemand::json_type::boolean) ||
      (doc.type() == simdjson::ondemand::json_type::null);
}
bool SimdJsonExtractor::isValueBasicType(
    simdjson::simdjson_result<simdjson::ondemand::value> rlt) {
  return (rlt.type() == simdjson::ondemand::json_type::number) ||
      (rlt.type() == simdjson::ondemand::json_type::boolean) ||
      (rlt.type() == simdjson::ondemand::json_type::null);
}

std::optional<std::string> SimdJsonExtractor::extract(const std::string& json) {
  ParserContext ctx(json.data(), json.length());

  try {
    ctx.parseElement();
  } catch (simdjson::simdjson_error& e) {
    // simdjson might throw a conversion error while parsing the input json. In
    // this case, let it return null. follow original version
    return std::nullopt;
  }

  std::optional<std::string> rlt;
  if (ctx.jsonEle.type() ==
      simdjson::SIMDJSON_BUILTIN_IMPLEMENTATION::dom::element_type::ARRAY) {
    rlt = extractFromArray(0, ctx.jsonEle);
  } else if (
      ctx.jsonEle.type() ==
      simdjson::SIMDJSON_BUILTIN_IMPLEMENTATION::dom::element_type::OBJECT) {
    rlt = extractFromObject(0, ctx.jsonEle);
  } else {
    return std::nullopt;
  }
  return rlt;
}

std::optional<std::string> SimdJsonExtractor::extractScalar(
    const std::string& json) {
  ParserContext ctx(json.data(), json.length());
  std::string jsonpath = "";

  try {
    ctx.parseDocument();
  } catch (simdjson::simdjson_error& e) {
    // simdjson might throw a conversion error while parsing the input json. In
    // this case, let it return null. follow original version
    return std::nullopt;
  }

  for (auto& token : tokens_) {
    jsonpath = jsonpath + "/" + token;
  }

  std::string_view rlt_tmp;
  if (jsonpath == "") {
    if (isDocBasicType(ctx.jsonDoc)) {
      rlt_tmp = simdjson::to_json_string(ctx.jsonDoc);
    } else if (ctx.jsonDoc.type() == simdjson::ondemand::json_type::string) {
      rlt_tmp = ctx.jsonDoc.get_string();
    } else {
      return std::nullopt;
    }
  } else {
    try {
      simdjson::simdjson_result<simdjson::ondemand::value> rlt_value =
          ctx.jsonDoc.at_pointer(jsonpath);
      if (isValueBasicType(rlt_value)) {
        rlt_tmp = simdjson::to_json_string(rlt_value);
      } else if (rlt_value.type() == simdjson::ondemand::json_type::string) {
        rlt_tmp = rlt_value.get_string();
      } else {
        return std::nullopt;
      }
    } catch (simdjson::simdjson_error& e) {
      // simdjson might throw a conversion error while parsing the input json.
      // In this case, let it return null. follow original version
      return std::nullopt;
    }
  }
  std::string rlt_s{rlt_tmp};
  return rlt_s;
}

std::optional<std::string> SimdJsonExtractor::extractFromObject(
    int pathIndex,
    simdjson::dom::object obj) {
  if (pathIndex == tokens_.size()) {
    std::string tmp = simdjson::to_string(obj);
    return std::string(tmp);
  }
  if (tokens_[pathIndex] == "*") {
    printf("error: extractFromObject can't include *\n");
    return std::nullopt;
  }
  auto path = "/" + tokens_[pathIndex];
  std::optional<std::string> rlt_string;
  try {
    auto rlt = obj.at_pointer(path);
    if (rlt.type() ==
        simdjson::SIMDJSON_BUILTIN_IMPLEMENTATION::dom::element_type::OBJECT) {
      rlt_string = extractFromObject(pathIndex + 1, rlt);
    } else if (
        rlt.type() ==
        simdjson::SIMDJSON_BUILTIN_IMPLEMENTATION::dom::element_type::ARRAY) {
      rlt_string = extractFromArray(pathIndex + 1, rlt);
    } else {
      std::string tmp = simdjson::to_string(rlt);
      rlt_string = std::optional<std::string>(std::string(tmp));
    }
  } catch (simdjson::simdjson_error& e) {
    // simdjson might throw a conversion error while parsing the input json. In
    // this case, let it return null. follow original version
    return std::nullopt;
  }
  return rlt_string;
}

std::optional<std::string> SimdJsonExtractor::extractFromArray(
    int pathIndex,
    simdjson::dom::array arr) {
  if (pathIndex == tokens_.size()) {
    std::string tmp = simdjson::to_string(arr);
    return std::string(tmp);
  }
  if (tokens_[pathIndex] == "*") {
    std::optional<std::string> rlt_tmp;
    std::string rlt = "[";
    int ii = 0;
    for (auto&& a : arr) {
      ii++;
      if (a.type() ==
          simdjson::SIMDJSON_BUILTIN_IMPLEMENTATION::dom::element_type::
              OBJECT) {
        rlt_tmp = extractFromObject(pathIndex + 1, a);
        if (rlt_tmp.has_value()) {
          rlt += rlt_tmp.value();
          if (ii != arr.size()) {
            rlt += ",";
          }
        }
      } else if (
          a.type() ==
          simdjson::SIMDJSON_BUILTIN_IMPLEMENTATION::dom::element_type::ARRAY) {
        rlt_tmp = extractFromArray(pathIndex + 1, a);
        if (rlt_tmp.has_value()) {
          rlt += rlt_tmp.value();
          if (ii != arr.size()) {
            rlt += ",";
          }
        }
      } else {
        std::string tmp = simdjson::to_string(a);
        rlt += std::string(tmp);
        if (ii != arr.size()) {
          rlt += ",";
        }
      }
      if (ii == arr.size()) {
        rlt += "]";
      }
    }
    return rlt;
  } else {
    auto path = "/" + tokens_[pathIndex];
    std::optional<std::string> rlt_string;
    try {
      auto rlt = arr.at_pointer(path);
      if (rlt.type() ==
          simdjson::SIMDJSON_BUILTIN_IMPLEMENTATION::dom::element_type::
              OBJECT) {
        rlt_string = extractFromObject(pathIndex + 1, rlt);
      } else if (
          rlt.type() ==
          simdjson::SIMDJSON_BUILTIN_IMPLEMENTATION::dom::element_type::ARRAY) {
        rlt_string = extractFromArray(pathIndex + 1, rlt);
      } else {
        std::string tmp = simdjson::to_string(rlt);
        return std::string(tmp);
      }
    } catch (simdjson::simdjson_error& e) {
      // simdjson might throw a conversion error while parsing the input json.
      // In this case, let it return null. follow original version
      return std::nullopt;
    }
    return rlt_string;
  }
}

std::optional<std::string> SimdJsonExtractor::extractOndemand(
    const std::string& json) {
  ParserContext ctx(json.data(), json.length());

  try {
    ctx.parseDocument();
  } catch (simdjson::simdjson_error& e) {
    // simdjson might throw a conversion error while parsing the input json. In
    // this case, let it return null. follow original version
    return std::nullopt;
  }

  std::optional<std::string> rlt;
  if (ctx.jsonDoc.type() ==
      simdjson::SIMDJSON_BUILTIN_IMPLEMENTATION::ondemand::json_type::array) {
    rlt = extractFromArrayOndemand(0, ctx.jsonDoc);
  } else if (
      ctx.jsonDoc.type() ==
      simdjson::SIMDJSON_BUILTIN_IMPLEMENTATION::ondemand::json_type::object) {
    rlt = extractFromObjectOndemand(0, ctx.jsonDoc);
  } else {
    return std::nullopt;
  }
  return rlt;
}

std::optional<std::string> SimdJsonExtractor::extractFromObjectOndemand(
    int pathIndex,
    simdjson::ondemand::object obj) {
  if (pathIndex == tokens_.size()) {
    std::string_view tmp = simdjson::to_json_string(obj);
    return std::string(tmp);
  }
  if (tokens_[pathIndex] == "*") {
    printf("error: extractFromObjectOndemand can't include *\n");
    return std::nullopt;
  }
  obj.reset();
  auto path = "/" + tokens_[pathIndex];
  std::optional<std::string> rlt_string;
  try {
    auto rlt = obj.at_pointer(path);
    if (rlt.type() ==
        simdjson::SIMDJSON_BUILTIN_IMPLEMENTATION::ondemand::json_type::
            object) {
      rlt_string = extractFromObjectOndemand(pathIndex + 1, rlt);
    } else if (
        rlt.type() ==
        simdjson::SIMDJSON_BUILTIN_IMPLEMENTATION::ondemand::json_type::array) {
      rlt_string = extractFromArrayOndemand(pathIndex + 1, rlt);
    } else {
      std::string_view tmp = simdjson::to_json_string(rlt);
      rlt_string = std::optional<std::string>(std::string(tmp));
    }
  } catch (simdjson::simdjson_error& e) {
    // simdjson might throw a conversion error while parsing the input json. In
    // this case, let it return null. follow original version
    return std::nullopt;
  }
  return rlt_string;
}

std::optional<std::string> SimdJsonExtractor::extractFromArrayOndemand(
    int pathIndex,
    simdjson::ondemand::array arr) {
  if (pathIndex == tokens_.size()) {
    std::string_view tmp = simdjson::to_json_string(arr);
    return std::string(tmp);
  }
  arr.reset();
  if (tokens_[pathIndex] == "*") {
    std::optional<std::string> rlt_tmp;
    std::string rlt = "[";
    for (simdjson::ondemand::array_iterator a = arr.begin(); a != arr.end();) {
      if ((*a).type() ==
          simdjson::SIMDJSON_BUILTIN_IMPLEMENTATION::ondemand::json_type::
              object) {
        rlt_tmp = extractFromObjectOndemand(pathIndex + 1, *a);
        if (rlt_tmp.has_value()) {
          rlt += rlt_tmp.value();
          if (++a != arr.end()) {
            rlt += ",";
          }
        } else {
          ++a;
        }
      } else if (
          (*a).type() ==
          simdjson::SIMDJSON_BUILTIN_IMPLEMENTATION::ondemand::json_type::
              array) {
        rlt_tmp = extractFromArrayOndemand(pathIndex + 1, *a);
        if (rlt_tmp.has_value()) {
          rlt += rlt_tmp.value();
          if (++a != arr.end()) {
            rlt += ",";
          }
        } else {
          ++a;
        }
      } else {
        std::string_view tmp = simdjson::to_json_string(*a);
        rlt += std::string(tmp);
        if (++a != arr.end()) {
          rlt += ",";
        }
      }
      if (a == arr.end()) {
        rlt += "]";
      }
    }
    return rlt;
  } else {
    auto path = "/" + tokens_[pathIndex];
    std::optional<std::string> rlt_string;
    try {
      auto rlt = arr.at_pointer(path);
      if (rlt.type() ==
          simdjson::SIMDJSON_BUILTIN_IMPLEMENTATION::ondemand::json_type::
              object) {
        rlt_string = extractFromObjectOndemand(pathIndex + 1, rlt);
      } else if (
          rlt.type() ==
          simdjson::SIMDJSON_BUILTIN_IMPLEMENTATION::ondemand::json_type::
              array) {
        rlt_string = extractFromArrayOndemand(pathIndex + 1, rlt);
      } else {
        std::string_view tmp = simdjson::to_json_string(rlt);
        return std::string(tmp);
      }
    } catch (simdjson::simdjson_error& e) {
      // simdjson might throw a conversion error while parsing the input json.
      // In this case, let it return null. follow original version
      return std::nullopt;
    }
    return rlt_string;
  }
}

std::optional<int64_t> SimdJsonExtractor::getJsonSize(const std::string& json) {
  ParserContext ctx(json.data(), json.length());
  std::string jsonpath = "";
  int64_t len = 0;

  try {
    ctx.parseDocument();
  } catch (simdjson::simdjson_error& e) {
    // simdjson might throw a conversion error while parsing the input json. In
    // this case, let it return null. follow original version
    return std::nullopt;
  }

  for (auto& token : tokens_) {
    jsonpath = jsonpath + "/" + token;
  }

  try {
    auto rlt = ctx.jsonDoc.at_pointer(jsonpath);
    if (rlt.type() ==
        simdjson::SIMDJSON_BUILTIN_IMPLEMENTATION::ondemand::json_type::array) {
      for (auto&& v : rlt) {
        len++;
      }
    } else if (
        rlt.type() ==
        simdjson::SIMDJSON_BUILTIN_IMPLEMENTATION::ondemand::json_type::
            object) {
      len = rlt.count_fields();
    }
  } catch (simdjson::simdjson_error& e) {
    // simdjson might throw a conversion error while parsing the input json. In
    // this case, let it return null. follow original version
    return std::nullopt;
  }
  return len;
}

} // namespace

ParserContext::ParserContext() noexcept = default;
ParserContext::ParserContext(const char* data, size_t length) noexcept
    : padded_json(data, length) {}
void ParserContext::parseElement() {
  jsonEle = domParser.parse(padded_json);
}
void ParserContext::parseDocument() {
  jsonDoc = ondemandParser.iterate(padded_json);
}

std::optional<std::string> simdJsonExtractString(
    const std::string& json,
    const std::string& path) {
  try {
    // If extractor fails to parse the path, this will throw a VeloxUserError,
    // and we want to let this exception bubble up to the client. We only catch
    // json parsing failures (in which cases we return folly::none instead of
    // throw).
    auto& extractor = SimdJsonExtractor::getInstance(path);
    return extractor.extractOndemand(json);
  } catch (const simdjson::simdjson_error& e) {
    // simdjson might throw a conversion error while parsing the input json. In
    // this case, let it return null. follow original version
  }
  return std::nullopt;
}

std::optional<std::string> simdJsonExtractObject(
    const std::string& json,
    const std::string& path) {
  try {
    // If extractor fails to parse the path, this will throw a VeloxUserError,
    // and we want to let this exception bubble up to the client. We only catch
    // json parsing failures (in which cases we return folly::none instead of
    // throw).
    auto& extractor = SimdJsonExtractor::getInstance(path);
    return extractor.extract(json);
  } catch (const simdjson::simdjson_error& e) {
    // simdjson might throw a conversion error while parsing the input json. In
    // this case, let it return null. follow original version
  }
  return std::nullopt;
}

std::optional<std::string> simdJsonExtractScalar(
    const std::string& json,
    const std::string& path) {
  try {
    // If extractor fails to parse the path, this will throw a VeloxUserError,
    // and we want to let this exception bubble up to the client. We only catch
    // json parsing failures (in which cases we return folly::none instead of
    // throw).
    auto& extractor = SimdJsonExtractor::getInstance(path);
    return extractor.extractScalar(json);
  } catch (const simdjson::simdjson_error& e) {
    // simdjson might throw a conversion error while parsing the input json. In
    // this case, let it return null. follow original version
  }
  return std::nullopt;
}

std::optional<int64_t> simdJsonSize(
    const std::string& json,
    const std::string& path) {
  try {
    // If extractor fails to parse the path, this will throw a VeloxUserError,
    // and we want to let this exception bubble up to the client. We only catch
    // json parsing failures (in which cases we return folly::none instead of
    // throw).
    auto& extractor = SimdJsonExtractor::getInstance(path);
    return extractor.getJsonSize(json);
  } catch (const simdjson::simdjson_error& e) {
    // simdjson might throw a conversion error while parsing the input json. In
    // this case, let it return null. follow original version
  }
  return std::nullopt;
}

} // namespace facebook::velox::functions
