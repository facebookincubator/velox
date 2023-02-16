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
#include "folly/json.h"
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
  std::optional<std::string> extractFromObject(int path_index, simdjson::dom::object obj);
  std::optional<std::string> extractFromArray(int path_index, simdjson::dom::array arr);
  std::optional<std::string> extractOndemand(const std::string& json);
  std::optional<std::string> extractKeysOndemand(const std::string& json);
  std::optional<std::string> extractFromObjectOndemand(int path_index, simdjson::ondemand::object obj);
  std::optional<std::string> extractFromArrayOndemand(int path_index, simdjson::ondemand::array arr);
  bool isDocBasicType(simdjson::ondemand::document& doc);
  bool isValueBasicType(simdjson::simdjson_result<simdjson::ondemand::value> rlt);

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

  // Cache tokenize operations in SimdJsonExtractor across invocations in the same
  // thread for the same JsonPath.
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
  return (doc.type() == simdjson::ondemand::json_type::number) || (doc.type() == simdjson::ondemand::json_type::boolean) || (doc.type() == simdjson::ondemand::json_type::null);
}
bool SimdJsonExtractor::isValueBasicType(simdjson::simdjson_result<simdjson::ondemand::value> rlt) {
  return (rlt.type() == simdjson::ondemand::json_type::number) || (rlt.type() == simdjson::ondemand::json_type::boolean) || (rlt.type() == simdjson::ondemand::json_type::null);
}

std::optional<std::string> SimdJsonExtractor::extractScalar(
    const std::string& json) {

  ParserContext ctx(json.data(), json.length());
  std::string jsonpath = "";

  try{
    ctx.parseDocument();
  } catch(simdjson::simdjson_error& e)
  {
    printf("error: Failed to parse json as document. error :%s\n",simdjson::error_message(e.error()));
    return std::nullopt;
  }

  for (auto& token : tokens_) {
    jsonpath = jsonpath + "/" + token;
  }

  std::string_view rlt_tmp;
  if(jsonpath == "")
  {
    if(isDocBasicType(ctx.jsonDoc)) {
      rlt_tmp = simdjson::to_json_string(ctx.jsonDoc);
    }
    else if(ctx.jsonDoc.type() == simdjson::ondemand::json_type::string){
      rlt_tmp = ctx.jsonDoc.get_string();
    }
    else {
      return std::nullopt;
    }
  }
  else {
    try{
      simdjson::simdjson_result<simdjson::ondemand::value> rlt_value = ctx.jsonDoc.at_pointer(jsonpath);
      if(isValueBasicType(rlt_value)) {
        rlt_tmp = simdjson::to_json_string(rlt_value);
      }
      else if(rlt_value.type() == simdjson::ondemand::json_type::string){
        rlt_tmp = rlt_value.get_string();
      }
      else {
        return std::nullopt;
      }
    }
    catch(simdjson::simdjson_error& e){
      printf("error: Failed to find jsonpath at json object. error :%s\n",simdjson::error_message(e.error()));
      return std::nullopt;
    }
  }
  std::string rlt_s{rlt_tmp};
  return rlt_s;
}

std::optional<std::string> SimdJsonExtractor::extract(
    const std::string& json) {
  ParserContext ctx(json.data(), json.length());

  try{
    ctx.parseElement();
  } catch(simdjson::simdjson_error& e)
  {
    printf("error: Failed to parse json as document. error :%s\n", simdjson::error_message(e.error()));
    return std::nullopt;
  }

  std::optional<std::string> rlt;
  if(ctx.jsonEle.type() == simdjson::SIMDJSON_BUILTIN_IMPLEMENTATION::dom::element_type::ARRAY) {
    rlt = extractFromArray(0, ctx.jsonEle);
  }
  else if(ctx.jsonEle.type() == simdjson::SIMDJSON_BUILTIN_IMPLEMENTATION::dom::element_type::OBJECT) {
    rlt = extractFromObject(0, ctx.jsonEle);
  }
  else{
    return std::nullopt;
  }
  return rlt;
}
std::optional<std::string> SimdJsonExtractor::extractFromObject(
    int path_index, simdjson::dom::object obj) {
  if(path_index == tokens_.size()) {
    std::string tmp = simdjson::to_string(obj);
    return std::string(tmp);
  }
  if(tokens_[path_index] == "*") {
    printf("error: extractFromObject can't include *\n");
    return std::nullopt;
  }
  auto path = "/" + tokens_[path_index];
  std::optional<std::string> rlt_string;
  try{
    auto rlt = obj.at_pointer(path);
    if(rlt.type() == simdjson::SIMDJSON_BUILTIN_IMPLEMENTATION::dom::element_type::OBJECT){
      rlt_string = extractFromObject(path_index+1, rlt);
    }
    else if(rlt.type() == simdjson::SIMDJSON_BUILTIN_IMPLEMENTATION::dom::element_type::ARRAY){
      rlt_string = extractFromArray(path_index+1, rlt);
    }
    else{
      std::string tmp = simdjson::to_string(rlt);
      rlt_string = std::optional<std::string>(std::string(tmp));
    }
  }
  catch(simdjson::simdjson_error& e) {
    printf("extractFromObject json failed, error: %s",simdjson::error_message(e.error()));
    return std::nullopt;
  } 
  return rlt_string;
}

std::optional<std::string> SimdJsonExtractor::extractFromArray(
    int path_index, simdjson::dom::array arr) {
  if(path_index == tokens_.size()) {
    std::string tmp = simdjson::to_string(arr);
    return std::string(tmp);
  }
  if(tokens_[path_index] == "*") {
    std::optional<std::string> rlt_tmp;
    std::string rlt = "[";
    int ii = 0;
    for(auto &&a:arr ) {
      ii++;
      if(a.type() == simdjson::SIMDJSON_BUILTIN_IMPLEMENTATION::dom::element_type::OBJECT) {
        rlt_tmp = extractFromObject(path_index+1, a);
        if(rlt_tmp.has_value()) {
          rlt += rlt_tmp.value();
          if(ii != arr.size()){
            rlt += ",";
          }
        }
      }
      else if(a.type() == simdjson::SIMDJSON_BUILTIN_IMPLEMENTATION::dom::element_type::ARRAY) {
        rlt_tmp = extractFromArray(path_index+1, a);
        if(rlt_tmp.has_value()) {
          rlt += rlt_tmp.value();
          if(ii != arr.size()){
            rlt += ",";
          }
        }
      }
      else{
        std::string tmp = simdjson::to_string(a);
        rlt += std::string(tmp);
        if(ii != arr.size()){
          rlt += ",";
        }
      }
      if(ii == arr.size()){
        rlt += "]";
      }
    }
    return rlt;
  }
  else {
    auto path = "/" + tokens_[path_index];
    std::optional<std::string> rlt_string;
    try{
      auto rlt = arr.at_pointer(path);
      if(rlt.type() == simdjson::SIMDJSON_BUILTIN_IMPLEMENTATION::dom::element_type::OBJECT){
        rlt_string = extractFromObject(path_index+1, rlt);
      }
      else if(rlt.type() == simdjson::SIMDJSON_BUILTIN_IMPLEMENTATION::dom::element_type::ARRAY){
        rlt_string = extractFromArray(path_index+1, rlt);
      }
      else{
        std::string tmp = simdjson::to_string(rlt);
        return std::string(tmp);
      }
    }
    catch(simdjson::simdjson_error& e){
      return std::nullopt;
    }
    return rlt_string;
  }
}

std::optional<std::string> SimdJsonExtractor::extractOndemand(
    const std::string& json) {
  ParserContext ctx(json.data(), json.length());

  try{
    ctx.parseDocument();
  } catch(simdjson::simdjson_error& e)
  {
    printf("error: Failed to parse json as document. error :%s\n",simdjson::error_message(e.error()));
    return std::nullopt;
  }

  std::optional<std::string> rlt;
  if(ctx.jsonDoc.type() == simdjson::SIMDJSON_BUILTIN_IMPLEMENTATION::ondemand::json_type::array) {
    rlt = extractFromArrayOndemand(0, ctx.jsonDoc);
  }
  else if(ctx.jsonDoc.type() == simdjson::SIMDJSON_BUILTIN_IMPLEMENTATION::ondemand::json_type::object) {
    rlt = extractFromObjectOndemand(0, ctx.jsonDoc);
  }
  else{
    return std::nullopt;
  }
  return rlt;
}

std::optional<std::string> SimdJsonExtractor::extractKeysOndemand(
    const std::string& json) {
  
  ParserContext ctx(json.data(), json.length());
  std::string jsonpath = "";

  try {
    ctx.parseDocument();
  }
  catch(simdjson::simdjson_error& e) {
    printf("error: Failed to parse json as document. error :%s\n",simdjson::error_message(e.error()));
    return std::nullopt;
  }
  for (auto& token : tokens_) {
    jsonpath = jsonpath + "/" + token;
  }

  try{
    simdjson::simdjson_result<simdjson::ondemand::value> rlt_value = ctx.jsonDoc.at_pointer(jsonpath);
    if(rlt_value.type() != simdjson::ondemand::json_type::object) {
      std::string_view tmp = simdjson::to_json_string(rlt_value);
      return std::nullopt;
    }

    int objCnt = rlt_value.count_fields();
    std::string rlt = "[";
    int count = 0;
    for (auto &&field : rlt_value.get_object()) {
      std::string_view tmp = field.unescaped_key();
      rlt += "\""+std::string(tmp)+"\"";
      if(++count != objCnt) {
        rlt += ",";
      }
    }
    rlt += "]";
    return rlt;
  }
  catch (simdjson::simdjson_error& e) {
    printf("error: Failed to find json key. error :%s\n",simdjson::error_message(e.error()));
    return std::nullopt;
  }
  return std::nullopt;
}

std::optional<std::string> SimdJsonExtractor::extractFromObjectOndemand(
    int path_index, simdjson::ondemand::object obj) {
  if(path_index == tokens_.size()) {
    std::string_view tmp = simdjson::to_json_string(obj);
    return std::string(tmp);
  }
  if(tokens_[path_index] == "*") {
    printf("error: extractFromObjectOndemand can't include *\n");
    return std::nullopt;
  }
  obj.reset();
  auto path = "/" + tokens_[path_index];
  std::optional<std::string> rlt_string;
  try{
    auto rlt = obj.at_pointer(path);
    if(rlt.type() == simdjson::SIMDJSON_BUILTIN_IMPLEMENTATION::ondemand::json_type::object){
      rlt_string = extractFromObjectOndemand(path_index+1, rlt);
    }
    else if(rlt.type() == simdjson::SIMDJSON_BUILTIN_IMPLEMENTATION::ondemand::json_type::array){
      rlt_string = extractFromArrayOndemand(path_index+1, rlt);
    }
    else{
      std::string_view tmp = simdjson::to_json_string(rlt);
      rlt_string = std::optional<std::string>(std::string(tmp));
    }
  }
  catch(simdjson::simdjson_error& e) {
    return std::nullopt;
  } 
  return rlt_string;
}

std::optional<std::string> SimdJsonExtractor::extractFromArrayOndemand(
    int path_index, simdjson::ondemand::array arr) {
  if(path_index == tokens_.size()) {
    std::string_view tmp = simdjson::to_json_string(arr);
    return std::string(tmp);
  }
  arr.reset();
  if(tokens_[path_index] == "*") {
    std::optional<std::string> rlt_tmp;
    std::string rlt = "[";
    for(simdjson::ondemand::array_iterator a=arr.begin(); a!=arr.end(); ) {
      if((*a).type() == simdjson::SIMDJSON_BUILTIN_IMPLEMENTATION::ondemand::json_type::object) {
        rlt_tmp = extractFromObjectOndemand(path_index+1, *a);
        if(rlt_tmp.has_value()) {
          rlt += rlt_tmp.value();
          if(++a != arr.end()){
            rlt += ",";
          }
        }
        else {
          ++a;
        }
      }
      else if((*a).type() == simdjson::SIMDJSON_BUILTIN_IMPLEMENTATION::ondemand::json_type::array) {
        rlt_tmp = extractFromArrayOndemand(path_index+1, *a);
        if(rlt_tmp.has_value()) {
          rlt += rlt_tmp.value();
          if(++a != arr.end()){
            rlt += ",";
          }
        }
        else {
          ++a;
        }
      }
      else{
        std::string_view tmp = simdjson::to_json_string(*a);
        rlt += std::string(tmp);
        if(++a != arr.end()){
          rlt += ",";
        }
      }
      if(a == arr.end()){
        rlt += "]";
      }
    }
    return rlt;
  }
  else {
    auto path = "/" + tokens_[path_index];
    std::optional<std::string> rlt_string;
    try{
      auto rlt = arr.at_pointer(path);
      if(rlt.type() == simdjson::SIMDJSON_BUILTIN_IMPLEMENTATION::ondemand::json_type::object){
        rlt_string = extractFromObjectOndemand(path_index+1, rlt);
      }
      else if(rlt.type() == simdjson::SIMDJSON_BUILTIN_IMPLEMENTATION::ondemand::json_type::array){
        rlt_string = extractFromArrayOndemand(path_index+1, rlt);
      }
      else{
        std::string_view tmp = simdjson::to_json_string(rlt);
        return std::string(tmp);
      }
    }
    catch(simdjson::simdjson_error& e){
      return std::nullopt;
    }
    return rlt_string;
  }
}

} // namespace

std::optional<std::string> SimdJsonExtract(
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
    printf("extract json failed, error: %s",simdjson::error_message(e.error()));
  }
  return std::nullopt;
}

std::optional<std::string> SimdJsonExtractOndemand(
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
    printf("extractOndemand json failed, error: %s",simdjson::error_message(e.error()));
  }
  return std::nullopt;
}

std::optional<std::string> SimdJsonExtractScalar(
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
    printf("extractScalar json failed, error: %s",simdjson::error_message(e.error()));
  }
  return std::nullopt;
}

std::optional<std::string> SimdJsonKeysWithJsonPathOndemand(
    const std::string& json,
    const std::string& path) {
  try {
    // If extractor fails to parse the path, this will throw a VeloxUserError,
    // and we want to let this exception bubble up to the client. We only catch
    // json parsing failures (in which cases we return folly::none instead of
    // throw).
    auto& extractor = SimdJsonExtractor::getInstance(path);
    return extractor.extractKeysOndemand(json);
  } catch (const simdjson::simdjson_error& e) {
    printf("extractKeysOndemand json failed, error: %s",simdjson::error_message(e.error()));
  }
  return std::nullopt;
}

ParserContext::ParserContext() noexcept = default;
ParserContext::ParserContext(const char *data, size_t length) noexcept 
    : padded_json(data, length){
}
void ParserContext::parseElement() {
    jsonEle = domParser.parse(padded_json);
}
void ParserContext::parseDocument() {
    jsonDoc = ondemandParser.iterate(padded_json);
}

std::optional<std::string> SimdJsonExtractString(
    const std::string& json,
    const std::string& path) {
  auto res = SimdJsonExtractOndemand(json, path);
  return res;
}

std::optional<std::string> SimdJsonExtractObject(
    const std::string& json,
    const std::string& path) {
  auto res = SimdJsonExtract(json, path);
  return res;
}

std::optional<std::string> SimdJsonKeysWithJsonPath(
    const std::string& json,
    const std::string& path) {
  auto res = SimdJsonKeysWithJsonPathOndemand(json, path);
  return res;
}

} // namespace facebook::velox::functions
