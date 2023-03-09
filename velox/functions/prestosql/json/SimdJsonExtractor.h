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

#pragma once

#include <string>
#include <optional>

#include "folly/Range.h"
#include "folly/dynamic.h"
#include "simdjson.h"

namespace facebook::velox::functions {

/**
 * Extract a json object from path
 * @param json: A json object
 * @param path: Path to locate a json object. Following operators are supported
 *              "$"      Root member of a json structure no matter it's an
 *                       object or an array
 *              "./[]"   Child operator to get a children object
 *              "[]"     Subscript operator for array and map
 *              "*"      Wildcard for [], get all the elements of an array
 * @return Return json string object on success.
 *         On invalid json path, returns folly::none (not json null) value
 *         On non-json value, returns the original value.
 * Example:
 * For the following example: ,
 * "{\"store\":,
 *   {\"fruit\":\\[{\"weight\":8,\"type\":\"apple\"},
 *                 {\"weight\":9,\"type\":\"pear\"}],
 *    \"bicycle\":{\"price\":19.95,\"color\":\"red\"}
 *   },
 *  \"email\":\"amy@only_for_json_udf_test.net\",
 *  \"owner\":\"amy\",
 * }",
 * jsonExtract(json, "$.owner") = "amy",
 * jsonExtract(json, "$.store.fruit[0]") =
 *    "{\"weight\":8,\"type\":\"apple\"}",
 * jsonExtract(json, "$.non_exist_key") = NULL
 * jsonExtract(json, "$.store.fruit[*].type") = "[\"apple\", \"pear\"]"
 */
struct ParserContext{
public:
    explicit ParserContext() noexcept;
    explicit ParserContext(const char *data, size_t length) noexcept;
    void parseElement();
    void parseDocument();
    simdjson::dom::element jsonEle;
    simdjson::ondemand::document jsonDoc;
private:
    simdjson::padded_string padded_json;
    simdjson::dom::parser domParser;
    simdjson::ondemand::parser ondemandParser;
};


std::optional<std::string> SimdJsonExtractString(
    const std::string& json,
    const std::string& path);

std::optional<std::string> SimdJsonExtractObject(
    const std::string& json,
    const std::string& path);

std::optional<std::string> SimdJsonExtractScalar(
    const std::string& json,
    const std::string& path);
std::optional<std::string> SimdJsonKeysWithJsonPath(
    const std::string& json,
    const std::string& path);
} // namespace facebook::velox::functions
