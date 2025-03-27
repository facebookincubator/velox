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

#include <boost/type_index.hpp>
#include "velox/exec/WindowFunction.h"
#include "velox/expression/SignatureBinder.h"
#include "velox/jni/cpp/memory/MemoryManager.h"

#include "velox/expression/Expr.h"
#include "velox/parse/Expressions.h"
#include "velox/parse/ExpressionsParser.h"

#include <jni.h>
#include <type/fbhive/HiveTypeParser.h>
#include "velox/jni/cpp/jni/JniCommon.h"

namespace facebook::velox::sdk::expression {

class ExprUtils {
 public:
  template <typename T>
  static std::vector<std::shared_ptr<const T>> asISerializableVector(
      JNIEnv* env,
      jobjectArray exprJsons) {
    std::vector<std::shared_ptr<const T>> typeExprs;
    std::vector<std::string> jsons =
        ConvertJStringArrayToVector(env, exprJsons);
    for (auto json : jsons) {
      std::shared_ptr<const T> result = ISerializable::deserialize<T>(
          folly::parseJson(json, getSerializationOptions()),
          sdk::memory::MemoryManager::get()->planMemoryPool().get());
      VELOX_CHECK_NOT_NULL(
          result,
          "failed to deserialize to class {}  with json {} ",
          boost::typeindex::type_id<T>().pretty_name().c_str(),
          json);
      typeExprs.emplace_back(result);
    }
    return typeExprs;
  }

  template <typename T>
  static std::shared_ptr<const T> asISerializable(
      JNIEnv* env,
      jstring exprJson) {
    auto json = jStringToCString(env, exprJson);
    std::shared_ptr<const T> result = ISerializable::deserialize<T>(
        folly::parseJson(json, getSerializationOptions()),
        sdk::memory::MemoryManager::get()->planMemoryPool().get());
    VELOX_CHECK_NOT_NULL(
        result,
        "failed to deserialize to class {}  with json {} ",
        boost::typeindex::type_id<T>().pretty_name().c_str(),
        json);
    return result;
  }

  template <typename T>
  static std::vector<T> deserializeArray(JNIEnv* env, jobjectArray exprJsons) {
    std::vector<T> typeExprs;
    std::vector<std::string> jsons =
        ConvertJStringArrayToVector(env, exprJsons);
    for (const auto& json : jsons) {
      typeExprs.push_back(
          T::deserialize(folly::parseJson(json, getSerializationOptions())));
    }
    return typeExprs;
  }

  static TypePtr jsonToVeloxType(std::string json) {
    return convertSparkStructToVelox(
        folly::parseJson(json, getSerializationOptions()));
  }

  static TypePtr convertSparkFieldToVelox(const folly::dynamic& field) {
    if (field["type"].isObject()) {
      return convertSparkStructToVelox(field);
    }
    auto type = field["type"].asString();

    if (type == "integer") {
      return INTEGER();
    } else if (type == "long") {
      return BIGINT();
    } else if (type == "string") {
      return VARCHAR();
    } else if (type == "double") {
      return DOUBLE();
    } else if (type == "date") {
      return DATE();
    } else if (type == "timestamp") {
      return TIMESTAMP();
    } else {
      throw std::runtime_error("Unsupported type: " + type);
    }
  }

  static TypePtr convertSparkStructToVelox(const folly::dynamic& sparkStruct) {
    std::vector<TypePtr> fields;
    std::vector<std::string> names;
    for (const auto& field : sparkStruct["fields"]) {
      fields.push_back(convertSparkFieldToVelox(field));
      names.push_back(field["name"].asString());
    }
    return ROW(std::move(names), std::move(fields));
  }

  static TypePtr toVeloxType(std::string str) {
    type::fbhive::HiveTypeParser parser;
    return parser.parse(str);
  }

  static std::string throwWindowFunctionSignatureNotSupported(
      const std::string& name,
      const std::vector<TypePtr>& types,
      const std::vector<exec::FunctionSignaturePtr>& signatures) {
    std::stringstream error;
    error << "Window function signature is not supported: "
          << facebook::velox::exec::toString(name, types)
          << ". Supported signatures: " << toString(signatures) << ".";
    VELOX_USER_FAIL(error.str());
  }

  static std::string throwWindowFunctionDoesntExist(const std::string& name) {
    std::stringstream error;
    error << "Window function doesn't exist: " << name << ".";
    if (exec::windowFunctions().empty()) {
      error << " Registry of window functions is empty. "
               "Make sure to register some window functions.";
    }
    VELOX_USER_FAIL(error.str());
  }

  static TypePtr resolveWindowType(
      const std::string& windowFunctionName,
      const std::vector<TypePtr>& inputTypes,
      bool nullOnFailure) {
    if (auto signatures =
            exec::getWindowFunctionSignatures(windowFunctionName)) {
      for (const auto& signature : signatures.value()) {
        exec::SignatureBinder binder(*signature, inputTypes);
        if (binder.tryBind()) {
          return binder.tryResolveType(signature->returnType());
        }
      }

      if (nullOnFailure) {
        return nullptr;
      }
      throwWindowFunctionSignatureNotSupported(
          windowFunctionName, inputTypes, signatures.value());
    }

    if (nullOnFailure) {
      return nullptr;
    }
    throwWindowFunctionDoesntExist(windowFunctionName);
    return nullptr;
  }
};

} // namespace facebook::velox::sdk::expression
