/*
* Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef EXPRESSIONUTILS_HPP
#define EXPRESSIONUTILS_HPP

#include <boost/type_index.hpp>
#include <exec/WindowFunction.h>
#include <expression/SignatureBinder.h>
#include "sdk/cpp/memory/MemoryManager.h"

#include "velox/expression/Expr.h"
#include "velox/parse/Expressions.h"
#include "velox/parse/ExpressionsParser.h"
#include "velox/parse/TypeResolver.h"

#include <type/fbhive/HiveTypeParser.h>

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

  static exec::ExprSet compileExpression(
      const std::string& text,
      const TypePtr& rowType) {
    parse::ParseOptions options_;
    auto previousHook = core::Expressions::getResolverHook();
    parse::registerTypeResolver();
    auto untyped = parse::parseExpr(text, options_);
    auto typed = core::Expressions::inferTypes(
        untyped,
        rowType,
        sdk::memory::MemoryManager::get()->dictExecutionMemoryPool().get());
    core::Expressions::setTypeResolverHook(previousHook);
    std::shared_ptr<core::QueryCtx> queryCtx_ = core::QueryCtx::create();
    core::ExecCtx execCtx_{
        sdk::memory::MemoryManager::get()->vectorBatchMemoryPool().get(),
        queryCtx_.get()};
    return exec::ExprSet({typed}, &execCtx_);
  }

  static VectorPtr evaluate(
      exec::ExprSet& exprSet,
      const RowVectorPtr& data,
      const SelectivityVector& rows) {
    std::shared_ptr<core::QueryCtx> queryCtx_ = core::QueryCtx::create();
    core::ExecCtx execCtx_{
        sdk::memory::MemoryManager::get()->dictExecutionMemoryPool().get(),
        queryCtx_.get()};
    exec::EvalCtx evalCtx(&execCtx_, &exprSet, data.get());
    std::vector<VectorPtr> results(1);
    exprSet.eval(rows, evalCtx, results);
    return results[0];
  }

  static VectorPtr evaluate(exec::ExprSet& exprSet, const RowVectorPtr& data) {
    SelectivityVector rows(data->size());
    return evaluate(exprSet, data, rows);
  }

  static VectorPtr evaluate(
      const std::string& expression,
      const RowVectorPtr& data) {
    auto exprSet = compileExpression(expression, asRowType(data->type()));
    return evaluate(exprSet, data);
  }

  static TypePtr jsonToVeloxType(std::string json) {
    return convertSparkStructToVelox(
        folly::parseJson(json, getSerializationOptions()));
  }

  // 将Spark的字段类型转换为Velox的Type
  static TypePtr convertSparkFieldToVelox(const folly::dynamic& field) {
    if (field["type"].isObject()) {
      return convertSparkStructToVelox(field);
    }
    auto type = field["type"].asString();

    // 这个例子没有处理nullable字段，实际使用时你可能需要考虑这个字段

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

  // 将Spark的StructType转换为Velox的Type
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

  static core::PlanNodePtr ToPlan(std::string planJson) {
    std::shared_ptr<const core::PlanNode> plan =
        ISerializable::deserialize<core::PlanNode>(
            folly::parseJson(planJson, getSerializationOptions()),
            sdk::memory::MemoryManager::get()->dictExecutionMemoryPool().get());
    return plan;
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

} // namespace facebook::velox::expression
#endif // EXPRESSIONUTILS_HPP
