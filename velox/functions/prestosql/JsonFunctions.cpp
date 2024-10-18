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
#include "velox/expression/VectorFunction.h"
#include "velox/functions/prestosql/json/SIMDJsonUtil.h"
#include "velox/functions/prestosql/types/JsonType.h"
#include "velox/type/Conversions.h"

namespace facebook::velox::functions {

namespace {
constexpr const char* kArrayStart = "[";
constexpr const char* kArrayEnd = "]";
constexpr const char* kSeparator = ",";
constexpr const char* kObjectStart = "{";
constexpr const char* kObjectEnd = "}";
constexpr const char* kObjectKeySeparator = ":";
constexpr const char* kQuote = "\"";

class JsonView {
 public:
  virtual void canonicalize(std::stringstream& stream) = 0;
};

using JsonViewPtr = std::shared_ptr<JsonView>;

struct JsonLeafView : public JsonView {
  JsonLeafView(const StringView view) : view_(view){};

  void canonicalize(std::stringstream& stream) override {
    stream << view_;
  }

 private:
  const StringView view_;
};

struct JsonArrayView : public JsonView {
  JsonArrayView(const std::vector<JsonViewPtr> array) : array_(array){};

  void canonicalize(std::stringstream& stream) override {
    stream << kArrayStart;
    for (auto i = 0; i < array_.size(); i++) {
      array_[i]->canonicalize(stream);
      if (i < array_.size() - 1) {
        stream << kSeparator;
      }
    }
    stream << kArrayEnd;
  }

 private:
  const std::vector<JsonViewPtr> array_;
};

struct JsonObjView : public JsonView {
  JsonObjView(std::vector<std::pair<StringView, JsonViewPtr>> objFields)
      : objFields_(objFields){};

  void canonicalize(std::stringstream& stream) override {
    std::sort(objFields_.begin(), objFields_.end(), [](auto& a, auto& b) {
      return a.first < b.first;
    });

    stream << kObjectStart;
    for (auto i = 0; i < objFields_.size(); i++) {
      auto field = objFields_[i];
      stream << kQuote << field.first << kQuote << kObjectKeySeparator;
      field.second->canonicalize(stream);
      if (i < objFields_.size() - 1) {
        stream << kSeparator;
      }
    }
    stream << kObjectEnd;
  }

 private:
  std::vector<std::pair<StringView, JsonViewPtr>> objFields_;
};

} // namespace

namespace {
class JsonFormatFunction : public exec::VectorFunction {
 public:
  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& /* outputType */,
      exec::EvalCtx& context,
      VectorPtr& result) const override {
    VectorPtr localResult;

    // Input can be constant or flat.
    assert(args.size() > 0);
    const auto& arg = args[0];
    if (arg->isConstantEncoding()) {
      auto value = arg->as<ConstantVector<StringView>>()->valueAt(0);
      localResult = std::make_shared<ConstantVector<StringView>>(
          context.pool(), rows.end(), false, VARCHAR(), std::move(value));
    } else {
      auto flatInput = arg->asFlatVector<StringView>();

      auto stringBuffers = flatInput->stringBuffers();
      VELOX_CHECK_LE(rows.end(), flatInput->size());
      localResult = std::make_shared<FlatVector<StringView>>(
          context.pool(),
          VARCHAR(),
          nullptr,
          rows.end(),
          flatInput->values(),
          std::move(stringBuffers));
    }

    context.moveOrCopyResult(localResult, rows, result);
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    // json -> varchar
    return {exec::FunctionSignatureBuilder()
                .returnType("varchar")
                .argumentType("json")
                .build()};
  }
};

class JsonParseFunction : public exec::VectorFunction {
 public:
  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& /* outputType */,
      exec::EvalCtx& context,
      VectorPtr& result) const override {
    // Initialize errors here so that we get the proper exception context.
    folly::call_once(
        initializeErrors_, [this] { simdjsonErrorsToExceptions(errors_); });

    VectorPtr localResult;

    // Input can be constant or flat.
    assert(args.size() > 0);
    const auto& arg = args[0];
    if (arg->isConstantEncoding()) {
      auto value = arg->as<ConstantVector<StringView>>()->valueAt(0);
      paddedInput_.resize(value.size() + simdjson::SIMDJSON_PADDING);
      memcpy(paddedInput_.data(), value.data(), value.size());
      auto canonicalStringStream = std::stringstream{};
      JsonViewPtr jsonView;

      if (auto error = parse(value.size(), jsonView)) {
        context.setErrors(rows, errors_[error]);
        return;
      }

      jsonView->canonicalize(canonicalStringStream);
      localResult = BaseVector::createConstant(
          JSON(), canonicalStringStream.str(), rows.end(), context.pool());

    } else {
      auto flatInput = arg->asFlatVector<StringView>();
      BufferPtr stringViews = AlignedBuffer::allocate<StringView>(
          rows.end(), context.pool(), StringView());

      // TODO: Optimize this
      localResult = std::make_shared<FlatVector<StringView>>(
          context.pool(),
          JSON(),
          nullptr,
          rows.end(),
          stringViews,
          std::vector<BufferPtr>{});
      auto flatResult = localResult->asFlatVector<StringView>();

      auto stringBuffers = flatInput->stringBuffers();
      VELOX_CHECK_LE(rows.end(), flatInput->size());

      size_t maxSize = 0;
      rows.applyToSelected([&](auto row) {
        auto value = flatInput->valueAt(row);
        maxSize = std::max(maxSize, value.size());
      });
      paddedInput_.resize(maxSize + simdjson::SIMDJSON_PADDING);

      auto canonicalStringStream = std::stringstream{};
      JsonViewPtr jsonView;
      rows.applyToSelected([&](auto row) {
        auto value = flatInput->valueAt(row);
        memcpy(paddedInput_.data(), value.data(), value.size());
        if (auto error = parse(value.size(), jsonView)) {
          context.setVeloxExceptionError(row, errors_[error]);
        } else {
          jsonView->canonicalize(canonicalStringStream);
          auto canonicalString = canonicalStringStream.str();
          // TODO: This creates a copy, can we optimize.
          flatResult->set(
              row, StringView(canonicalString.data(), canonicalString.size()));
          canonicalStringStream.str(std::string());
        }
      });
    }

    context.moveOrCopyResult(localResult, rows, result);
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    // varchar -> json
    return {exec::FunctionSignatureBuilder()
                .returnType("json")
                .argumentType("varchar")
                .build()};
  }

 private:
  simdjson::error_code parse(size_t size, JsonViewPtr& jsonView) const {
    simdjson::padded_string_view paddedInput(
        paddedInput_.data(), size, paddedInput_.size());
    SIMDJSON_ASSIGN_OR_RAISE(auto doc, simdjsonParse(paddedInput));
    SIMDJSON_TRY(validate<simdjson::ondemand::document&>(doc, jsonView));
    if (!doc.at_end()) {
      return simdjson::TRAILING_CONTENT;
    }
    return simdjson::SUCCESS;
  }

  template <typename T>
  static simdjson::error_code validate(T value, JsonViewPtr& jsonView) {
    SIMDJSON_ASSIGN_OR_RAISE(auto type, value.type());
    switch (type) {
      case simdjson::ondemand::json_type::array: {
        SIMDJSON_ASSIGN_OR_RAISE(auto array, value.get_array());

        std::vector<JsonViewPtr> arrayPtr;
        for (auto elementOrError : array) {
          SIMDJSON_ASSIGN_OR_RAISE(auto element, elementOrError);
          JsonViewPtr elementPtr;
          SIMDJSON_TRY(validate(element, elementPtr));
          arrayPtr.push_back(elementPtr);
        }

        jsonView = std::make_shared<JsonArrayView>(arrayPtr);
        return simdjson::SUCCESS;
      }
      case simdjson::ondemand::json_type::object: {
        SIMDJSON_ASSIGN_OR_RAISE(auto object, value.get_object());

        std::vector<std::pair<StringView, JsonViewPtr>> objFields;
        for (auto fieldOrError : object) {
          SIMDJSON_ASSIGN_OR_RAISE(auto field, fieldOrError);
          JsonViewPtr elementPtr;
          auto key = field.escaped_key();
          auto trimmedKey = velox::util::trimWhiteSpace(key.data(), key.size());
          SIMDJSON_TRY(validate(field.value(), elementPtr));
          objFields.push_back({trimmedKey, elementPtr});
        }

        jsonView = std::make_shared<JsonObjView>(objFields);
        return simdjson::SUCCESS;
      }
      case simdjson::ondemand::json_type::number: {
        std::string_view rawJsonv = value.raw_json_token();

        jsonView = std::make_shared<JsonLeafView>(
            velox::util::trimWhiteSpace(rawJsonv.data(), rawJsonv.size()));
        return value.get_double().error();
      }
      case simdjson::ondemand::json_type::string: {
        std::string_view rawJsonv = value.raw_json_token();

        auto s = velox::util::trimWhiteSpace(rawJsonv.data(), rawJsonv.size());
        jsonView = std::make_shared<JsonLeafView>(s);
        return value.get_string().error();
      }

      case simdjson::ondemand::json_type::boolean: {
        std::string_view rawJsonv = value.raw_json_token();

        jsonView = std::make_shared<JsonLeafView>(
            velox::util::trimWhiteSpace(rawJsonv.data(), rawJsonv.size()));
        return value.get_bool().error();
      }

      case simdjson::ondemand::json_type::null: {
        SIMDJSON_ASSIGN_OR_RAISE(auto isNull, value.is_null());
        std::string_view rawJsonv = value.raw_json_token();

        jsonView = std::make_shared<JsonLeafView>(
            velox::util::trimWhiteSpace(rawJsonv.data(), rawJsonv.size()));
        return isNull ? simdjson::SUCCESS : simdjson::N_ATOM_ERROR;
      }
    }
    VELOX_UNREACHABLE();
  }

  mutable folly::once_flag initializeErrors_;
  mutable std::exception_ptr errors_[simdjson::NUM_ERROR_CODES];
  // Padding is needed in case string view is inlined.
  mutable std::string paddedInput_;
};

} // namespace

VELOX_DECLARE_VECTOR_FUNCTION(
    udf_json_format,
    JsonFormatFunction::signatures(),
    std::make_unique<JsonFormatFunction>());

VELOX_DECLARE_STATEFUL_VECTOR_FUNCTION(
    udf_json_parse,
    JsonParseFunction::signatures(),
    [](const std::string& /*name*/,
       const std::vector<exec::VectorFunctionArg>&,
       const velox::core::QueryConfig&) {
      return std::make_shared<JsonParseFunction>();
    });

} // namespace facebook::velox::functions
