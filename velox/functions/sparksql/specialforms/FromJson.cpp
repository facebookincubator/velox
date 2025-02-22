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

#include "velox/functions/sparksql/specialforms/FromJson.h"

#include <limits>
#include <stdexcept>

#include "velox/expression/EvalCtx.h"
#include "velox/expression/SpecialForm.h"
#include "velox/expression/VectorWriters.h"
#include "velox/functions/prestosql/json/SIMDJsonUtil.h"

using namespace facebook::velox::exec;

namespace facebook::velox::functions::sparksql {
namespace {

// Struct for extracting JSON data and writing it with type-specific handling.
template <typename Input>
struct ExtractJsonTypeImpl {
  template <TypeKind kind>
  static simdjson::error_code
  apply(Input input, exec::GenericWriter& writer, bool isRoot) {
    return KindDispatcher<kind>::apply(input, writer, isRoot);
  }

 private:
  // Dummy is needed because full/explicit specialization is not allowed inside
  // class.
  template <TypeKind kind, typename Dummy = void>
  struct KindDispatcher {
    static simdjson::error_code apply(Input, exec::GenericWriter&, bool) {
      VELOX_NYI("Parse json to {} is not supported.", TypeTraits<kind>::name);
      return simdjson::error_code::UNEXPECTED_ERROR;
    }
  };

  template <typename Dummy>
  struct KindDispatcher<TypeKind::VARCHAR, Dummy> {
    static simdjson::error_code
    apply(Input value, exec::GenericWriter& writer, bool /*isRoot*/) {
      SIMDJSON_ASSIGN_OR_RAISE(auto type, value.type());
      std::string_view s;
      if (type == simdjson::ondemand::json_type::string) {
        SIMDJSON_ASSIGN_OR_RAISE(s, value.get_string());
      } else {
        s = value.raw_json();
      }
      writer.castTo<Varchar>().append(s);
      return simdjson::SUCCESS;
    }
  };

  template <typename Dummy>
  struct KindDispatcher<TypeKind::BOOLEAN, Dummy> {
    static simdjson::error_code
    apply(Input value, exec::GenericWriter& writer, bool /*isRoot*/) {
      SIMDJSON_ASSIGN_OR_RAISE(auto type, value.type());
      if (type == simdjson::ondemand::json_type::boolean) {
        auto& w = writer.castTo<bool>();
        SIMDJSON_ASSIGN_OR_RAISE(w, value.get_bool());
        return simdjson::SUCCESS;
      }
      return simdjson::INCORRECT_TYPE;
    }
  };

  template <typename Dummy>
  struct KindDispatcher<TypeKind::TINYINT, Dummy> {
    static simdjson::error_code
    apply(Input value, exec::GenericWriter& writer, bool /*isRoot*/) {
      return castJsonToInt<int8_t>(value, writer);
    }
  };

  template <typename Dummy>
  struct KindDispatcher<TypeKind::SMALLINT, Dummy> {
    static simdjson::error_code
    apply(Input value, exec::GenericWriter& writer, bool /*isRoot*/) {
      return castJsonToInt<int16_t>(value, writer);
    }
  };

  template <typename Dummy>
  struct KindDispatcher<TypeKind::INTEGER, Dummy> {
    static simdjson::error_code
    apply(Input value, exec::GenericWriter& writer, bool /*isRoot*/) {
      return castJsonToInt<int32_t>(value, writer);
    }
  };

  template <typename Dummy>
  struct KindDispatcher<TypeKind::BIGINT, Dummy> {
    static simdjson::error_code
    apply(Input value, exec::GenericWriter& writer, bool /*isRoot*/) {
      return castJsonToInt<int64_t>(value, writer);
    }
  };

  template <typename Dummy>
  struct KindDispatcher<TypeKind::REAL, Dummy> {
    static simdjson::error_code
    apply(Input value, exec::GenericWriter& writer, bool /*isRoot*/) {
      return castJsonToFloatingPoint<float>(value, writer);
    }
  };

  template <typename Dummy>
  struct KindDispatcher<TypeKind::DOUBLE, Dummy> {
    static simdjson::error_code
    apply(Input value, exec::GenericWriter& writer, bool /*isRoot*/) {
      return castJsonToFloatingPoint<double>(value, writer);
    }
  };

  template <typename Dummy>
  struct KindDispatcher<TypeKind::ARRAY, Dummy> {
    static simdjson::error_code
    apply(Input value, exec::GenericWriter& writer, bool isRoot) {
      auto& writerTyped = writer.castTo<Array<Any>>();
      const auto& elementType = writer.type()->childAt(0);
      SIMDJSON_ASSIGN_OR_RAISE(auto type, value.type());
      if (type == simdjson::ondemand::json_type::array) {
        SIMDJSON_ASSIGN_OR_RAISE(auto array, value.get_array());
        for (const auto& elementResult : array) {
          SIMDJSON_ASSIGN_OR_RAISE(auto element, elementResult);
          // If casting to array of JSON, nulls in array elements should become
          // the JSON text "null".
          if (element.is_null()) {
            writerTyped.add_null();
          } else {
            SIMDJSON_TRY(VELOX_DYNAMIC_TYPE_DISPATCH(
                ExtractJsonTypeImpl<simdjson::ondemand::value>::apply,
                elementType->kind(),
                element,
                writerTyped.add_item(),
                false));
          }
        }
      } else if (elementType->kind() == TypeKind::ROW && isRoot) {
        SIMDJSON_TRY(VELOX_DYNAMIC_TYPE_DISPATCH(
            ExtractJsonTypeImpl<simdjson::ondemand::value>::apply,
            elementType->kind(),
            value,
            writerTyped.add_item(),
            false));
      } else {
        return simdjson::INCORRECT_TYPE;
      }
      return simdjson::SUCCESS;
    }
  };

  template <typename Dummy>
  struct KindDispatcher<TypeKind::MAP, Dummy> {
    static simdjson::error_code
    apply(Input value, exec::GenericWriter& writer, bool /*isRoot*/) {
      auto& writerTyped = writer.castTo<Map<Any, Any>>();
      const auto& valueType = writer.type()->childAt(1);
      SIMDJSON_ASSIGN_OR_RAISE(auto object, value.get_object());
      for (const auto& fieldResult : object) {
        SIMDJSON_ASSIGN_OR_RAISE(auto field, fieldResult);
        SIMDJSON_ASSIGN_OR_RAISE(auto key, field.unescaped_key(true));
        // If casting to map of JSON values, nulls in map values should become
        // the JSON text "null".
        if (field.value().is_null()) {
          writerTyped.add_null().castTo<Varchar>().append(key);
        } else {
          auto writers = writerTyped.add_item();
          std::get<0>(writers).castTo<Varchar>().append(key);
          SIMDJSON_TRY(VELOX_DYNAMIC_TYPE_DISPATCH(
              ExtractJsonTypeImpl<simdjson::ondemand::value>::apply,
              valueType->kind(),
              field.value(),
              std::get<1>(writers),
              false));
        }
      }
      return simdjson::SUCCESS;
    }
  };

  template <typename Dummy>
  struct KindDispatcher<TypeKind::ROW, Dummy> {
    static simdjson::error_code
    apply(Input value, exec::GenericWriter& writer, bool isRoot) {
      const auto& rowType = writer.type()->asRow();
      auto& writerTyped = writer.castTo<DynamicRow>();
      if (value.type().error() != ::simdjson::SUCCESS) {
        writerTyped.set_null_at(0);
        return simdjson::SUCCESS;
      }
      const auto type = value.type().value_unsafe();
      if (type == simdjson::ondemand::json_type::object) {
        SIMDJSON_ASSIGN_OR_RAISE(auto object, value.get_object());

        const auto& names = rowType.names();
        bool allFieldsAreAscii =
            std::all_of(names.begin(), names.end(), [](const auto& name) {
              return functions::stringCore::isAscii(name.data(), name.size());
            });

        auto fieldIndices = makeFieldIndicesMap(rowType, allFieldsAreAscii);

        std::string key;
        for (const auto& fieldResult : object) {
          if (fieldResult.error() != ::simdjson::SUCCESS) {
            continue;
          }
          auto field = fieldResult.value_unsafe();
          if (!field.value().is_null()) {
            SIMDJSON_ASSIGN_OR_RAISE(key, field.unescaped_key(true));

            if (allFieldsAreAscii) {
              folly::toLowerAscii(key);
            } else {
              boost::algorithm::to_lower(key);
            }
            auto it = fieldIndices.find(key);
            if (it != fieldIndices.end() && it->second >= 0) {
              const auto index = it->second;
              it->second = -1;

              const auto res = VELOX_DYNAMIC_TYPE_DISPATCH(
                  ExtractJsonTypeImpl<simdjson::ondemand::value>::apply,
                  rowType.childAt(index)->kind(),
                  field.value(),
                  writerTyped.get_writer_at(index),
                  false);
              if (res != simdjson::SUCCESS) {
                writerTyped.set_null_at(index);
              }
            }
          }
        }

        for (const auto& [_, index] : fieldIndices) {
          if (index >= 0) {
            writerTyped.set_null_at(index);
          }
        }
      } else {
        // Handle other JSON types: set null to the writer if it's the root doc,
        // otherwise return INCORRECT_TYPE to the caller.
        if (isRoot) {
          writerTyped.set_null_at(0);
          return simdjson::SUCCESS;
        } else {
          return simdjson::INCORRECT_TYPE;
        }
      }
      return simdjson::SUCCESS;
    }
  };

  template <typename T>
  static simdjson::error_code castJsonToInt(
      Input value,
      exec::GenericWriter& writer) {
    SIMDJSON_ASSIGN_OR_RAISE(auto type, value.type());
    switch (type) {
      case simdjson::ondemand::json_type::number: {
        SIMDJSON_ASSIGN_OR_RAISE(auto num, value.get_number());
        switch (num.get_number_type()) {
          case simdjson::ondemand::number_type::signed_integer:
            return convertIfInRange<T>(num.get_int64(), writer);
          case simdjson::ondemand::number_type::unsigned_integer:
            return simdjson::NUMBER_OUT_OF_RANGE;
          default:
            return simdjson::INCORRECT_TYPE;
        }
      }
      default:
        return simdjson::INCORRECT_TYPE;
    }
    return simdjson::SUCCESS;
  }

  // Casts a JSON value to a float point, handling both numeric special cases
  // for NaN and Infinity.
  template <typename T>
  static simdjson::error_code castJsonToFloatingPoint(
      Input value,
      exec::GenericWriter& writer) {
    SIMDJSON_ASSIGN_OR_RAISE(auto type, value.type());
    switch (type) {
      case simdjson::ondemand::json_type::number: {
        SIMDJSON_ASSIGN_OR_RAISE(auto num, value.get_double());
        return convertIfInRange<T>(num, writer);
      }
      case simdjson::ondemand::json_type::string: {
        SIMDJSON_ASSIGN_OR_RAISE(auto s, value.get_string());
        constexpr T kNaN = std::numeric_limits<T>::quiet_NaN();
        constexpr T kInf = std::numeric_limits<T>::infinity();
        if (s == "NaN") {
          writer.castTo<T>() = kNaN;
        } else if (s == "+INF" || s == "+Infinity" || s == "Infinity") {
          writer.castTo<T>() = kInf;
        } else if (s == "-INF" || s == "-Infinity") {
          writer.castTo<T>() = -kInf;
        } else {
          return simdjson::INCORRECT_TYPE;
        }
        break;
      }
      default:
        return simdjson::INCORRECT_TYPE;
    }
    return simdjson::SUCCESS;
  }

  template <typename To, typename From>
  static simdjson::error_code convertIfInRange(
      From x,
      exec::GenericWriter& writer) {
    static_assert(std::is_signed_v<From> && std::is_signed_v<To>);
    if constexpr (sizeof(To) < sizeof(From)) {
      constexpr From kMin = std::numeric_limits<To>::lowest();
      constexpr From kMax = std::numeric_limits<To>::max();
      if (!(kMin <= x && x <= kMax)) {
        return simdjson::NUMBER_OUT_OF_RANGE;
      }
    }
    writer.castTo<To>() = x;
    return simdjson::SUCCESS;
  }

  // Creates a map of lower case field names to their indices in the row type.
  static folly::F14FastMap<std::string, int32_t> makeFieldIndicesMap(
      const RowType& rowType,
      bool allFieldsAreAscii) {
    folly::F14FastMap<std::string, int32_t> fieldIndices;
    const auto size = rowType.size();
    for (auto i = 0; i < size; ++i) {
      std::string key = rowType.nameOf(i);
      if (allFieldsAreAscii) {
        folly::toLowerAscii(key);
      } else {
        boost::algorithm::to_lower(key);
      }

      fieldIndices[key] = i;
    }

    return fieldIndices;
  }
};

/// @brief Parses a JSON string into the specified data type. Supports ROW,
/// ARRAY, and MAP as root types. Key Behavior:
/// - Failure Handling: Returns `NULL` for invalid JSON or incompatible values.
/// - Boolean: Only `true` and `false` are valid; others return `NULL`.
/// - Integral Types: Accepts only integers; floats or strings return `NULL`.
/// - Float/Double: All numbers are valid; strings like `"NaN"`, `"+INF"`,
/// `"+Infinity"`, `"Infinity"`, `"-INF"`, `"-Infinity"` are accepted, others
/// return `NULL`.
/// - Array: Accepts JSON objects only if the array is the root type with ROW
/// child type.
/// - Map: Keys must be `VARCHAR` type.
/// - Row: Partial parsing is supported, but JSON arrays cannot be parsed into a
/// ROW type.
template <TypeKind kind>
class FromJsonFunction final : public exec::VectorFunction {
 public:
  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args, // Not using const ref so we can reuse args
      const TypePtr& outputType,
      exec::EvalCtx& context,
      VectorPtr& result) const final {
    VELOX_USER_CHECK(
        args[0]->isConstantEncoding() || args[0]->isFlatEncoding(),
        "Single-arg deterministic functions receive their only argument as flat or constant vector.");
    context.ensureWritable(rows, outputType, result);
    result->clearNulls(rows);
    if (args[0]->isConstantEncoding()) {
      parseJsonConstant(args[0], context, rows, *result);
    } else {
      parseJsonFlat(args[0], context, rows, *result);
    }
  }

 private:
  void parseJsonConstant(
      VectorPtr& input,
      exec::EvalCtx& context,
      const SelectivityVector& rows,
      BaseVector& result) const {
    // Result is guaranteed to be a flat writable vector.
    auto* flatResult = result.as<typename KindToFlatVector<kind>::type>();
    exec::VectorWriter<Any> writer;
    writer.init(*flatResult);
    const auto constInput = input->asUnchecked<ConstantVector<StringView>>();
    if (constInput->isNullAt(0)) {
      context.applyToSelectedNoThrow(rows, [&](auto row) {
        writer.setOffset(row);
        writer.commitNull();
      });
    } else {
      const auto constant = constInput->valueAt(0);
      paddedInput_.resize(constant.size() + simdjson::SIMDJSON_PADDING);
      memcpy(paddedInput_.data(), constant.data(), constant.size());
      simdjson::padded_string_view paddedInput(
          paddedInput_.data(), constant.size(), paddedInput_.size());

      simdjson::ondemand::document jsonDoc;
      auto error = simdjsonParse(paddedInput).get(jsonDoc);

      context.applyToSelectedNoThrow(rows, [&](auto row) {
        writer.setOffset(row);
        if (error != simdjson::SUCCESS ||
            extractJsonToWriter(jsonDoc, writer) != simdjson::SUCCESS) {
          writer.commitNull();
        }
      });
    }

    writer.finish();
  }

  void parseJsonFlat(
      VectorPtr& input,
      exec::EvalCtx& context,
      const SelectivityVector& rows,
      BaseVector& result) const {
    auto* flatResult = result.as<typename KindToFlatVector<kind>::type>();
    exec::VectorWriter<Any> writer;
    writer.init(*flatResult);
    auto* inputVector = input->asUnchecked<FlatVector<StringView>>();
    size_t maxSize = 0;
    rows.applyToSelected([&](auto row) {
      if (inputVector->isNullAt(row)) {
        return;
      }
      const auto& input = inputVector->valueAt(row);
      maxSize = std::max(maxSize, input.size());
    });
    paddedInput_.resize(maxSize + simdjson::SIMDJSON_PADDING);
    context.applyToSelectedNoThrow(rows, [&](auto row) {
      writer.setOffset(row);
      if (inputVector->isNullAt(row)) {
        writer.commitNull();
        return;
      }
      const auto& input = inputVector->valueAt(row);
      memcpy(paddedInput_.data(), input.data(), input.size());
      simdjson::padded_string_view paddedInput(
          paddedInput_.data(), input.size(), paddedInput_.size());
      simdjson::ondemand::document doc;
      auto error = simdjsonParse(paddedInput).get(doc);
      if (error != simdjson::SUCCESS ||
          extractJsonToWriter(doc, writer) != simdjson::SUCCESS) {
        writer.commitNull();
      }
    });
    writer.finish();
  }

  // Extracts data from json doc and writes it to writer.
  static simdjson::error_code extractJsonToWriter(
      simdjson::ondemand::document& doc,
      exec::VectorWriter<Any>& writer) {
    if (doc.is_null()) {
      writer.commitNull();
    } else {
      SIMDJSON_TRY(
          ExtractJsonTypeImpl<simdjson::ondemand::document&>::apply<kind>(
              doc, writer.current(), true));
      writer.commit(true);
    }
    return simdjson::SUCCESS;
  }

  // The buffer with extra bytes for parser::parse(),
  mutable std::string paddedInput_;
};

/// Determines whether a given type is supported.
/// @param isRootType A flag indicating whether the type is the root type in
/// the evaluation context. Only ROW, ARRAY, and MAP are allowed as root types;
/// this flag helps differentiate such cases.
bool isSupportedType(const TypePtr& type, bool isRootType) {
  switch (type->kind()) {
    case TypeKind::ARRAY: {
      return isSupportedType(type->childAt(0), false);
    }
    case TypeKind::ROW: {
      for (const auto& child : asRowType(type)->children()) {
        if (!isSupportedType(child, false)) {
          return false;
        }
      }
      return true;
    }
    case TypeKind::MAP: {
      return (
          type->childAt(0)->kind() == TypeKind::VARCHAR &&
          isSupportedType(type->childAt(1), false));
    }
    case TypeKind::BIGINT: {
      if (type->isDecimal()) {
        return false;
      }
      return !isRootType;
    }
    case TypeKind::INTEGER: {
      if (type->isDate()) {
        return false;
      }
      return !isRootType;
    }
    case TypeKind::BOOLEAN:
    case TypeKind::SMALLINT:
    case TypeKind::TINYINT:
    case TypeKind::DOUBLE:
    case TypeKind::REAL:
    case TypeKind::VARCHAR: {
      return !isRootType;
    }
    default:
      return false;
  }
}

} // namespace

TypePtr FromJsonCallToSpecialForm::resolveType(
    const std::vector<TypePtr>& /*argTypes*/) {
  VELOX_FAIL("from_json function does not support type resolution.");
}

exec::ExprPtr FromJsonCallToSpecialForm::constructSpecialForm(
    const TypePtr& type,
    std::vector<exec::ExprPtr>&& args,
    bool trackCpuUsage,
    const core::QueryConfig& /*config*/) {
  VELOX_USER_CHECK_EQ(args.size(), 1, "from_json expects one argument.");
  VELOX_USER_CHECK_EQ(
      args[0]->type()->kind(),
      TypeKind::VARCHAR,
      "The first argument of from_json should be of varchar type.");

  VELOX_USER_CHECK(
      isSupportedType(type, true), "Unsupported type {}.", type->toString());

  std::shared_ptr<exec::VectorFunction> func;
  if (type->kind() == TypeKind::ARRAY) {
    func = std::make_shared<FromJsonFunction<TypeKind::ARRAY>>();
  } else if (type->kind() == TypeKind::MAP) {
    func = std::make_shared<FromJsonFunction<TypeKind::MAP>>();
  } else {
    func = std::make_shared<FromJsonFunction<TypeKind::ROW>>();
  }

  return std::make_shared<exec::Expr>(
      type,
      std::move(args),
      func,
      exec::VectorFunctionMetadata{},
      kFromJson,
      trackCpuUsage);
}
} // namespace facebook::velox::functions::sparksql
