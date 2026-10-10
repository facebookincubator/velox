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

#include <algorithm>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>

#include "velox/expression/ConstantExpr.h"
#include "velox/expression/EvalCtx.h"
#include "velox/expression/SpecialForm.h"
#include "velox/expression/VectorWriters.h"
#include "velox/functions/lib/DateTimeFormatter.h"
#include "velox/functions/lib/string/StringCore.h"
#include "velox/functions/lib/string/StringImpl.h"
#include "velox/functions/prestosql/json/SIMDJsonUtil.h"
#include "velox/functions/sparksql/SparkQueryConfig.h"
#include "velox/type/DecimalUtil.h"
#include "velox/type/TimestampConversion.h"
#include "velox/type/tz/TimeZoneMap.h"

using namespace facebook::velox::exec;

namespace facebook::velox::functions::sparksql {
namespace {

// Bound recursive schema traversal to prevent worker stack overflow.
static constexpr int kMaxSchemaDepth = 1000;

enum class ParseMode {
  kPermissive,
  kFailFast,
};

struct FromJsonConfig {
  bool allowNonNumericNumbers = true;
  bool enablePartialResults = true;
  ParseMode mode = ParseMode::kPermissive;
  std::optional<std::string> columnNameOfCorruptRecord;
  std::optional<column_index_t> corruptRecordIndex;
  std::optional<std::string> dateFormat;
  std::optional<std::string> timestampFormat;
  bool sparkLegacyDateFormatter = false;
  std::string sessionTimezone;
  bool caseSensitiveFieldMatch = false;
};

// Per-apply parsing state. Pointer members refer to objects owned by the
// FromJsonFunction instance and remain valid for the apply() call.
struct FromJsonParseContext {
  bool allowNonNumericNumbers = true;
  const DateTimeFormatter* dateFormatter = nullptr;
  const DateTimeFormatter* timestampFormatter = nullptr;
  bool hasDateFormat = false;
  bool hasTimestampFormat = false;
  bool sparkLegacyDateFormatter = false;
  bool enablePartialResults = true;
  bool caseSensitiveFieldMatch = false;
  // Session timezone for TIMESTAMP values without an explicit zone.
  const tz::TimeZone* sessionTimezone = nullptr;

  bool hadPartialFailure = false;
};

struct FieldInfo {
  column_index_t fieldIndex;
  column_index_t nodeIndex;
};

// Struct to store schema information for a JSON row, used for efficient field
// lookup and null handling.
struct JsonRowSchemaInfo {
  // Unique key for this schema info, computed from the nesting level and field
  // index.
  uint64_t key;

  // Indicates if all field names in this row are ASCII (for optimized
  // case-insensitive comparison).
  bool allFieldsAreAscii;

  // Shared pointer to a vector indicating which fields are missing in the
  // current JSON object.
  std::shared_ptr<std::vector<bool>> isFieldMissing;

  // Maps field names (lowercased if case-insensitive) to their FieldInfo
  // (column index + node index), enabling a single hash lookup per field.
  folly::F14FastMap<std::string, FieldInfo> fieldMap;

  JsonRowSchemaInfo(
      uint64_t key,
      bool allFieldsAreAscii,
      std::shared_ptr<std::vector<bool>>&& isFieldMissing,
      folly::F14FastMap<std::string, FieldInfo>&& fieldMap)
      : key(key),
        allFieldsAreAscii(allFieldsAreAscii),
        isFieldMissing(std::move(isFieldMissing)),
        fieldMap(std::move(fieldMap)) {}
};

// Struct for extracting JSON data and writing it with type-specific handling.
template <typename Input>
struct ExtractJsonTypeImpl {
  template <TypeKind kind>
  static simdjson::error_code apply(
      Input input,
      exec::GenericWriter& writer,
      bool isRoot,
      const folly::F14FastMap<int64_t, JsonRowSchemaInfo>& jsonRowSchemaInfo,
      column_index_t nodeIndex,
      FromJsonParseContext& parseContext) {
    return KindDispatcher<kind>::apply(
        input, writer, isRoot, jsonRowSchemaInfo, nodeIndex, parseContext);
  }

 private:
  // Dummy is needed because full/explicit specialization is not allowed inside
  // class.
  template <TypeKind kind, typename Dummy = void>
  struct KindDispatcher {
    static simdjson::error_code apply(
        Input,
        exec::GenericWriter&,
        bool /*isRoot*/,
        const folly::
            F14FastMap<int64_t, JsonRowSchemaInfo>& /*jsonRowSchemaInfo*/,
        column_index_t /*nodeIndex*/,
        const FromJsonParseContext& /*parseContext*/) {
      VELOX_NYI("Parse json to {} is not supported.", TypeTraits<kind>::name);
      return simdjson::error_code::UNEXPECTED_ERROR;
    }
  };

  template <typename Dummy>
  struct KindDispatcher<TypeKind::VARCHAR, Dummy> {
    static simdjson::error_code apply(
        Input value,
        exec::GenericWriter& writer,
        bool /*isRoot*/,
        const folly::
            F14FastMap<int64_t, JsonRowSchemaInfo>& /*jsonRowSchemaInfo*/,
        column_index_t /*nodeIndex*/,
        const FromJsonParseContext& /*parseContext*/) {
      SIMDJSON_ASSIGN_OR_RAISE(auto type, value.type());
      std::string_view s;
      if (type == simdjson::ondemand::json_type::string) {
        SIMDJSON_ASSIGN_OR_RAISE(s, value.get_string());
      } else {
        s = value.raw_json();
      }
      // Assignment replaces the current value, which is required for
      // last-value-wins duplicate-key handling.
      writer.castTo<Varchar>() = s;
      return simdjson::SUCCESS;
    }
  };

  template <typename Dummy>
  struct KindDispatcher<TypeKind::BOOLEAN, Dummy> {
    static simdjson::error_code apply(
        Input value,
        exec::GenericWriter& writer,
        bool /*isRoot*/,
        const folly::
            F14FastMap<int64_t, JsonRowSchemaInfo>& /*jsonRowSchemaInfo*/,
        column_index_t /*nodeIndex*/,
        const FromJsonParseContext& /*parseContext*/) {
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
    static simdjson::error_code apply(
        Input value,
        exec::GenericWriter& writer,
        bool /*isRoot*/,
        const folly::
            F14FastMap<int64_t, JsonRowSchemaInfo>& /*jsonRowSchemaInfo*/,
        column_index_t /*nodeIndex*/,
        const FromJsonParseContext& /*parseContext*/) {
      return castJsonToInt<int8_t>(value, writer);
    }
  };

  template <typename Dummy>
  struct KindDispatcher<TypeKind::SMALLINT, Dummy> {
    static simdjson::error_code apply(
        Input value,
        exec::GenericWriter& writer,
        bool /*isRoot*/,
        const folly::
            F14FastMap<int64_t, JsonRowSchemaInfo>& /*jsonRowSchemaInfo*/,
        column_index_t /*nodeIndex*/,
        const FromJsonParseContext& /*parseContext*/) {
      return castJsonToInt<int16_t>(value, writer);
    }
  };

  template <typename Dummy>
  struct KindDispatcher<TypeKind::INTEGER, Dummy> {
    static simdjson::error_code apply(
        Input value,
        exec::GenericWriter& writer,
        bool /*isRoot*/,
        const folly::
            F14FastMap<int64_t, JsonRowSchemaInfo>& /*jsonRowSchemaInfo*/,
        column_index_t /*nodeIndex*/,
        const FromJsonParseContext& parseContext) {
      if (writer.type() == DATE()) {
        return castJsonToDate(value, writer, parseContext);
      }
      return castJsonToInt<int32_t>(value, writer);
    }
  };

  template <typename Dummy>
  struct KindDispatcher<TypeKind::BIGINT, Dummy> {
    static simdjson::error_code apply(
        Input value,
        exec::GenericWriter& writer,
        bool /*isRoot*/,
        const folly::
            F14FastMap<int64_t, JsonRowSchemaInfo>& /*jsonRowSchemaInfo*/,
        column_index_t /*nodeIndex*/,
        const FromJsonParseContext& /*parseContext*/) {
      if (writer.type()->isShortDecimal()) {
        return castJsonToDecimal<int64_t>(value, writer);
      }
      return castJsonToInt<int64_t>(value, writer);
    }
  };

  template <typename Dummy>
  struct KindDispatcher<TypeKind::HUGEINT, Dummy> {
    static simdjson::error_code apply(
        Input value,
        exec::GenericWriter& writer,
        bool /*isRoot*/,
        const folly::
            F14FastMap<int64_t, JsonRowSchemaInfo>& /*jsonRowSchemaInfo*/,
        column_index_t /*nodeIndex*/,
        const FromJsonParseContext& /*parseContext*/) {
      VELOX_CHECK(writer.type()->isLongDecimal());
      return castJsonToDecimal<int128_t>(value, writer);
    }
  };

  template <typename Dummy>
  struct KindDispatcher<TypeKind::REAL, Dummy> {
    static simdjson::error_code apply(
        Input value,
        exec::GenericWriter& writer,
        bool /*isRoot*/,
        const folly::
            F14FastMap<int64_t, JsonRowSchemaInfo>& /*jsonRowSchemaInfo*/,
        column_index_t /*nodeIndex*/,
        const FromJsonParseContext& parseContext) {
      return castJsonToFloatingPoint<float>(value, writer, parseContext);
    }
  };

  template <typename Dummy>
  struct KindDispatcher<TypeKind::DOUBLE, Dummy> {
    static simdjson::error_code apply(
        Input value,
        exec::GenericWriter& writer,
        bool /*isRoot*/,
        const folly::
            F14FastMap<int64_t, JsonRowSchemaInfo>& /*jsonRowSchemaInfo*/,
        column_index_t /*nodeIndex*/,
        const FromJsonParseContext& parseContext) {
      return castJsonToFloatingPoint<double>(value, writer, parseContext);
    }
  };

  template <typename Dummy>
  struct KindDispatcher<TypeKind::TIMESTAMP, Dummy> {
    static simdjson::error_code apply(
        Input value,
        exec::GenericWriter& writer,
        bool /*isRoot*/,
        const folly::
            F14FastMap<int64_t, JsonRowSchemaInfo>& /*jsonRowSchemaInfo*/,
        column_index_t /*nodeIndex*/,
        const FromJsonParseContext& parseContext) {
      return castJsonToTimestamp(value, writer, parseContext);
    }
  };

  template <typename Dummy>
  struct KindDispatcher<TypeKind::ARRAY, Dummy> {
    static simdjson::error_code apply(
        Input value,
        exec::GenericWriter& writer,
        bool isRoot,
        const folly::F14FastMap<int64_t, JsonRowSchemaInfo>& jsonRowSchemaInfo,
        column_index_t nodeIndex,
        FromJsonParseContext& parseContext) {
      auto& writerTyped = writer.castTo<Array<Any>>();
      const auto& elementType = writer.type()->childAt(0);
      SIMDJSON_ASSIGN_OR_RAISE(auto type, value.type());
      if (type == simdjson::ondemand::json_type::array) {
        SIMDJSON_ASSIGN_OR_RAISE(auto array, value.get_array());
        for (const auto& elementResult : array) {
          SIMDJSON_ASSIGN_OR_RAISE(auto element, elementResult);
          SIMDJSON_ASSIGN_OR_RAISE(auto isNull, element.is_null());
          // If casting to array of JSON, nulls in array elements should become
          // the JSON text "null".
          if (isNull) {
            writerTyped.add_null();
          } else {
            SIMDJSON_TRY(VELOX_DYNAMIC_TYPE_DISPATCH(
                ExtractJsonTypeImpl<simdjson::ondemand::value>::apply,
                elementType->kind(),
                element,
                writerTyped.add_item(),
                false,
                jsonRowSchemaInfo,
                nodeIndex + 1,
                parseContext));
          }
        }
      } else if (
          type == simdjson::ondemand::json_type::object &&
          elementType->kind() == TypeKind::ROW && isRoot) {
        SIMDJSON_TRY(VELOX_DYNAMIC_TYPE_DISPATCH(
            ExtractJsonTypeImpl<simdjson::ondemand::value>::apply,
            elementType->kind(),
            value,
            writerTyped.add_item(),
            false,
            jsonRowSchemaInfo,
            nodeIndex + 1,
            parseContext));
      } else {
        return simdjson::INCORRECT_TYPE;
      }
      return simdjson::SUCCESS;
    }
  };

  template <typename Dummy>
  struct KindDispatcher<TypeKind::MAP, Dummy> {
    static simdjson::error_code apply(
        Input value,
        exec::GenericWriter& writer,
        bool /*isRoot*/,
        const folly::F14FastMap<int64_t, JsonRowSchemaInfo>& jsonRowSchemaInfo,
        column_index_t nodeIndex,
        FromJsonParseContext& parseContext) {
      auto& writerTyped = writer.castTo<Map<Any, Any>>();
      const auto& valueType = writer.type()->childAt(1);
      SIMDJSON_ASSIGN_OR_RAISE(auto object, value.get_object());
      for (const auto& fieldResult : object) {
        SIMDJSON_ASSIGN_OR_RAISE(auto field, fieldResult);
        SIMDJSON_ASSIGN_OR_RAISE(auto key, field.unescaped_key(true));
        auto fieldValue = field.value();
        SIMDJSON_ASSIGN_OR_RAISE(auto isNull, fieldValue.is_null());
        // If casting to map of JSON values, nulls in map values should become
        // the JSON text "null".
        if (isNull) {
          writerTyped.add_null().castTo<Varchar>().append(key);
        } else {
          auto writers = writerTyped.add_item();
          std::get<0>(writers).castTo<Varchar>().append(key);
          SIMDJSON_TRY(VELOX_DYNAMIC_TYPE_DISPATCH(
              ExtractJsonTypeImpl<simdjson::ondemand::value>::apply,
              valueType->kind(),
              fieldValue,
              std::get<1>(writers),
              false,
              jsonRowSchemaInfo,
              nodeIndex + 1,
              parseContext));
        }
      }
      return simdjson::SUCCESS;
    }
  };

  template <typename Dummy>
  struct KindDispatcher<TypeKind::ROW, Dummy> {
    static simdjson::error_code apply(
        Input value,
        exec::GenericWriter& writer,
        bool isRoot,
        const folly::F14FastMap<int64_t, JsonRowSchemaInfo>& jsonRowSchemaInfo,
        column_index_t nodeIndex,
        FromJsonParseContext& parseContext) {
      const auto& rowType = writer.type()->asRow();
      auto& writerTyped = writer.castTo<DynamicRow>();
      auto typeResult = value.type();
      if (typeResult.error() != ::simdjson::SUCCESS) {
        // Propagate the failure so the caller can apply PERMISSIVE, FAILFAST,
        // and corrupt-record handling to the whole root value.
        return simdjson::INCORRECT_TYPE;
      }
      const auto type = typeResult.value_unsafe();
      if (type == simdjson::ondemand::json_type::object) {
        SIMDJSON_ASSIGN_OR_RAISE(auto object, value.get_object());
        const auto& schemaInfo = jsonRowSchemaInfo.at(nodeIndex);
        const auto& isFieldMissing = schemaInfo.isFieldMissing;
        const auto& fieldMap = schemaInfo.fieldMap;
        std::fill(isFieldMissing->begin(), isFieldMissing->end(), true);
        std::string key;
        for (const auto& fieldResult : object) {
          SIMDJSON_ASSIGN_OR_RAISE(auto field, fieldResult);
          SIMDJSON_ASSIGN_OR_RAISE(key, field.unescaped_key(true));
          auto fieldValue = field.value();

          // Match JSON key against schema field names.
          if (!parseContext.caseSensitiveFieldMatch) {
            // Legacy case-insensitive matching: lowercase the key.
            if (schemaInfo.allFieldsAreAscii) {
              folly::toLowerAscii(key);
            } else {
              boost::algorithm::to_lower(key);
            }
          }
          auto it = fieldMap.find(key);
          if (it == fieldMap.end()) {
            // Dropping a field does not exempt its value from JSON validation.
            SIMDJSON_TRY(skipJsonValue(fieldValue, parseContext));
            continue;
          }
          const auto index = it->second.fieldIndex;
          const auto childNodeIndex = it->second.nodeIndex;
          SIMDJSON_ASSIGN_OR_RAISE(auto isNull, fieldValue.is_null());
          // Preserve legacy first-non-null-wins behavior for case-insensitive
          // matching. Case-sensitive matching follows Spark's last-wins
          // behavior, including an explicit null as the final value.
          const bool isDuplicate = !isFieldMissing->at(index);

          if (!parseContext.caseSensitiveFieldMatch) {
            // First-writer-wins. A populated field, or a null value, is
            // skipped; the field is only marked populated once a non-null
            // value is written.
            if (isDuplicate) {
              SIMDJSON_TRY(skipJsonValue(fieldValue, parseContext));
              continue;
            }
            if (isNull) {
              continue;
            }
            isFieldMissing->at(index) = false;
          } else {
            isFieldMissing->at(index) = false;

            if (isNull) {
              writerTyped.set_null_at(index);
              continue;
            }
            // Discard state accumulated by a prior complex value before
            // writing the replacement.
            if (isDuplicate) {
              writerTyped.set_null_at(index);
            }
          }

          const auto conversionResult = VELOX_DYNAMIC_TYPE_DISPATCH(
              ExtractJsonTypeImpl<simdjson::ondemand::value>::apply,
              rowType.childAt(index)->kind(),
              fieldValue,
              writerTyped.get_writer_at(index),
              false,
              jsonRowSchemaInfo,
              childNodeIndex,
              parseContext);
          if (conversionResult != simdjson::SUCCESS) {
            writerTyped.set_null_at(index);
            // Root rows preserve successful siblings. Nested rows collapse
            // when partial results are disabled.
            if (!parseContext.enablePartialResults && !isRoot) {
              return simdjson::INCORRECT_TYPE;
            }
            parseContext.hadPartialFailure = true;
          }
        }

        for (column_index_t i = 0; i < rowType.size(); ++i) {
          if (isFieldMissing->at(i)) {
            writerTyped.set_null_at(i);
          }
        }
      } else {
        // A ROW schema requires an object root. Propagate the mismatch so the
        // caller applies the configured malformed-record policy.
        return simdjson::INCORRECT_TYPE;
      }
      return simdjson::SUCCESS;
    }
  };

  static simdjson::error_code castJsonToDate(
      Input value,
      exec::GenericWriter& writer,
      const FromJsonParseContext& parseContext) {
    SIMDJSON_ASSIGN_OR_RAISE(auto type, value.type());
    if (type != simdjson::ondemand::json_type::string) {
      return simdjson::INCORRECT_TYPE;
    }
    std::string_view s;
    SIMDJSON_ASSIGN_OR_RAISE(s, value.get_string());
    int32_t day = 0;

    // If a custom dateFormat is specified, use the pre-built formatter.
    if (parseContext.dateFormatter) {
      auto result = parseContext.dateFormatter->parse(s);
      if (result.hasError()) {
        return simdjson::INCORRECT_TYPE;
      }
      // Extract date part (days since epoch) from the parsed timestamp.
      // Spark's modern Iso8601DateFormatter parses DATE as a wall-clock value
      // (LocalDate), independent of session timezone. The Joda parser used here
      // returns a UTC-frame timestamp for date-only inputs (no zone in the
      // string), so dividing the UTC seconds by 86400 yields the same calendar
      // day. Applying any timezone shift here would produce off-by-one-day
      // errors for UTC-minus session zones (e.g., "America/Los_Angeles").
      const auto timestamp = result.value().timestamp;
      const auto seconds = timestamp.getSeconds();
      int64_t daysSinceEpoch = seconds / 86400;
      if (seconds < 0 && seconds % 86400 != 0) {
        --daysSinceEpoch; // Truncate toward negative infinity.
      }
      if (daysSinceEpoch < std::numeric_limits<int32_t>::min() ||
          daysSinceEpoch > std::numeric_limits<int32_t>::max()) {
        return simdjson::INCORRECT_TYPE;
      }
      day = static_cast<int32_t>(daysSinceEpoch);
      writer.castTo<int32_t>() = day;
      return simdjson::SUCCESS;
    }
    if (parseContext.hasDateFormat) {
      return simdjson::INCORRECT_TYPE;
    }

    // Default parsing path.
    // If the value has fewer than four digits, it is interpreted as the number
    // of days since January 1, 1970.
    if (s.size() < 4) {
      auto result = folly::tryTo<int32_t>(s);
      if (!result.hasError()) {
        day = result.value();
      } else {
        return simdjson::INCORRECT_TYPE;
      }
    } else {
      auto castResult =
          util::fromDateString(StringView(s), util::ParseMode::kSparkCast);
      if (!castResult.hasError()) {
        day = castResult.value();
      } else if (stringImpl::stringPosition</*isAscii=*/true>(s, kGMT, 1) > 0) {
        // Try converting the string view as a date value after cleaning up the
        // legacy timestamp string by removing the "GMT" string.
        std::vector<char> dateStr(s.size());
        auto size = stringCore::replace</*ignoreEmptyReplaced=*/true>(
            dateStr.data(), s, kGMT, std::string_view(), false);
        castResult = util::fromDateString(
            StringView(dateStr.data(), size), util::ParseMode::kSparkCast);
        if (!castResult.hasError()) {
          day = castResult.value();
        } else {
          return simdjson::INCORRECT_TYPE;
        }
      } else {
        return simdjson::INCORRECT_TYPE;
      }
    }
    writer.castTo<int32_t>() = day;
    return simdjson::SUCCESS;
  }

  // Converts epoch seconds to a Timestamp using Spark's
  // `getLongValue * 1000000L` semantics, including 64-bit wraparound.
  static Timestamp epochSecondsToTimestamp(int64_t seconds) {
    const auto micros = static_cast<int64_t>(
        static_cast<uint64_t>(seconds) *
        static_cast<uint64_t>(Timestamp::kMicrosecondsInSecond));
    auto wholeSeconds = micros / Timestamp::kMicrosecondsInSecond;
    auto remainderMicros = micros % Timestamp::kMicrosecondsInSecond;
    if (remainderMicros < 0) {
      --wholeSeconds;
      remainderMicros += Timestamp::kMicrosecondsInSecond;
    }
    return Timestamp(
        wholeSeconds, remainderMicros * Timestamp::kNanosecondsInMicrosecond);
  }

  // Converts a local timestamp to GMT. Local times in a DST gap shift forward
  // by the gap length and ambiguous local times use the earlier offset,
  // matching Spark's ZonedDateTime resolution. Returns false when the value is
  // outside the range supported by the time zone library.
  static bool tryToGMT(Timestamp& timestamp, const tz::TimeZone& zone) {
    try {
      const std::chrono::seconds localSeconds(timestamp.getSeconds());
      tz::validateRange(tz::time_point<std::chrono::seconds>(localSeconds));
      timestamp = Timestamp(
          zone.correct_nonexistent_time(localSeconds).count(),
          timestamp.getNanos());
      timestamp.toGMT(zone);
    } catch (const VeloxUserError&) {
      return false;
    }
    return true;
  }

  // Rejects string timestamps that cannot be represented as Spark's int64
  // microseconds. Custom corrected formatters also check the intermediate
  // whole-second multiplication before adding the fractional microseconds.
  static simdjson::error_code writeTimestamp(
      const Timestamp& timestamp,
      exec::GenericWriter& writer,
      bool checkWholeSeconds) {
    int128_t micros = static_cast<int128_t>(timestamp.getSeconds()) *
        Timestamp::kMicrosecondsInSecond;
    const auto inRange = [](int128_t value) {
      return value >= std::numeric_limits<int64_t>::min() &&
          value <= std::numeric_limits<int64_t>::max();
    };
    if (checkWholeSeconds && !inRange(micros)) {
      return simdjson::NUMBER_OUT_OF_RANGE;
    }
    micros += timestamp.getNanos() / Timestamp::kNanosecondsInMicrosecond;
    if (!inRange(micros)) {
      return simdjson::NUMBER_OUT_OF_RANGE;
    }
    writer.castTo<Timestamp>() = timestamp;
    return simdjson::SUCCESS;
  }

  // Parses a JSON string value into a Timestamp. Custom formats use the
  // pre-built formatter; otherwise Spark-cast timestamp parsing applies the
  // session timezone to values without an explicit zone.
  static simdjson::error_code castJsonToTimestamp(
      Input value,
      exec::GenericWriter& writer,
      const FromJsonParseContext& parseContext) {
    SIMDJSON_ASSIGN_OR_RAISE(auto type, value.type());
    if (type == simdjson::ondemand::json_type::number) {
      SIMDJSON_ASSIGN_OR_RAISE(auto number, value.get_number());
      switch (number.get_number_type()) {
        case simdjson::ondemand::number_type::signed_integer:
          writer.castTo<Timestamp>() =
              epochSecondsToTimestamp(number.get_int64());
          return simdjson::SUCCESS;
        case simdjson::ondemand::number_type::unsigned_integer: {
          const auto seconds = number.get_uint64();
          if (seconds > std::numeric_limits<int64_t>::max()) {
            return simdjson::NUMBER_OUT_OF_RANGE;
          }
          writer.castTo<Timestamp>() =
              epochSecondsToTimestamp(static_cast<int64_t>(seconds));
          return simdjson::SUCCESS;
        }
        default:
          return simdjson::INCORRECT_TYPE;
      }
    }
    if (type != simdjson::ondemand::json_type::string) {
      return simdjson::INCORRECT_TYPE;
    }
    std::string_view timestampText;
    SIMDJSON_ASSIGN_OR_RAISE(timestampText, value.get_string());

    if (!parseContext.timestampFormatter) {
      if (parseContext.hasTimestampFormat) {
        return simdjson::INCORRECT_TYPE;
      }
      auto result = util::fromTimestampWithTimezoneString(
          timestampText.data(),
          timestampText.size(),
          util::TimestampParseMode::kSparkCast);
      if (result.hasError()) {
        return simdjson::INCORRECT_TYPE;
      }
      auto parsed = result.value();
      const auto* zone = parsed.timeZone;
      if (zone == nullptr && !parsed.offsetMillis.has_value()) {
        zone = parseContext.sessionTimezone;
      }
      if (zone == nullptr) {
        return writeTimestamp(
            util::fromParsedTimestampWithTimeZone(parsed, nullptr),
            writer,
            false);
      }
      if (!tryToGMT(parsed.timestamp, *zone)) {
        return simdjson::NUMBER_OUT_OF_RANGE;
      }
      return writeTimestamp(parsed.timestamp, writer, false);
    }

    auto result = parseContext.sparkLegacyDateFormatter
        ? parseContext.timestampFormatter->parse(timestampText)
        : parseContext.timestampFormatter->parseWithMicrosecondPrecision(
              timestampText);
    if (result.hasError()) {
      return simdjson::INCORRECT_TYPE;
    }
    auto timestamp = result.value().timestamp;
    const auto* zone = result.value().timezone ? result.value().timezone
                                               : parseContext.sessionTimezone;
    if (zone != nullptr && !tryToGMT(timestamp, *zone)) {
      return simdjson::NUMBER_OUT_OF_RANGE;
    }
    return writeTimestamp(
        timestamp, writer, !parseContext.sparkLegacyDateFormatter);
  }

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

  // Detects Spark's special floating-point token spellings without modifying
  // the writer, so disallowed values can fail before touching result state.
  template <typename T>
  static bool detectSpecialFloatingString(
      const std::string_view& s,
      T& outValue) {
    constexpr T kNaN = std::numeric_limits<T>::quiet_NaN();
    constexpr T kInf = std::numeric_limits<T>::infinity();
    if (s == "NaN") {
      outValue = kNaN;
      return true;
    }
    if (s == "+INF" || s == "+Infinity" || s == "Infinity") {
      outValue = kInf;
      return true;
    }
    if (s == "-INF" || s == "-Infinity") {
      outValue = -kInf;
      return true;
    }
    return false;
  }

  // Validates JSON number syntax before using the overflow-tolerant converter,
  // which also accepts non-JSON spellings such as "+1" and "inf".
  static bool isJsonNumber(std::string_view token) {
    size_t position = 0;
    if (position < token.size() && token[position] == '-') {
      ++position;
    }
    if (position == token.size()) {
      return false;
    }
    const auto isDigit = [](char c) { return c >= '0' && c <= '9'; };
    if (token[position] == '0') {
      ++position;
    } else {
      if (token[position] < '1' || token[position] > '9') {
        return false;
      }
      do {
        ++position;
      } while (position < token.size() && isDigit(token[position]));
    }
    if (position < token.size() && token[position] == '.') {
      const auto fractionStart = ++position;
      while (position < token.size() && isDigit(token[position])) {
        ++position;
      }
      if (position == fractionStart) {
        return false;
      }
    }
    if (position < token.size() &&
        (token[position] == 'e' || token[position] == 'E')) {
      ++position;
      if (position < token.size() &&
          (token[position] == '+' || token[position] == '-')) {
        ++position;
      }
      const auto exponentStart = position;
      while (position < token.size() && isDigit(token[position])) {
        ++position;
      }
      if (position == exponentStart) {
        return false;
      }
    }
    return position == token.size();
  }

  // Validates discarded values because on-demand iterator advancement skips
  // their contents without fully checking strings, numbers, or nested syntax.
  static simdjson::error_code skipJsonValue(
      simdjson::ondemand::value value,
      const FromJsonParseContext& parseContext) {
    std::string_view token = value.raw_json_token();
    while (!token.empty() &&
           (token.back() == ' ' || token.back() == '\t' ||
            token.back() == '\n' || token.back() == '\r')) {
      token.remove_suffix(1);
    }
    double specialValue;
    if (detectSpecialFloatingString(token, specialValue)) {
      return parseContext.allowNonNumericNumbers ? simdjson::SUCCESS
                                                 : simdjson::NUMBER_ERROR;
    }
    SIMDJSON_ASSIGN_OR_RAISE(auto type, value.type());
    if ((type == simdjson::ondemand::json_type::array ||
         type == simdjson::ondemand::json_type::object) &&
        value.current_depth() > kMaxSchemaDepth) {
      return simdjson::DEPTH_ERROR;
    }
    switch (type) {
      case simdjson::ondemand::json_type::array: {
        SIMDJSON_ASSIGN_OR_RAISE(auto array, value.get_array());
        for (auto element : array) {
          SIMDJSON_ASSIGN_OR_RAISE(auto child, element);
          SIMDJSON_TRY(skipJsonValue(child, parseContext));
        }
        return simdjson::SUCCESS;
      }
      case simdjson::ondemand::json_type::object: {
        SIMDJSON_ASSIGN_OR_RAISE(auto object, value.get_object());
        for (auto fieldResult : object) {
          SIMDJSON_ASSIGN_OR_RAISE(auto field, fieldResult);
          SIMDJSON_TRY(field.unescaped_key(true).error());
          SIMDJSON_TRY(skipJsonValue(field.value(), parseContext));
        }
        return simdjson::SUCCESS;
      }
      case simdjson::ondemand::json_type::string:
        return value.get_string().error();
      case simdjson::ondemand::json_type::number:
        return isJsonNumber(token) ? simdjson::SUCCESS : simdjson::NUMBER_ERROR;
      case simdjson::ondemand::json_type::boolean:
        return value.get_bool().error();
      case simdjson::ondemand::json_type::null:
        return value.is_null().error();
    }
    return simdjson::UNEXPECTED_ERROR;
  }

  // Casts a JSON value to a float point, handling both numeric special cases
  // for NaN and Infinity.
  template <typename T>
  static simdjson::error_code castJsonToFloatingPoint(
      Input value,
      exec::GenericWriter& writer,
      const FromJsonParseContext& parseContext) {
    std::string_view token = value.raw_json_token();
    while (!token.empty() &&
           (token.back() == ' ' || token.back() == '\t' ||
            token.back() == '\n' || token.back() == '\r')) {
      token.remove_suffix(1);
    }
    const bool isString = !token.empty() && token.front() == '"';
    if (isString) {
      SIMDJSON_ASSIGN_OR_RAISE(token, value.get_string());
    }
    T specialValue;
    if (detectSpecialFloatingString<T>(token, specialValue)) {
      if (!parseContext.allowNonNumericNumbers) {
        return simdjson::NUMBER_ERROR;
      }
      writer.castTo<T>() = specialValue;
      return simdjson::SUCCESS;
    }
    if (isString) {
      return simdjson::INCORRECT_TYPE;
    }

    auto result = value.get_double();
    if (result.error() == simdjson::SUCCESS) {
      // The option gates special tokens, not overflow of JSON numbers.
      writer.castTo<T>() = static_cast<T>(result.value_unsafe());
      return simdjson::SUCCESS;
    }

    // simdjson rejects double overflow; Spark/Jackson returns infinity.
    if (!isJsonNumber(token)) {
      return simdjson::NUMBER_ERROR;
    }
    auto castResult = util::Converter<TypeKind::DOUBLE>::tryCast(token);
    if (!castResult.hasError()) {
      writer.castTo<T>() = static_cast<T>(castResult.value());
      return simdjson::SUCCESS;
    }
    return simdjson::NUMBER_ERROR;
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

  template <typename T>
  static simdjson::error_code castJsonToDecimal(
      Input value,
      exec::GenericWriter& writer) {
    SIMDJSON_ASSIGN_OR_RAISE(auto type, value.type());
    std::string_view s;
    switch (type) {
      case simdjson::ondemand::json_type::string: {
        SIMDJSON_ASSIGN_OR_RAISE(s, value.get_string());
        break;
      }
      case simdjson::ondemand::json_type::number: {
        s = value.raw_json_token();
        break;
      }
      default:
        return simdjson::INCORRECT_TYPE;
    }
    const auto toPrecisionScale = getDecimalPrecisionScale(*writer.type());
    T decimalValue;
    const auto status = velox::DecimalUtil::castFromString<T>(
        StringView(s),
        toPrecisionScale.first,
        toPrecisionScale.second,
        decimalValue);
    if (!status.ok()) {
      return simdjson::INCORRECT_TYPE;
    }
    writer.castTo<T>() = decimalValue;
    return simdjson::SUCCESS;
  }

  constexpr static std::string_view kGMT{"GMT"};
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
  explicit FromJsonFunction(const TypePtr& type, FromJsonConfig config)
      : config_(std::move(config)) {
    column_index_t index = 0;
    constructRowSchemaInfoMap(type, index, 0);
    if (config_.dateFormat && !config_.dateFormat->empty()) {
      auto result = config_.sparkLegacyDateFormatter
          ? buildSimpleDateTimeFormatter(*config_.dateFormat, /*lenient=*/false)
          : buildJodaDateTimeFormatter(*config_.dateFormat);
      VELOX_USER_CHECK(
          !result.hasError(),
          "Invalid dateFormat '{}': {}",
          *config_.dateFormat,
          result.error().message());
      dateFormatter_ = std::move(result.value());
    }
    if (config_.timestampFormat && !config_.timestampFormat->empty()) {
      auto result = config_.sparkLegacyDateFormatter
          ? buildSimpleDateTimeFormatter(
                *config_.timestampFormat, /*lenient=*/false)
          : buildJodaDateTimeFormatter(*config_.timestampFormat);
      VELOX_USER_CHECK(
          !result.hasError(),
          "Invalid timestampFormat '{}': {}",
          *config_.timestampFormat,
          result.error().message());
      timestampFormatter_ = std::move(result.value());
    }
    // Values without an explicit zone use the configured session timezone.
    if (!config_.sessionTimezone.empty()) {
      auto* timezone = tz::locateZone(config_.sessionTimezone, false);
      VELOX_USER_CHECK_NOT_NULL(
          timezone, "Invalid sessionTimezone '{}'.", config_.sessionTimezone);
      sessionTimezone_ = timezone;
    } else {
      sessionTimezone_ = tz::locateZone("UTC", false);
    }
  }

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args, // Not using const ref so we can reuse args
      const TypePtr& outputType,
      exec::EvalCtx& context,
      VectorPtr& result) const final {
    VELOX_USER_CHECK(
        args[0]->isConstantEncoding() || args[0]->isFlatEncoding(),
        "Single-arg deterministic functions receive their only argument as flat or constant vector.");
    FromJsonParseContext parseContext;
    parseContext.allowNonNumericNumbers = config_.allowNonNumericNumbers;
    parseContext.dateFormatter = dateFormatter_.get();
    parseContext.timestampFormatter = timestampFormatter_.get();
    parseContext.hasDateFormat = config_.dateFormat.has_value();
    parseContext.hasTimestampFormat = config_.timestampFormat.has_value();
    parseContext.sparkLegacyDateFormatter = config_.sparkLegacyDateFormatter;
    parseContext.enablePartialResults = config_.enablePartialResults;
    parseContext.caseSensitiveFieldMatch = config_.caseSensitiveFieldMatch;
    parseContext.sessionTimezone = sessionTimezone_;
    context.ensureWritable(rows, outputType, result);
    result->clearNulls(rows);
    if (args[0]->isConstantEncoding()) {
      parseJsonConstant(args[0], context, rows, *result, parseContext);
    } else {
      parseJsonFlat(args[0], context, rows, *result, parseContext);
    }
  }

 private:
  void parseJsonConstant(
      VectorPtr& input,
      exec::EvalCtx& context,
      const SelectivityVector& rows,
      BaseVector& result,
      FromJsonParseContext& parseContext) const {
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
      writer.finish();
    } else {
      const auto constant = constInput->valueAt(0);
      // Guard against empty SelectivityVector (no rows selected).
      if (!rows.hasSelections()) {
        writer.finish();
        return;
      }
      paddedInput_.resize(constant.size() + simdjson::SIMDJSON_PADDING);
      memcpy(paddedInput_.data(), constant.data(), constant.size());
      simdjson::padded_string_view paddedInput(
          paddedInput_.data(), constant.size(), paddedInput_.size());

      // Parse constant JSON once for the first selected row. User errors are
      // recorded for every selected row through EvalCtx, as on the flat-input
      // path, so TRY can suppress them.
      vector_size_t firstRow = rows.begin();
      writer.setOffset(firstRow);
      parseContext.hadPartialFailure = false;

      std::exception_ptr rowError;
      try {
        simdjson::ondemand::document jsonDoc;
        simdjson::error_code error;
        try {
          error = simdjsonParse(paddedInput).get(jsonDoc);
        } catch (const simdjson::simdjson_error& e) {
          error = e.error();
        }

        bool parseError = error != simdjson::SUCCESS;
        if (!parseError) {
          try {
            parseError =
                extractJsonToWriter(
                    jsonDoc, writer, rowSchemaInfoMap_, parseContext) !=
                simdjson::SUCCESS;
          } catch (const simdjson::simdjson_error&) {
            parseError = true;
          } catch (const std::exception&) {
            // Finalize partially written state before reporting the error.
            writer.commitNull();
            throw;
          }
        }
        if (parseError) {
          writer.commitNull();
          if (!isBlank(constant)) {
            if (config_.mode == ParseMode::kFailFast) {
              failfast(constant);
            }
            if (config_.corruptRecordIndex.has_value()) {
              writeCorruptRecord(result, firstRow, constant);
            } else {
              permissivelyClearRow(result, firstRow);
            }
          }
        } else if (parseContext.hadPartialFailure) {
          if (config_.mode == ParseMode::kFailFast) {
            failfast(constant);
          }
          if (config_.corruptRecordIndex.has_value()) {
            writeCorruptRecordPartial(result, firstRow, constant);
          }
        }
      } catch (const VeloxException& e) {
        if (!e.isUserError()) {
          throw;
        }
        rowError = std::current_exception();
      } catch (const std::exception&) {
        rowError = std::current_exception();
      }

      // Finalize the writer BEFORE copying so nested ARRAY/MAP child
      // buffers are fully materialized and won't be truncated.
      writer.finish();

      // Copy the first row's result to all other selected rows.
      rows.applyToSelected([&](auto row) {
        if (row != firstRow) {
          result.copy(&result, row, firstRow, 1);
        }
      });

      if (rowError) {
        context.setErrors(rows, rowError);
      }
    }
  }

  void parseJsonFlat(
      VectorPtr& input,
      exec::EvalCtx& context,
      const SelectivityVector& rows,
      BaseVector& result,
      FromJsonParseContext& parseContext) const {
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
      simdjson::error_code error;
      try {
        error = simdjsonParse(paddedInput).get(doc);
      } catch (const simdjson::simdjson_error& e) {
        error = e.error();
      }
      parseContext.hadPartialFailure = false;
      bool parseError = error != simdjson::SUCCESS;
      if (!parseError) {
        try {
          parseError = extractJsonToWriter(
                           doc, writer, rowSchemaInfoMap_, parseContext) !=
              simdjson::SUCCESS;
        } catch (const simdjson::simdjson_error&) {
          // Lazy parse failures follow the configured malformed-record policy.
          parseError = true;
        } catch (const std::exception&) {
          // Finalize partially written state before applyToSelectedNoThrow
          // records the error, so later rows cannot reuse it.
          writer.commitNull();
          throw;
        }
      }
      if (parseError) {
        writer.commitNull();
        if (isBlank(input)) {
          // Empty and JSON-whitespace-only input remains a top-level null.
          return;
        }
        if (config_.mode == ParseMode::kFailFast) {
          failfast(input);
        }
        if (config_.corruptRecordIndex.has_value()) {
          writeCorruptRecord(result, row, input);
        } else {
          // Malformed ROW input is a non-null row with null fields.
          permissivelyClearRow(result, row);
        }
      } else if (parseContext.hadPartialFailure) {
        if (config_.mode == ParseMode::kFailFast) {
          failfast(input);
        }
        if (config_.corruptRecordIndex.has_value()) {
          writeCorruptRecordPartial(result, row, input);
        }
      }
    });
    writer.finish();
  }

  // Throws the FAILFAST user error with Spark's error-class label and the
  // offending record content.
  static void failfast(StringView record) {
    VELOX_USER_FAIL(
        "[MALFORMED_RECORD_IN_PARSING.WITHOUT_SUGGESTION] "
        "Malformed records are detected in record parsing. "
        "Parse Mode: FAILFAST. Record: {}",
        std::string_view(record.data(), record.size()));
  }

  // Returns true for empty input and JSON whitespace.
  static bool isBlank(StringView input) {
    return std::all_of(input.begin(), input.end(), [](char c) {
      return c == ' ' || c == '\t' || c == '\n' || c == '\r';
    });
  }

  // Recursively marks `row` as null and clears complex-vector state. This is
  // required before un-nulling the parent row to populate a corrupt-record
  // field, because ARRAY/MAP offsets and nested ROW children remain observable.
  static void clearVectorAtRow(BaseVector& vector, vector_size_t row) {
    vector.setNull(row, true);
    if (auto* arrayVector = vector.as<ArrayVector>()) {
      arrayVector->setOffsetAndSize(row, 0, 0);
    } else if (auto* mapVector = vector.as<MapVector>()) {
      mapVector->setOffsetAndSize(row, 0, 0);
    } else if (auto* rowVector = vector.as<RowVector>()) {
      for (column_index_t i = 0; i < rowVector->childrenSize(); ++i) {
        if (auto& child = rowVector->childAt(i)) {
          clearVectorAtRow(*child, row);
        }
      }
    }
  }

  // Writes a non-null ROW with all-null children for malformed input in
  // PERMISSIVE mode. ARRAY and MAP roots remain null.
  void permissivelyClearRow(BaseVector& result, vector_size_t row) const {
    auto* rowVector = result.as<RowVector>();
    if (rowVector == nullptr) {
      return;
    }
    rowVector->setNull(row, false);
    for (column_index_t i = 0; i < rowVector->childrenSize(); ++i) {
      if (auto& child = rowVector->childAt(i)) {
        clearVectorAtRow(*child, row);
      }
    }
  }

  // Writes a non-null ROW containing only the raw corrupt-record value after
  // writer.commitNull() has finalized partially written children.
  void writeCorruptRecord(
      BaseVector& result,
      vector_size_t row,
      StringView rawInput) const {
    result.setNull(row, false);
    auto* rowVector = result.asUnchecked<RowVector>();
    for (column_index_t i = 0; i < rowVector->childrenSize(); ++i) {
      if (i == *config_.corruptRecordIndex) {
        auto* stringChild =
            rowVector->childAt(i)->asUnchecked<FlatVector<StringView>>();
        stringChild->setNull(row, false);
        stringChild->set(row, rawInput);
      } else {
        clearVectorAtRow(*rowVector->childAt(i), row);
      }
    }
  }

  // Populates only the corrupt-record column for a partial result, preserving
  // successfully parsed fields and clearing state behind null children.
  void writeCorruptRecordPartial(
      BaseVector& result,
      vector_size_t row,
      StringView rawInput) const {
    auto* rowVector = result.asUnchecked<RowVector>();
    auto* stringChild = rowVector->childAt(*config_.corruptRecordIndex)
                            ->asUnchecked<FlatVector<StringView>>();
    stringChild->setNull(row, false);
    stringChild->set(row, rawInput);
    for (column_index_t i = 0; i < rowVector->childrenSize(); ++i) {
      if (i == *config_.corruptRecordIndex) {
        continue;
      }
      auto& child = rowVector->childAt(i);
      if (child && child->isNullAt(row)) {
        clearVectorAtRow(*child, row);
      }
    }
  }

  // Constructs schema lookup data keyed by schema-tree node index.
  column_index_t constructRowSchemaInfoMap(
      const TypePtr& type,
      column_index_t& index,
      int depth) {
    VELOX_USER_CHECK(
        depth <= kMaxSchemaDepth,
        "from_json: schema nesting depth {} exceeds maximum {} -- "
        "deeply nested ARRAY/MAP/ROW schemas can stack-overflow the worker.",
        depth,
        kMaxSchemaDepth);
    auto nodeKey = index++;
    switch (type->kind()) {
      case TypeKind::ARRAY: {
        constructRowSchemaInfoMap(type->childAt(0), index, depth + 1);
        break;
      }
      case TypeKind::MAP: {
        constructRowSchemaInfoMap(type->childAt(1), index, depth + 1);
        break;
      }
      case TypeKind::ROW: {
        const auto& rowType = asRowType(type);
        const auto& names = rowType->names();
        bool allFieldsAreAscii =
            std::all_of(names.begin(), names.end(), [](const auto& name) {
              return functions::stringCore::isAscii(name.data(), name.size());
            });
        auto isFieldMissing = std::make_shared<std::vector<bool>>();
        isFieldMissing->resize(rowType->size(), true);
        folly::F14FastMap<std::string, FieldInfo> fieldMap;
        const auto size = rowType->size();
        for (column_index_t i = 0; i < size; ++i) {
          const auto childNodeIndex =
              constructRowSchemaInfoMap(type->childAt(i), index, depth + 1);
          if (nodeKey == 0 && config_.corruptRecordIndex.has_value() &&
              *config_.corruptRecordIndex == i) {
            continue;
          }
          std::string key = rowType->nameOf(i);
          // Lowercase for case-insensitive matching (legacy mode).
          // When caseSensitiveFieldMatch=true, preserve original case.
          if (!config_.caseSensitiveFieldMatch) {
            if (allFieldsAreAscii) {
              folly::toLowerAscii(key);
            } else {
              boost::algorithm::to_lower(key);
            }
          }
          fieldMap[key] = FieldInfo{i, childNodeIndex};
        }
        rowSchemaInfoMap_.insert_or_assign(
            nodeKey,
            JsonRowSchemaInfo(
                nodeKey,
                allFieldsAreAscii,
                std::move(isFieldMissing),
                std::move(fieldMap)));
        break;
      }
      default:
        break;
    }
    return nodeKey;
  }

  // Extracts data from json doc and writes it to writer.
  // @param rowSchemaInfoMap A map from schema tree node index to
  // JsonRowSchemaInfo.
  static simdjson::error_code extractJsonToWriter(
      simdjson::ondemand::document& doc,
      exec::VectorWriter<Any>& writer,
      const folly::F14FastMap<int64_t, JsonRowSchemaInfo>& rowSchemaInfoMap,
      FromJsonParseContext& parseContext) {
    SIMDJSON_ASSIGN_OR_RAISE(auto isNull, doc.is_null());
    if (isNull) {
      // Spark treats a JSON null root as a malformed record. SQL NULL inputs
      // are handled separately before parsing.
      return simdjson::INCORRECT_TYPE;
    }
    SIMDJSON_TRY(
        ExtractJsonTypeImpl<simdjson::ondemand::document&>::apply<kind>(
            doc, writer.current(), true, rowSchemaInfoMap, 0, parseContext));
    writer.commit(true);
    return simdjson::SUCCESS;
  }

  // Reusable padded buffer for simdjson parsing. Mutable because apply() is
  // const but needs scratch space. The buffer is owned by the vector-function
  // instance and follows the same evaluation-lifetime pattern as other
  // simdjson vector functions.
  mutable std::string paddedInput_;
  // Map from row schema tree node index to schema information for JSON rows.
  folly::F14FastMap<int64_t, JsonRowSchemaInfo> rowSchemaInfoMap_;
  // Configuration options for from_json parsing.
  FromJsonConfig config_;
  // Pre-built date formatter for custom dateFormat (null if using default).
  std::shared_ptr<DateTimeFormatter> dateFormatter_;
  // Pre-built timestamp formatter for custom timestampFormat (null if default).
  std::shared_ptr<DateTimeFormatter> timestampFormatter_;
  // Resolved session timezone. Defaults to UTC when no timezone is configured.
  const tz::TimeZone* sessionTimezone_ = nullptr;
};

bool parseBoolOption(const std::string& value) {
  auto normalized = value;
  folly::toLowerAscii(normalized);
  VELOX_USER_CHECK(
      normalized == "true" || normalized == "false",
      "from_json: boolean option must be true or false, got '{}'.",
      value);
  return normalized == "true";
}

/// Returns whether the type is supported and validates schema depth.
bool isSupportedType(const TypePtr& type, bool isRootType, int depth) {
  VELOX_USER_CHECK(
      depth <= kMaxSchemaDepth,
      "from_json: schema nesting depth {} exceeds maximum {} -- "
      "deeply nested ARRAY/MAP/ROW schemas can stack-overflow the worker.",
      depth,
      kMaxSchemaDepth);
  switch (type->kind()) {
    case TypeKind::ARRAY: {
      return isSupportedType(type->childAt(0), false, depth + 1);
    }
    case TypeKind::ROW: {
      for (const auto& child : asRowType(type)->children()) {
        if (!isSupportedType(child, false, depth + 1)) {
          return false;
        }
      }
      return true;
    }
    case TypeKind::MAP: {
      return (
          type->childAt(0)->kind() == TypeKind::VARCHAR &&
          isSupportedType(type->childAt(1), false, depth + 1));
    }
    case TypeKind::HUGEINT:
    case TypeKind::BIGINT:
    case TypeKind::INTEGER:
    case TypeKind::BOOLEAN:
    case TypeKind::SMALLINT:
    case TypeKind::TINYINT:
    case TypeKind::DOUBLE:
    case TypeKind::REAL:
    case TypeKind::VARCHAR:
    case TypeKind::TIMESTAMP: {
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
    const core::QueryConfig& config) {
  VELOX_USER_CHECK(!args.empty(), "from_json expects at least one argument.");
  VELOX_USER_CHECK(
      args[0]->type()->kind() == TypeKind::VARCHAR,
      "The first argument of from_json should be of varchar type.");
  VELOX_USER_CHECK(
      isSupportedType(type, true, 0), "Unsupported type {}.", type->toString());

  FromJsonConfig fromJsonConfig;
  fromJsonConfig.sparkLegacyDateFormatter =
      SparkQueryConfig{config}.legacyDateFormatter();
  fromJsonConfig.sessionTimezone = config.sessionTimezone();
  for (size_t i = 1; i < args.size(); ++i) {
    auto* constantExpression = dynamic_cast<exec::ConstantExpr*>(args[i].get());
    VELOX_USER_CHECK_NOT_NULL(
        constantExpression,
        "from_json: optional argument {} must be a constant string of the "
        "form \"key=value\".",
        i);
    VELOX_USER_CHECK(
        constantExpression->type()->kind() == TypeKind::VARCHAR,
        "from_json: optional argument {} must be VARCHAR, got {}.",
        i,
        constantExpression->type()->toString());
    auto* constantValue =
        constantExpression->value()->as<ConstantVector<StringView>>();
    VELOX_USER_CHECK_NOT_NULL(
        constantValue,
        "from_json: optional argument {} must be a constant VARCHAR.",
        i);
    VELOX_USER_CHECK(
        !constantValue->isNullAt(0),
        "from_json: optional argument {} must not be null.",
        i);

    auto option = constantValue->valueAt(0).str();
    const auto separatorPosition = option.find('=');
    VELOX_USER_CHECK(
        separatorPosition != std::string::npos,
        "from_json: optional argument {} must be of the form "
        "\"key=value\", got '{}'.",
        i,
        option);
    auto key = option.substr(0, separatorPosition);
    auto value = option.substr(separatorPosition + 1);
    folly::toLowerAscii(key);
    if (key == "allownonnumericnumbers") {
      fromJsonConfig.allowNonNumericNumbers = parseBoolOption(value);
    } else if (key == "mode") {
      auto mode = value;
      folly::toUpperAscii(mode);
      VELOX_USER_CHECK(
          mode != "DROPMALFORMED",
          "[PARSE_MODE_UNSUPPORTED] The function `from_json` doesn't support "
          "the {} mode. Acceptable modes are PERMISSIVE and FAILFAST.",
          value);
      if (mode == "FAILFAST") {
        fromJsonConfig.mode = ParseMode::kFailFast;
      } else {
        if (mode != "PERMISSIVE") {
          LOG(WARNING) << "from_json: unrecognized parse mode '" << value
                       << "'; using PERMISSIVE.";
        }
        fromJsonConfig.mode = ParseMode::kPermissive;
      }
    } else if (key == "columnnameofcorruptrecord") {
      fromJsonConfig.columnNameOfCorruptRecord = value;
    } else if (key == "dateformat") {
      fromJsonConfig.dateFormat = value;
    } else if (key == "timestampformat") {
      fromJsonConfig.timestampFormat = value;
    } else if (key == "enablepartialresults") {
      fromJsonConfig.enablePartialResults = parseBoolOption(value);
    } else if (key == "sessiontimezone" || key == "timezone") {
      VELOX_USER_CHECK(!value.empty(), "Invalid sessionTimezone '{}'.", value);
      fromJsonConfig.sessionTimezone = value;
    } else if (key == "casesensitivefieldmatch") {
      fromJsonConfig.caseSensitiveFieldMatch = parseBoolOption(value);
    } else if (
        key == "allowsinglequotes" || key == "allowcomments" ||
        key == "allowunquotedfieldnames" ||
        key == "allowbackslashescapinganycharacter" ||
        key == "allowunquotedcontrolchars" ||
        key == "allownumericleadingzeros") {
      parseBoolOption(value);
      LOG(WARNING) << "from_json: option '" << key
                   << "' is recognized by Spark but not enforceable in "
                      "Velox/simdjson (simdjson is RFC-8259 strict). "
                      "Ignoring; per-row parse outcome is determined by "
                      "simdjson defaults and may differ from Spark "
                      "(value='"
                   << value << "').";
    } else if (
        key == "multiline" || key == "prefersdecimal" ||
        key == "dropfieldifallnull" || key == "ignorenullfields") {
      parseBoolOption(value);
      LOG(WARNING) << "from_json: option '" << key
                   << "' is recognized by Spark but not implemented in "
                      "Velox (no-op for from_json on a string input or "
                      "applies only to schema inference / source readers). "
                      "Ignoring (value='"
                   << value << "').";
    } else if (
        key == "samplingratio" || key == "linesep" || key == "encoding" ||
        key == "charset" || key == "locale") {
      LOG(WARNING) << "from_json: option '" << key
                   << "' is recognized by Spark but not implemented in "
                      "Velox (no-op for from_json on a string input or "
                      "applies only to schema inference / source readers). "
                      "Ignoring (value='"
                   << value << "').";
    } else {
      VELOX_USER_FAIL(
          "from_json: unrecognized option '{}'. Supported options: "
          "allowNonNumericNumbers, mode, "
          "columnNameOfCorruptRecord, timestampFormat, dateFormat, "
          "enablePartialResults, sessionTimezone (alias: timeZone), "
          "caseSensitiveFieldMatch. "
          "Spark allow* flags (allowSingleQuotes, allowComments, "
          "allowUnquotedFieldNames, allowBackslashEscapingAnyCharacter, "
          "allowUnquotedControlChars, allowNumericLeadingZeros) and "
          "inert Spark JSONOptions keys (multiLine, prefersDecimal, "
          "dropFieldIfAllNull, samplingRatio, lineSep, encoding, charset, "
          "ignoreNullFields, locale) are accepted-but-ignored with a "
          "warning.",
          option.substr(0, separatorPosition));
    }
  }

  // Use the corrupt-record column when it is present in a ROW schema.
  if (fromJsonConfig.columnNameOfCorruptRecord.has_value() &&
      type->kind() == TypeKind::ROW) {
    const auto& rowType = asRowType(type);
    for (column_index_t i = 0; i < rowType->size(); ++i) {
      if (rowType->nameOf(i) == *fromJsonConfig.columnNameOfCorruptRecord) {
        VELOX_USER_CHECK(
            rowType->childAt(i)->kind() == TypeKind::VARCHAR,
            "The corrupt record column '{}' must be of VARCHAR type, got {}.",
            *fromJsonConfig.columnNameOfCorruptRecord,
            rowType->childAt(i)->toString());
        fromJsonConfig.corruptRecordIndex = i;
        break;
      }
    }
  }

  std::vector<exec::ExprPtr> inputs;
  inputs.push_back(std::move(args[0]));

  std::shared_ptr<exec::VectorFunction> vectorFunction;
  if (type->kind() == TypeKind::ARRAY) {
    vectorFunction = std::make_shared<FromJsonFunction<TypeKind::ARRAY>>(
        type, std::move(fromJsonConfig));
  } else if (type->kind() == TypeKind::MAP) {
    vectorFunction = std::make_shared<FromJsonFunction<TypeKind::MAP>>(
        type, std::move(fromJsonConfig));
  } else {
    vectorFunction = std::make_shared<FromJsonFunction<TypeKind::ROW>>(
        type, std::move(fromJsonConfig));
  }

  return std::make_shared<exec::Expr>(
      type,
      std::move(inputs),
      vectorFunction,
      exec::VectorFunctionMetadata{},
      kFromJson,
      trackCpuUsage);
}
} // namespace facebook::velox::functions::sparksql
