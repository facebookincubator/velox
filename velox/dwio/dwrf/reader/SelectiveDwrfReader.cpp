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

#include "velox/dwio/dwrf/reader/SelectiveDwrfReader.h"
#include "velox/dwio/common/TypeUtils.h"

#include "velox/dwio/dwrf/reader/SelectiveByteRleColumnReader.h"
#include "velox/dwio/dwrf/reader/SelectiveDecimalColumnReader.h"
#include "velox/dwio/dwrf/reader/SelectiveFlatMapColumnReader.h"
#include "velox/dwio/dwrf/reader/SelectiveFloatingPointColumnReader.h"
#include "velox/dwio/dwrf/reader/SelectiveIntegerDictionaryColumnReader.h"
#include "velox/dwio/dwrf/reader/SelectiveIntegerDirectColumnReader.h"
#include "velox/dwio/dwrf/reader/SelectiveRepeatedColumnReader.h"
#include "velox/dwio/dwrf/reader/SelectiveStringDictionaryColumnReader.h"
#include "velox/dwio/dwrf/reader/SelectiveStringDirectColumnReader.h"
#include "velox/dwio/dwrf/reader/SelectiveStructColumnReader.h"
#include "velox/dwio/dwrf/reader/SelectiveTimestampColumnReader.h"

namespace facebook::velox::dwrf {

using namespace facebook::velox::dwio::common;

void checkType(
    const TypePtr& fileType,
    const TypePtr& requestedType,
    const std::function<bool(const TypePtr&)>& isCompatible) {
  if (!isCompatible(requestedType)) {
    VELOX_SCHEMA_MISMATCH_ERROR(
        fmt::format(
            "Schema mismatch, From Kind: {}, To Kind: {}",
            TypeKindName::toName(fileType->kind()),
            TypeKindName::toName(requestedType->kind())));
  }
}

bool isIntegerCompatible(
    TypeKind fileType,
    const TypePtr& requestedType,
    const bool projectOnly) {
  if (requestedType->isVarchar()) {
    // Integer values are converted to VARCHAR in getValues(). Filters and
    // value hooks run before this conversion and operate on integer values.
    VELOX_USER_CHECK(
        projectOnly,
        "INTEGER to VARCHAR schema evolution only supports projection");
    return true;
  }

  return !requestedType->isDecimal() &&
      dwio::common::typeutils::isCompatible(fileType, requestedType->kind());
}

std::unique_ptr<SelectiveColumnReader> buildIntegerReader(
    const TypePtr& requestedType,
    const std::shared_ptr<const dwio::common::TypeWithId>& fileType,
    DwrfParams& params,
    common::ScanSpec& scanSpec) {
  const EncodingKey encodingKey{
      fileType->id(), params.flatMapContext().sequence};
  auto& stripe = params.stripeStreams();

  const auto numBytes = sizeOfIntKind(fileType->type()->kind());
  if (StripeStreamsUtil::isColumnEncodingKindDictionary(stripe, encodingKey)) {
    return std::make_unique<SelectiveIntegerDictionaryColumnReader>(
        requestedType, fileType, params, scanSpec, numBytes);
  } else if (StripeStreamsUtil::isColumnEncodingKindDirect(
                 stripe, encodingKey)) {
    return std::make_unique<SelectiveIntegerDirectColumnReader>(
        requestedType, fileType, params, numBytes, scanSpec);
  } else {
    const auto encodingKind = stripe.format() == DwrfFormat::kDwrf
        ? static_cast<int64_t>(stripe.getEncoding(encodingKey).kind())
        : static_cast<int64_t>(stripe.getEncodingOrc(encodingKey).kind());
    VELOX_FAIL("buildReader unhandled integer encoding: {}", encodingKind);
  }
}

// static
std::unique_ptr<SelectiveColumnReader> SelectiveDwrfReader::build(
    const dwio::common::ColumnReaderOptions& columnReaderOptions,
    const TypePtr& requestedType,
    const std::shared_ptr<const dwio::common::TypeWithId>& fileType,
    DwrfParams& params,
    common::ScanSpec& scanSpec,
    bool isRoot) {
  const auto fileTypeKind = fileType->type()->kind();
  VELOX_CHECK(
      !isRoot || fileTypeKind == TypeKind::ROW,
      "The root object can only be a row.");

  EncodingKey ek{fileType->id(), params.flatMapContext().sequence};
  auto& stripe = params.stripeStreams();
  const bool projectOnly = !scanSpec.filter() && !scanSpec.valueHook();
  switch (fileTypeKind) {
    case TypeKind::BOOLEAN:
    case TypeKind::TINYINT:
      checkType(fileType->type(), requestedType, [=](const TypePtr& type) {
        return isIntegerCompatible(fileTypeKind, requestedType, projectOnly);
      });
      return std::make_unique<SelectiveByteRleColumnReader>(
          requestedType,
          fileType,
          params,
          scanSpec,
          fileType->type()->isBoolean());
    case TypeKind::BIGINT:
      if (fileType->type()->isDecimal()) {
        checkType(fileType->type(), requestedType, [](const TypePtr& type) {
          return type->isShortDecimal();
        });
        return std::make_unique<SelectiveDecimalColumnReader<int64_t>>(
            requestedType, fileType, params, scanSpec);
      }
    case TypeKind::SMALLINT:
    case TypeKind::INTEGER:
      checkType(fileType->type(), requestedType, [=](const TypePtr& type) {
        return isIntegerCompatible(fileTypeKind, requestedType, projectOnly);
      });
      return buildIntegerReader(requestedType, fileType, params, scanSpec);
    case TypeKind::ARRAY:
      checkType(fileType->type(), requestedType, [](const TypePtr& type) {
        return type->isArray();
      });
      return std::make_unique<SelectiveListColumnReader>(
          columnReaderOptions, requestedType, fileType, params, scanSpec);
    case TypeKind::MAP:
      checkType(fileType->type(), requestedType, [](const TypePtr& type) {
        return type->isMap();
      });
      if (stripe.format() == DwrfFormat::kDwrf &&
          stripe.getEncoding(ek).kind() ==
              proto::ColumnEncoding_Kind_MAP_FLAT) {
        return createSelectiveFlatMapColumnReader(
            columnReaderOptions, requestedType, fileType, params, scanSpec);
      }
      if (scanSpec.isFlatMapAsStruct()) {
        return std::make_unique<SelectiveMapAsStructColumnReader>(
            columnReaderOptions, requestedType, fileType, params, scanSpec);
      }
      return std::make_unique<SelectiveMapColumnReader>(
          columnReaderOptions, requestedType, fileType, params, scanSpec);
    case TypeKind::REAL:
      checkType(fileType->type(), requestedType, [](const TypePtr& type) {
        return type->isReal() || type->isDouble();
      });
      if (requestedType->isReal()) {
        return std::make_unique<
            SelectiveFloatingPointColumnReader<float, float>>(
            requestedType, fileType, params, scanSpec);
      } else {
        return std::make_unique<
            SelectiveFloatingPointColumnReader<float, double>>(
            requestedType, fileType, params, scanSpec);
      }
    case TypeKind::DOUBLE:
      checkType(fileType->type(), requestedType, [](const TypePtr& type) {
        return type->isDouble();
      });
      return std::make_unique<
          SelectiveFloatingPointColumnReader<double, double>>(
          requestedType, fileType, params, scanSpec);
    case TypeKind::ROW:
      checkType(fileType->type(), requestedType, [](const TypePtr& type) {
        return type->isRow();
      });
      return std::make_unique<SelectiveStructColumnReader>(
          columnReaderOptions,
          requestedType,
          fileType,
          params,
          scanSpec,
          isRoot);
    case TypeKind::VARBINARY:
    case TypeKind::VARCHAR: {
      checkType(fileType->type(), requestedType, [&](const TypePtr& type) {
        return type->kind() == fileTypeKind;
      });
      if (StripeStreamsUtil::isColumnEncodingKindDictionary(stripe, ek)) {
        return std::make_unique<SelectiveStringDictionaryColumnReader>(
            fileType, params, scanSpec);
      } else if (StripeStreamsUtil::isColumnEncodingKindDirect(stripe, ek)) {
        return std::make_unique<SelectiveStringDirectColumnReader>(
            fileType, params, scanSpec);
      } else {
        DWIO_RAISE("buildReader string unknown encoding");
      }
    }
    case TypeKind::TIMESTAMP:
      checkType(fileType->type(), requestedType, [](const TypePtr& type) {
        return type->isTimestamp();
      });
      return std::make_unique<SelectiveTimestampColumnReader>(
          fileType, params, scanSpec);
    case TypeKind::HUGEINT:
      if (fileType->type()->isDecimal()) {
        checkType(fileType->type(), requestedType, [](const TypePtr& type) {
          return type->isLongDecimal();
        });
        return std::make_unique<SelectiveDecimalColumnReader<int128_t>>(
            requestedType, fileType, params, scanSpec);
      }
      [[fallthrough]];
    default:
      VELOX_FAIL(
          "buildReader unhandled type: " +
          std::string(TypeKindName::toName(fileType->type()->kind())));
  }
}

} // namespace facebook::velox::dwrf
