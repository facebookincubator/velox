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

#include <utility>

#include "velox/connectors/clp/search_lib/ClpVectorLoader.h"
#include "velox/vector/ComplexVector.h"

namespace facebook::velox::connector::clp::search_lib {
ClpVectorLoader::ClpVectorLoader(
    clp_s::BaseColumnReader* columnReader,
    ColumnType nodeType,
    std::shared_ptr<std::vector<uint64_t>> filteredRowIndices)
    : columnReader_(columnReader),
      nodeType_(nodeType),
      filteredRowIndices_(std::move(filteredRowIndices)) {}

template <typename T, typename VectorPtr>
void ClpVectorLoader::populateData(RowSet rows, VectorPtr vector) {
  if (columnReader_ == nullptr) {
    for (int vectorIndex : rows) {
      vector->setNull(vectorIndex, true);
    }
    return;
  }

  for (int vectorIndex : rows) {
    auto messageIndex = (*filteredRowIndices_)[vectorIndex];

    if constexpr (std::is_same_v<T, std::string>) {
      auto string_value =
          std::get<std::string>(columnReader_->extract_value(messageIndex));
      vector->set(vectorIndex, StringView(string_value));
    } else {
      vector->set(
          vectorIndex, std::get<T>(columnReader_->extract_value(messageIndex)));
    }

    vector->setNull(vectorIndex, false);
  }
}

void ClpVectorLoader::loadInternal(
    RowSet rows,
    ValueHook* hook,
    vector_size_t resultSize,
    VectorPtr* result) {
  if (!result) {
    VELOX_USER_FAIL("vector is null");
  }

  auto vector = *result;
  switch (nodeType_) {
    case ColumnType::Integer: {
      auto intVector = vector->asFlatVector<int64_t>();
      populateData<int64_t>(rows, intVector);
      break;
    }
    case ColumnType::Float: {
      auto floatVector = vector->asFlatVector<double>();
      populateData<double>(rows, floatVector);
      break;
    }
    case ColumnType::Boolean: {
      auto boolVector = vector->asFlatVector<bool>();
      populateData<uint8_t>(rows, boolVector);
      break;
    }
    case ColumnType::String: {
      auto stringVector = vector->asFlatVector<StringView>();
      populateData<std::string>(rows, stringVector);
      break;
    }
    case ColumnType::Array: {
      auto arrayVector = std::dynamic_pointer_cast<ArrayVector>(vector);
      auto elements = arrayVector->elements()->asFlatVector<StringView>();
      vector_size_t elementIndex = 0;

      for (int vectorIndex : rows) {
        auto messageIndex = (*filteredRowIndices_)[vectorIndex];

        auto jsonString =
            std::get<std::string>(columnReader_->extract_value(messageIndex));

        simdjson::padded_string padded(jsonString);
        simdjson::ondemand::document doc;
        try {
          doc = arrayParser_->iterate(padded);
        } catch (const simdjson::simdjson_error& e) {
          VELOX_FAIL("JSON parse error at row {}: {}", vectorIndex, e.what());
        }

        simdjson::ondemand::array array;
        try {
          array = doc.get_array();
        } catch (const simdjson::simdjson_error& e) {
          VELOX_FAIL(
              "Expected JSON array at row {}: {}", vectorIndex, e.what());
        }

        std::vector<std::string_view> arrayElements;
        for (auto arrayElement : array) {
          arrayElements.emplace_back(
              simdjson::to_json_string(arrayElement).value());
        }

        if (elementIndex + arrayElements.size() > elements->size()) {
          size_t newSize = std::max<size_t>(
              elementIndex + arrayElements.size(),
              static_cast<size_t>(elements->size()) * 2);
          elements->resize(newSize);
        }

        arrayVector->setOffsetAndSize(
            vectorIndex, elementIndex, arrayElements.size());
        for (auto& arrayElement : arrayElements) {
          elements->set(elementIndex++, StringView(arrayElement));
        }

        arrayVector->setNull(vectorIndex, false);
      }
      break;
    }
    default:
      VELOX_FAIL("Unsupported column type");
  }
}

// Explicit template instantiations for linker
template void ClpVectorLoader::populateData<int64_t>(
    RowSet rows,
    FlatVector<int64_t>* vector);
template void ClpVectorLoader::populateData<double>(
    RowSet rows,
    FlatVector<double>* vector);
template void ClpVectorLoader::populateData<uint8_t>(
    RowSet rows,
    FlatVector<bool>* vector);
template void ClpVectorLoader::populateData<std::string>(
    RowSet rows,
    FlatVector<StringView>* vector);
} // namespace facebook::velox::connector::clp::search_lib
