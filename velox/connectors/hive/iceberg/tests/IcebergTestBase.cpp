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

#include "velox/connectors/hive/iceberg/tests/IcebergTestBase.h"

#include <map>
#include <random>
#include "connectors/hive/iceberg/IcebergSplit.h"
#include "exec/tests/utils/PlanBuilder.h"

namespace facebook::velox::connector::hive::iceberg {
namespace {
template <TypeKind KIND>
std::string makeNotInList(
    const std::vector<typename TypeTraits<KIND>::NativeType>& deleteValues) {
  using T = TypeTraits<KIND>::NativeType;
  if (deleteValues.empty()) {
    return "";
  }

  if constexpr (KIND == TypeKind::BOOLEAN) {
    VELOX_FAIL("Unsupported Type : {}", TypeTraits<KIND>::name);
  } else if constexpr (
      KIND == TypeKind::VARCHAR || KIND == TypeKind::VARBINARY) {
    return std::accumulate(
        deleteValues.begin() + 1,
        deleteValues.end(),
        fmt::format("'{}'", to<std::string>(deleteValues[0])),
        [](const std::string& a, const T& b) {
          return a + fmt::format(", '{}'", to<std::string>(b));
        });
  } else if (std::is_integral_v<T> || std::is_floating_point_v<T>) {
    return std::accumulate(
        deleteValues.begin() + 1,
        deleteValues.end(),
        to<std::string>(deleteValues[0]),
        [](const std::string& a, const T& b) {
          return a + ", " + to<std::string>(b);
        });
  } else {
    VELOX_FAIL("Unsupported Type : {}", TypeTraits<KIND>::name);
  }
}
} // namespace
std::vector<int64_t> IcebergTestBase::makeRandomDeleteValues(
    int32_t maxRowNumber) {
  std::mt19937 gen{0};
  std::vector<int64_t> deleteRows;
  for (int i = 0; i < maxRowNumber; i++) {
    if (folly::Random::rand32(0, 10, gen) > 8) {
      deleteRows.push_back(i);
    }
  }
  return deleteRows;
}

template <typename T>
std::vector<T> IcebergTestBase::makeSequenceValues(
    int32_t numRows,
    int8_t repeat) {
  static_assert(std::is_integral_v<T>, "T must be an integral type");
  VELOX_CHECK_GT(repeat, 0);

  auto maxValue = std::ceil(static_cast<double>(numRows) / repeat);
  std::vector<T> values;
  values.reserve(numRows);
  for (int32_t i = 0; i < maxValue; i++) {
    for (int8_t j = 0; j < repeat; j++) {
      values.push_back(static_cast<T>(i));
    }
  }
  values.resize(numRows);
  return values;
}

std::vector<std::shared_ptr<ConnectorSplit>> IcebergTestBase::makeIcebergSplits(
    const std::string& dataFilePath,
    const std::vector<IcebergDeleteFile>& deleteFiles,
    const std::unordered_map<std::string, std::optional<std::string>>&
        partitionKeys,
    const uint32_t splitCount) {
  std::unordered_map<std::string, std::string> customSplitInfo;
  customSplitInfo["table_format"] = "hive-iceberg";

  auto file = filesystems::getFileSystem(dataFilePath, nullptr)
                  ->openFileForRead(dataFilePath);
  const int64_t fileSize = file->size();
  std::vector<std::shared_ptr<ConnectorSplit>> splits;
  const uint64_t splitSize = std::floor((fileSize) / splitCount);

  for (int i = 0; i < splitCount; ++i) {
    splits.emplace_back(std::make_shared<HiveIcebergSplit>(
        kHiveConnectorId,
        dataFilePath,
        fileFormat_,
        i * splitSize,
        splitSize,
        partitionKeys,
        std::nullopt,
        customSplitInfo,
        nullptr,
        /*cacheable=*/true,
        deleteFiles));
  }

  return splits;
}

core::PlanNodePtr IcebergTestBase::tableScanNode(
    const RowTypePtr& outputRowType) const {
  return PlanBuilder(pool_.get()).tableScan(outputRowType).planNode();
}

std::vector<RowVectorPtr> IcebergTestBase::makeVectors(
    int32_t count,
    int32_t rowsPerVector,
    const std::vector<TypeKind>& columnTypes,
    const std::vector<NullParam>& nullParams) {
  VELOX_CHECK_EQ(
      columnTypes.size(),
      nullParams.size(),
      "columnTypes and nullParams must have the same size");

  // Sample strings for VARCHAR/VARBINARY data generation
  const std::vector<std::string> sampleStrings = {
      "apple",     "banana",     "cherry",    "date",       "elderberry",
      "fig",       "grape",      "honeydew",  "kiwi",       "lemon",
      "mango",     "nectarine",  "orange",    "papaya",     "quince",
      "raspberry", "strawberry", "tangerine", "watermelon", "zucchini"};

  std::vector<TypePtr> types;
  std::vector<std::string> names;

  for (size_t i = 0; i < columnTypes.size(); ++i) {
    switch (columnTypes[i]) {
      case TypeKind::BOOLEAN:
        types.push_back(createScalarType<TypeKind::BOOLEAN>());
        break;
      case TypeKind::TINYINT:
        types.push_back(createScalarType<TypeKind::TINYINT>());
        break;
      case TypeKind::SMALLINT:
        types.push_back(createScalarType<TypeKind::SMALLINT>());
        break;
      case TypeKind::INTEGER:
        types.push_back(createScalarType<TypeKind::INTEGER>());
        break;
      case TypeKind::BIGINT:
        types.push_back(createScalarType<TypeKind::BIGINT>());
        break;
      case TypeKind::REAL:
        types.push_back(createScalarType<TypeKind::REAL>());
        break;
      case TypeKind::DOUBLE:
        types.push_back(createScalarType<TypeKind::DOUBLE>());
        break;
      case TypeKind::VARCHAR:
        types.push_back(createScalarType<TypeKind::VARCHAR>());
        break;
      case TypeKind::VARBINARY:
        types.push_back(createScalarType<TypeKind::VARBINARY>());
        break;
      case TypeKind::TIMESTAMP:
        types.push_back(createScalarType<TypeKind::TIMESTAMP>());
        break;
      case TypeKind::HUGEINT:
        types.push_back(createScalarType<TypeKind::HUGEINT>());
        break;
      default:
        VELOX_FAIL("Unsupported type: {}", mapTypeKindToName(columnTypes[i]));
    }
    names.push_back(fmt::format("c{}", i));
  }

  // Track the column index for each type to implement the repeat pattern
  // Map from TypeKind to the current column index for that type
  std::map<TypeKind, int> typeColumnIndex;

  std::vector<RowVectorPtr> rowVectors;
  for (int i = 0; i < count; i++) {
    std::vector<VectorPtr> vectors;

    // Reset type column indices for each row vector
    typeColumnIndex.clear();

    for (size_t j = 0; j < columnTypes.size(); ++j) {
      VectorPtr columnVector;
      TypeKind columnType = columnTypes[j];
      NullParam nullParam = nullParams[j];

      // Increment the column index for this type
      typeColumnIndex[columnType]++;
      int repeatCount = typeColumnIndex[columnType];

      if (nullParam == NullParam::kAllNulls) {
        // Create all-null vector based on type
        switch (columnType) {
          case TypeKind::BOOLEAN:
            columnVector = vectorMaker_.allNullFlatVector<bool>(rowsPerVector);
            break;
          case TypeKind::TINYINT:
            columnVector =
                vectorMaker_.allNullFlatVector<int8_t>(rowsPerVector);
            break;
          case TypeKind::SMALLINT:
            columnVector =
                vectorMaker_.allNullFlatVector<int16_t>(rowsPerVector);
            break;
          case TypeKind::INTEGER:
            columnVector =
                vectorMaker_.allNullFlatVector<int32_t>(rowsPerVector);
            break;
          case TypeKind::BIGINT:
            columnVector =
                vectorMaker_.allNullFlatVector<int64_t>(rowsPerVector);
            break;
          case TypeKind::REAL:
            columnVector = vectorMaker_.allNullFlatVector<float>(rowsPerVector);
            break;
          case TypeKind::DOUBLE:
            columnVector =
                vectorMaker_.allNullFlatVector<double>(rowsPerVector);
            break;
          case TypeKind::VARCHAR:
          case TypeKind::VARBINARY:
            columnVector =
                vectorMaker_.allNullFlatVector<StringView>(rowsPerVector);
            break;
          case TypeKind::TIMESTAMP:
            columnVector =
                vectorMaker_.allNullFlatVector<Timestamp>(rowsPerVector);
            break;
          case TypeKind::HUGEINT:
            columnVector =
                vectorMaker_.allNullFlatVector<int128_t>(rowsPerVector);
            break;
          default:
            VELOX_FAIL(
                "Unsupported type for all-null vector: {}",
                mapTypeKindToName(columnType));
        }
      } else {
        // Use repeatCount to implement the pattern where each column of the
        // same type gets incrementally increasing repeat patterns
        switch (columnType) {
          case TypeKind::BOOLEAN: {
            auto intData =
                makeSequenceValues<int64_t>(rowsPerVector, repeatCount);
            std::vector<bool> boolData;
            boolData.reserve(intData.size());
            for (auto val : intData) {
              boolData.push_back(val % 2 == 0);
            }
            columnVector = vectorMaker_.flatVector<bool>(boolData);
            break;
          }
          case TypeKind::TINYINT: {
            auto data = makeSequenceValues<int8_t>(rowsPerVector, repeatCount);
            columnVector = vectorMaker_.flatVector<int8_t>(data);
            break;
          }
          case TypeKind::SMALLINT: {
            auto data = makeSequenceValues<int16_t>(rowsPerVector, repeatCount);
            columnVector = vectorMaker_.flatVector<int16_t>(data);
            break;
          }
          case TypeKind::INTEGER: {
            auto data = makeSequenceValues<int32_t>(rowsPerVector, repeatCount);
            columnVector = vectorMaker_.flatVector<int32_t>(data);
            break;
          }
          case TypeKind::BIGINT: {
            auto data = makeSequenceValues<int64_t>(rowsPerVector, repeatCount);
            columnVector = vectorMaker_.flatVector<int64_t>(data);
            break;
          }
          case TypeKind::REAL: {
            auto intData =
                makeSequenceValues<int64_t>(rowsPerVector, repeatCount);
            std::vector<float> floatData;
            floatData.reserve(intData.size());
            for (auto val : intData) {
              floatData.push_back(static_cast<float>(val) + 0.5f);
            }
            columnVector = vectorMaker_.flatVector<float>(floatData);
            break;
          }
          case TypeKind::DOUBLE: {
            auto intData =
                makeSequenceValues<int64_t>(rowsPerVector, repeatCount);
            std::vector<double> doubleData;
            doubleData.reserve(intData.size());
            for (auto val : intData) {
              doubleData.push_back(static_cast<double>(val) + 0.5);
            }
            columnVector = vectorMaker_.flatVector<double>(doubleData);
            break;
          }
          case TypeKind::VARCHAR: {
            auto intData =
                makeSequenceValues<int64_t>(rowsPerVector, repeatCount);
            auto stringVector = BaseVector::create<FlatVector<StringView>>(
                VARCHAR(), rowsPerVector, pool_.get());

            for (int idx = 0; idx < rowsPerVector; ++idx) {
              auto stringIndex = intData[idx] % sampleStrings.size();
              const std::string& selectedString = sampleStrings[stringIndex];
              stringVector->set(idx, StringView(selectedString));
            }
            columnVector = stringVector;
            break;
          }
          case TypeKind::VARBINARY: {
            auto intData =
                makeSequenceValues<int64_t>(rowsPerVector, repeatCount);
            auto binaryVector = BaseVector::create<FlatVector<StringView>>(
                VARBINARY(), rowsPerVector, pool_.get());

            for (int idx = 0; idx < rowsPerVector; ++idx) {
              auto stringIndex = intData[idx] % sampleStrings.size();
              const std::string& baseString = sampleStrings[stringIndex];

              std::string binaryStr;
              for (char c : baseString) {
                binaryStr += static_cast<unsigned char>(c);
              }
              binaryVector->set(idx, StringView(binaryStr));
            }
            columnVector = binaryVector;
            break;
          }
          case TypeKind::TIMESTAMP: {
            auto intData =
                makeSequenceValues<int64_t>(rowsPerVector, repeatCount);
            std::vector<Timestamp> timestampData;
            timestampData.reserve(intData.size());
            for (auto val : intData) {
              timestampData.push_back(Timestamp(val, 0));
            }
            columnVector = vectorMaker_.flatVector<Timestamp>(timestampData);
            break;
          }
          case TypeKind::HUGEINT: {
            auto data =
                makeSequenceValues<int128_t>(rowsPerVector, repeatCount);
            columnVector = vectorMaker_.flatVector<int128_t>(data);
            break;
          }
          default:
            VELOX_FAIL(
                "Unsupported type for makeVectors: {}",
                mapTypeKindToName(columnType));
        }
      }

      // Apply partial nulls by randomly setting some positions to null
      if (nullParam == NullParam::kPartialNulls) {
        std::mt19937 gen(42); // Fixed seed for reproducibility
        std::uniform_real_distribution<> dis(0.0, 1.0);
        constexpr double nullProbability = 0.2;

        for (vector_size_t idx = 0; idx < rowsPerVector; ++idx) {
          if (dis(gen) < nullProbability) {
            columnVector->setNull(idx, true);
          }
        }
      }

      vectors.push_back(columnVector);
    }

    rowVectors.push_back(makeRowVector(names, vectors));
  }
  return rowVectors;
}

std::map<std::string, std::shared_ptr<TempFilePath>>
IcebergTestBase::writeDataFiles(const WriteDataFilesConfig& config) {
  std::map<std::string, std::shared_ptr<TempFilePath>> dataFilePaths;
  std::vector<RowVectorPtr> dataVectorsJoined;

  // Handle custom dataVectors case first
  if (!config.dataVectors.empty()) {
    for (int i = 0; i < config.dataVectors.size(); i++) {
      std::string fileName = fmt::format("data_file_{}", i);
      dataFilePaths[fileName] = TempFilePath::create();
      writeToFile(dataFilePaths[fileName]->getPath(), config.dataVectors[i]);
    }
    createDuckDbTable(config.dataVectors);
    return dataFilePaths;
  }

  // Determine file structure - either from rowGroupSizesForFiles or generate
  // simple structure
  std::map<std::string, std::vector<int64_t>> fileStructure;
  if (config.rowGroupSizesForFiles.has_value()) {
    fileStructure = config.rowGroupSizesForFiles.value();
  } else {
    // Generate simple uniform structure
    for (int i = 0; i < config.splitCount; i++) {
      std::string fileName = fmt::format("data_file_{}", i);
      fileStructure[fileName] = {static_cast<int64_t>(config.numRows)};
    }
  }

  // Create data files using unified approach
  dataVectorsJoined.reserve(fileStructure.size());
  int64_t startingValue = 0;

  for (const auto& dataFile : fileStructure) {
    dataFilePaths[dataFile.first] = TempFilePath::create();
    std::vector<RowVectorPtr> dataVectors;
    dataVectors.reserve(dataFile.second.size());

    // Use makeVectors to create data instead of manually creating vectors
    for (int64_t size : dataFile.second) {
      // Create vectors with the specified number of columns, all BIGINT
      std::vector<TypeKind> columnTypes(config.numColumns, TypeKind::BIGINT);
      std::vector<NullParam> nullParams(config.numColumns, NullParam::kNoNulls);
      auto rowVectors = makeVectors(1, size, columnTypes, nullParams);

      // Update the values to be continuous from startingValue for first column
      // Other columns follow the makeVectors pattern (with repeat counts)
      auto vector = rowVectors[0];
      auto firstColumn = vector->childAt(0)->as<FlatVector<int64_t>>();
      for (int64_t i = 0; i < size; ++i) {
        firstColumn->set(i, startingValue + i);
      }

      dataVectors.push_back(vector);
      startingValue += size;
    }

    // Choose appropriate writeToFile method based on configuration
    if (config.useConfigAndFlushPolicy) {
      writeToFile(
          dataFilePaths[dataFile.first]->getPath(),
          dataVectors,
          config_,
          flushPolicyFactory_);
    } else {
      // For simple cases, write all vectors for this file
      if (dataVectors.size() == 1) {
        writeToFile(dataFilePaths[dataFile.first]->getPath(), dataVectors[0]);
      } else {
        // Multiple row groups - use vector version but without config
        writeToFile(dataFilePaths[dataFile.first]->getPath(), dataVectors);
      }
    }

    dataVectorsJoined.insert(
        dataVectorsJoined.end(), dataVectors.begin(), dataVectors.end());
  }

  createDuckDbTable(dataVectorsJoined);
  return dataFilePaths;
}

std::string IcebergTestBase::makeTypePredicates(
    const std::vector<RowVectorPtr>& deleteVectors,
    const std::vector<int32_t>& equalityFieldIds,
    const std::vector<TypeKind>& columnTypes) {
  VELOX_CHECK_EQ(deleteVectors.size(), 1);
  VELOX_CHECK_EQ(equalityFieldIds.size(), columnTypes.size());

  if (deleteVectors.empty() || deleteVectors[0]->size() == 0) {
    return "";
  }

  auto deleteRowVector = deleteVectors[0];
  int32_t numDeletedRows = deleteRowVector->size();

  if (equalityFieldIds.size() == 1) {
    // Single column delete - use NOT IN approach like original code
    auto deleteVector = deleteRowVector->childAt(0);
    auto fieldId = equalityFieldIds[0];
    auto columnType = columnTypes[0];
    std::string columnName = fmt::format("c{}", fieldId - 1);

    // Extract delete values based on type
    switch (columnType) {
      case TypeKind::TINYINT: {
        auto vector = deleteVector->as<FlatVector<int8_t>>();
        std::vector<int8_t> deleteValues;
        for (int row = 0; row < numDeletedRows; ++row) {
          if (!vector->isNullAt(row)) {
            deleteValues.push_back(vector->valueAt(row));
          }
        }
        if (deleteValues.empty()) {
          return "";
        }
        return fmt::format(
            "({} IS NULL OR {} NOT IN ({}))",
            columnName,
            columnName,
            makeNotInList<TypeKind::TINYINT>(deleteValues));
      }
      case TypeKind::SMALLINT: {
        auto vector = deleteVector->as<FlatVector<int16_t>>();
        std::vector<int16_t> deleteValues;
        for (int row = 0; row < numDeletedRows; ++row) {
          if (!vector->isNullAt(row)) {
            deleteValues.push_back(vector->valueAt(row));
          }
        }
        if (deleteValues.empty()) {
          return "";
        }
        return fmt::format(
            "({} IS NULL OR {} NOT IN ({}))",
            columnName,
            columnName,
            makeNotInList<TypeKind::SMALLINT>(deleteValues));
      }
      case TypeKind::INTEGER: {
        auto vector = deleteVector->as<FlatVector<int32_t>>();
        std::vector<int32_t> deleteValues;
        for (int row = 0; row < numDeletedRows; ++row) {
          if (!vector->isNullAt(row)) {
            deleteValues.push_back(vector->valueAt(row));
          }
        }
        if (deleteValues.empty()) {
          return "";
        }
        return fmt::format(
            "({} IS NULL OR {} NOT IN ({}))",
            columnName,
            columnName,
            makeNotInList<TypeKind::INTEGER>(deleteValues));
      }
      case TypeKind::BIGINT: {
        auto vector = deleteVector->as<FlatVector<int64_t>>();
        std::vector<int64_t> deleteValues;
        for (int row = 0; row < numDeletedRows; ++row) {
          if (!vector->isNullAt(row)) {
            deleteValues.push_back(vector->valueAt(row));
          }
        }
        if (deleteValues.empty()) {
          return "";
        }
        return fmt::format(
            "({} IS NULL OR {} NOT IN ({}))",
            columnName,
            columnName,
            makeNotInList<TypeKind::BIGINT>(deleteValues));
      }
      case TypeKind::REAL: {
        auto vector = deleteVector->as<FlatVector<float>>();
        std::vector<float> deleteValues;
        for (int row = 0; row < numDeletedRows; ++row) {
          if (!vector->isNullAt(row)) {
            deleteValues.push_back(vector->valueAt(row));
          }
        }
        if (deleteValues.empty()) {
          return "";
        }
        return fmt::format(
            "({} IS NULL OR {} NOT IN ({}))",
            columnName,
            columnName,
            makeNotInList<TypeKind::REAL>(deleteValues));
      }
      case TypeKind::DOUBLE: {
        auto vector = deleteVector->as<FlatVector<double>>();
        std::vector<double> deleteValues;
        for (int row = 0; row < numDeletedRows; ++row) {
          if (!vector->isNullAt(row)) {
            deleteValues.push_back(vector->valueAt(row));
          }
        }
        if (deleteValues.empty()) {
          return "";
        }
        return fmt::format(
            "({} IS NULL OR {} NOT IN ({}))",
            columnName,
            columnName,
            makeNotInList<TypeKind::DOUBLE>(deleteValues));
      }
      case TypeKind::VARCHAR: {
        auto vector = deleteVector->as<FlatVector<StringView>>();
        std::vector<StringView> deleteValues;
        for (int row = 0; row < numDeletedRows; ++row) {
          if (!vector->isNullAt(row)) {
            deleteValues.push_back(vector->valueAt(row));
          }
        }
        if (deleteValues.empty()) {
          return "";
        }
        return fmt::format(
            "({} IS NULL OR {} NOT IN ({}))",
            columnName,
            columnName,
            makeNotInList<TypeKind::VARCHAR>(deleteValues));
      }
      case TypeKind::VARBINARY: {
        auto vector = deleteVector->as<FlatVector<StringView>>();
        std::vector<StringView> deleteValues;
        for (int row = 0; row < numDeletedRows; ++row) {
          if (!vector->isNullAt(row)) {
            deleteValues.push_back(vector->valueAt(row));
          }
        }
        if (deleteValues.empty()) {
          return "";
        }
        return fmt::format(
            "({} IS NULL OR {} NOT IN ({}))",
            columnName,
            columnName,
            makeNotInList<TypeKind::VARBINARY>(deleteValues));
      }
      default:
        VELOX_FAIL(
            "Unsupported type for predicate: {}",
            mapTypeKindToName(columnType));
    }
  } else {
    // Multi-column delete - use row-by-row OR logic like original code
    std::string predicates;
    for (int32_t row = 0; row < numDeletedRows; ++row) {
      std::string oneRow;
      for (size_t i = 0; i < equalityFieldIds.size(); ++i) {
        auto deleteVector = deleteRowVector->childAt(i);
        auto fieldId = equalityFieldIds[i];
        auto columnType = columnTypes[i];
        std::string columnName = fmt::format("c{}", fieldId - 1);

        std::string predicate;
        if (deleteVector->isNullAt(row)) {
          predicate = fmt::format("({} IS NOT NULL)", columnName);
        } else {
          std::string valueStr;
          switch (columnType) {
            case TypeKind::TINYINT: {
              auto vector = deleteVector->as<FlatVector<int8_t>>();
              valueStr = std::to_string(vector->valueAt(row));
              predicate = fmt::format("({} <> {})", columnName, valueStr);
              break;
            }
            case TypeKind::SMALLINT: {
              auto vector = deleteVector->as<FlatVector<int16_t>>();
              valueStr = std::to_string(vector->valueAt(row));
              predicate = fmt::format("({} <> {})", columnName, valueStr);
              break;
            }
            case TypeKind::INTEGER: {
              auto vector = deleteVector->as<FlatVector<int32_t>>();
              valueStr = std::to_string(vector->valueAt(row));
              predicate = fmt::format("({} <> {})", columnName, valueStr);
              break;
            }
            case TypeKind::BIGINT: {
              auto vector = deleteVector->as<FlatVector<int64_t>>();
              valueStr = std::to_string(vector->valueAt(row));
              predicate = fmt::format("({} <> {})", columnName, valueStr);
              break;
            }
            case TypeKind::VARCHAR:
            case TypeKind::VARBINARY: {
              auto vector = deleteVector->as<FlatVector<StringView>>();
              valueStr = vector->valueAt(row).str();
              predicate = fmt::format("({} <> '{}')", columnName, valueStr);
              break;
            }
            default:
              VELOX_FAIL(
                  "Unsupported type for predicate: {}",
                  mapTypeKindToName(columnType));
          }
        }

        oneRow = oneRow.empty() ? predicate
                                : fmt::format("({} OR {})", oneRow, predicate);
      }

      predicates = predicates.empty()
          ? oneRow
          : fmt::format("{} AND {}", predicates, oneRow);
    }
    return predicates;
  }
}

// Explicit template instantiations for makeSequenceValues
template std::vector<bool> IcebergTestBase::makeSequenceValues<bool>(
    int32_t,
    int8_t);
template std::vector<int8_t> IcebergTestBase::makeSequenceValues<int8_t>(
    int32_t,
    int8_t);
template std::vector<int16_t> IcebergTestBase::makeSequenceValues<int16_t>(
    int32_t,
    int8_t);
template std::vector<int32_t> IcebergTestBase::makeSequenceValues<int32_t>(
    int32_t,
    int8_t);
template std::vector<int64_t> IcebergTestBase::makeSequenceValues<int64_t>(
    int32_t,
    int8_t);
template std::vector<int128_t> IcebergTestBase::makeSequenceValues<int128_t>(
    int32_t,
    int8_t);
} // namespace facebook::velox::connector::hive::iceberg
