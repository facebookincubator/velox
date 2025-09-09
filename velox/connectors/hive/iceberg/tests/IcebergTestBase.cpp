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

template <typename T>
std::string IcebergTestBase::getListAsCSVString(
    const std::vector<T>& deleteValues) {
  if (deleteValues.empty()) {
    return "";
  }

  if constexpr (std::is_same_v<T, StringView>) {
    return std::accumulate(
        deleteValues.begin() + 1,
        deleteValues.end(),
        fmt::format("'{}'", deleteValues[0].str()),
        [](const std::string& a, const StringView& b) {
          return a + fmt::format(", '{}'", b.str());
        });
  } else {
    return std::accumulate(
        deleteValues.begin() + 1,
        deleteValues.end(),
        std::to_string(deleteValues[0]),
        [](const std::string& a, const T& b) {
          return a + ", " + std::to_string(b);
        });
  }
}

template <typename T>
std::string IcebergTestBase::makeSingleColumnNotInPredicate(
    const VectorPtr& deleteVector,
    int32_t numDeletedRows,
    const std::string& columnName) {
  auto vector = deleteVector->as<FlatVector<T>>();
  std::vector<T> deleteValues;
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
      getListAsCSVString(deleteValues));
}

template <typename T>
std::string IcebergTestBase::makeSingleValueInequalityPredicate(
    const VectorPtr& deleteVector,
    int32_t row,
    const std::string& columnName) {
  auto vector = deleteVector->as<FlatVector<T>>();
  if constexpr (std::is_same_v<T, StringView>) {
    std::string valueStr = vector->valueAt(row).str();
    return fmt::format("({} <> '{}')", columnName, valueStr);
  } else {
    std::string valueStr = std::to_string(vector->valueAt(row));
    return fmt::format("({} <> {})", columnName, valueStr);
  }
}

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

  std::map<TypeKind, int> typeColumnIndex;

  std::vector<RowVectorPtr> rowVectors;
  for (int i = 0; i < count; i++) {
    std::vector<VectorPtr> vectors;

    typeColumnIndex.clear();

    for (size_t j = 0; j < columnTypes.size(); ++j) {
      VectorPtr columnVector;
      TypeKind columnType = columnTypes[j];
      NullParam nullParam = nullParams[j];

      typeColumnIndex[columnType]++;
      int repeatCount = typeColumnIndex[columnType];

      if (nullParam == NullParam::kAllNulls) {
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

      if (nullParam == NullParam::kPartialNulls) {
        std::mt19937 gen(42);
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

  if (!config.dataVectors.empty()) {
    for (int i = 0; i < config.dataVectors.size(); i++) {
      std::string fileName = fmt::format("data_file_{}", i);
      dataFilePaths[fileName] = TempFilePath::create();
      writeToFile(dataFilePaths[fileName]->getPath(), config.dataVectors[i]);
    }
    createDuckDbTable(config.dataVectors);
    return dataFilePaths;
  }

  std::map<std::string, std::vector<int64_t>> fileStructure;
  if (config.rowGroupSizesForFiles.has_value()) {
    fileStructure = config.rowGroupSizesForFiles.value();
  } else {
    for (int i = 0; i < config.splitCount; i++) {
      std::string fileName = fmt::format("data_file_{}", i);
      fileStructure[fileName] = {static_cast<int64_t>(config.numRows)};
    }
  }

  dataVectorsJoined.reserve(fileStructure.size());
  int64_t startingValue = 0;

  for (const auto& dataFile : fileStructure) {
    dataFilePaths[dataFile.first] = TempFilePath::create();
    std::vector<RowVectorPtr> dataVectors;
    dataVectors.reserve(dataFile.second.size());

    for (int64_t size : dataFile.second) {
      std::vector<TypeKind> columnTypes(config.numColumns, TypeKind::BIGINT);
      std::vector<NullParam> nullParams(config.numColumns, NullParam::kNoNulls);
      auto rowVectors = makeVectors(1, size, columnTypes, nullParams);

      auto vector = rowVectors[0];
      auto firstColumn = vector->childAt(0)->as<FlatVector<int64_t>>();
      for (int64_t i = 0; i < size; ++i) {
        firstColumn->set(i, startingValue + i);
      }

      dataVectors.push_back(vector);
      startingValue += size;
    }

    if (config.useConfigAndFlushPolicy) {
      writeToFile(
          dataFilePaths[dataFile.first]->getPath(),
          dataVectors,
          config_,
          flushPolicyFactory_);
    } else {
      if (dataVectors.size() == 1) {
        writeToFile(dataFilePaths[dataFile.first]->getPath(), dataVectors[0]);
      } else {
        writeToFile(dataFilePaths[dataFile.first]->getPath(), dataVectors);
      }
    }

    dataVectorsJoined.insert(
        dataVectorsJoined.end(), dataVectors.begin(), dataVectors.end());
  }

  createDuckDbTable(dataVectorsJoined);
  return dataFilePaths;
}

std::string IcebergTestBase::makePredicates(
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
    auto deleteVector = deleteRowVector->childAt(0);
    auto fieldId = equalityFieldIds[0];
    auto columnType = columnTypes[0];
    std::string columnName = fmt::format("c{}", fieldId - 1);

    switch (columnType) {
      case TypeKind::TINYINT:
        return makeSingleColumnNotInPredicate<int8_t>(
            deleteVector, numDeletedRows, columnName);
      case TypeKind::SMALLINT:
        return makeSingleColumnNotInPredicate<int16_t>(
            deleteVector, numDeletedRows, columnName);
      case TypeKind::INTEGER:
        return makeSingleColumnNotInPredicate<int32_t>(
            deleteVector, numDeletedRows, columnName);
      case TypeKind::BIGINT:
        return makeSingleColumnNotInPredicate<int64_t>(
            deleteVector, numDeletedRows, columnName);
      case TypeKind::REAL:
        return makeSingleColumnNotInPredicate<float>(
            deleteVector, numDeletedRows, columnName);
      case TypeKind::DOUBLE:
        return makeSingleColumnNotInPredicate<double>(
            deleteVector, numDeletedRows, columnName);
      case TypeKind::VARCHAR:
        return makeSingleColumnNotInPredicate<StringView>(
            deleteVector, numDeletedRows, columnName);
      case TypeKind::VARBINARY:
        return makeSingleColumnNotInPredicate<StringView>(
            deleteVector, numDeletedRows, columnName);
      default:
        VELOX_FAIL(
            "Unsupported type for predicate: {}",
            mapTypeKindToName(columnType));
    }
  } else {
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
          switch (columnType) {
            case TypeKind::TINYINT:
              predicate = makeSingleValueInequalityPredicate<int8_t>(
                  deleteVector, row, columnName);
              break;
            case TypeKind::SMALLINT:
              predicate = makeSingleValueInequalityPredicate<int16_t>(
                  deleteVector, row, columnName);
              break;
            case TypeKind::INTEGER:
              predicate = makeSingleValueInequalityPredicate<int32_t>(
                  deleteVector, row, columnName);
              break;
            case TypeKind::BIGINT:
              predicate = makeSingleValueInequalityPredicate<int64_t>(
                  deleteVector, row, columnName);
              break;
            case TypeKind::VARCHAR:
            case TypeKind::VARBINARY:
              predicate = makeSingleValueInequalityPredicate<StringView>(
                  deleteVector, row, columnName);
              break;
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

// Explicit template instantiations for getListAsCSVString
template std::string IcebergTestBase::getListAsCSVString<int8_t>(
    const std::vector<int8_t>&);
template std::string IcebergTestBase::getListAsCSVString<int16_t>(
    const std::vector<int16_t>&);
template std::string IcebergTestBase::getListAsCSVString<int32_t>(
    const std::vector<int32_t>&);
template std::string IcebergTestBase::getListAsCSVString<int64_t>(
    const std::vector<int64_t>&);
template std::string IcebergTestBase::getListAsCSVString<float>(
    const std::vector<float>&);
template std::string IcebergTestBase::getListAsCSVString<double>(
    const std::vector<double>&);
template std::string IcebergTestBase::getListAsCSVString<StringView>(
    const std::vector<StringView>&);

// Explicit template instantiations for makeSingleColumnNotInPredicate
template std::string IcebergTestBase::makeSingleColumnNotInPredicate<int8_t>(
    const VectorPtr&,
    int32_t,
    const std::string&);
template std::string IcebergTestBase::makeSingleColumnNotInPredicate<int16_t>(
    const VectorPtr&,
    int32_t,
    const std::string&);
template std::string IcebergTestBase::makeSingleColumnNotInPredicate<int32_t>(
    const VectorPtr&,
    int32_t,
    const std::string&);
template std::string IcebergTestBase::makeSingleColumnNotInPredicate<int64_t>(
    const VectorPtr&,
    int32_t,
    const std::string&);
template std::string IcebergTestBase::makeSingleColumnNotInPredicate<float>(
    const VectorPtr&,
    int32_t,
    const std::string&);
template std::string IcebergTestBase::makeSingleColumnNotInPredicate<double>(
    const VectorPtr&,
    int32_t,
    const std::string&);
template std::string IcebergTestBase::makeSingleColumnNotInPredicate<
    StringView>(const VectorPtr&, int32_t, const std::string&);

// Explicit template instantiations for makeSingleValueInequalityPredicate
template std::string IcebergTestBase::makeSingleValueInequalityPredicate<
    int8_t>(const VectorPtr&, int32_t, const std::string&);
template std::string IcebergTestBase::makeSingleValueInequalityPredicate<
    int16_t>(const VectorPtr&, int32_t, const std::string&);
template std::string IcebergTestBase::makeSingleValueInequalityPredicate<
    int32_t>(const VectorPtr&, int32_t, const std::string&);
template std::string IcebergTestBase::makeSingleValueInequalityPredicate<
    int64_t>(const VectorPtr&, int32_t, const std::string&);
template std::string IcebergTestBase::makeSingleValueInequalityPredicate<
    StringView>(const VectorPtr&, int32_t, const std::string&);
} // namespace facebook::velox::connector::hive::iceberg
