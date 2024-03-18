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
#include <limits>
#include "velox/exec/Aggregate.h"
#include "velox/exec/Strings.h"
#include "velox/expression/FunctionSignature.h"
#include "velox/functions/lib/CheckedArithmeticImpl.h"
#include "velox/functions/prestosql/aggregates/AggregateNames.h"
#include "velox/vector/FlatVector.h"

namespace facebook::velox::aggregate::prestosql {

namespace {

template <typename T>
struct MinMaxTrait : public std::numeric_limits<T> {};

template <typename K, typename S>
struct Accumulator {
  folly::F14FastMap<
      K,
      S,
      std::hash<K>,
      std::equal_to<K>,
      AlignedStlAllocator<std::pair<const K, S>, 16>>
      accumulates;

  explicit Accumulator(HashStringAllocator* allocator)
      : accumulates{AlignedStlAllocator<std::pair<const K, S>, 16>(allocator)} {
  }

  size_t size() const {
    return accumulates.size();
  }

  void addValues(
      const MapVector* mapVector,
      const SimpleVector<K>* mapKeys,
      const SimpleVector<S>* mapValues,
      vector_size_t row,
      HashStringAllocator* allocator,
      std::function<S(S, S)> compareFunction) {
    auto offset = mapVector->offsetAt(row);
    auto size = mapVector->sizeAt(row);

    for (auto i = 0; i < size; ++i) {
      // Ignore null map keys.
      if (!mapKeys->isNullAt(offset + i)) {
        auto key = mapKeys->valueAt(offset + i);
        addValue(
            key, mapValues, offset + i, mapValues->typeKind(), compareFunction);
      }
    }
  }

  void addValue(
      K key,
      const SimpleVector<S>* mapValues,
      vector_size_t row,
      TypeKind valueKind,
      std::function<S(S, S)> compareFunction) {
    if (mapValues->isNullAt(row)) {
      return;
    } else {
      auto value = mapValues->valueAt(row);
      if constexpr (std::is_same_v<S, double> || std::is_same_v<S, float>) {
        // Skip NaN
        if (std::isnan(value)) {
          return;
        }
      }
      auto it = accumulates.find(key);
      if (it != accumulates.end()) {
        accumulates[key] = compareFunction(it->second, value);
      } else {
        accumulates.emplace(key, value);
      }
    }
  }

  vector_size_t extractValues(
      FlatVector<K>& mapKeys,
      FlatVector<S>& mapValues,
      vector_size_t offset) {
    auto index = offset;
    for (const auto& [key, sum] : accumulates) {
      mapKeys.set(index, key);
      mapValues.set(index, sum);

      ++index;
    }

    return accumulates.size();
  }
};

template <typename S>
struct StringViewAccumulator {
  Accumulator<StringView, S> base;

  Strings strings;

  explicit StringViewAccumulator(HashStringAllocator* allocator)
      : base{allocator} {}

  size_t size() const {
    return base.size();
  }

  void addValues(
      const MapVector* mapVector,
      const SimpleVector<StringView>* mapKeys,
      const SimpleVector<S>* mapValues,
      vector_size_t row,
      HashStringAllocator* allocator,
      std::function<S(S, S)> compareFunction) {
    auto offset = mapVector->offsetAt(row);
    auto size = mapVector->sizeAt(row);

    for (auto i = 0; i < size; ++i) {
      // Ignore null map keys.
      if (!mapKeys->isNullAt(offset + i)) {
        auto key = mapKeys->valueAt(offset + i);

        if (!key.isInline()) {
          auto it = base.accumulates.find(key);
          if (it != base.accumulates.end()) {
            key = it->first;
          } else {
            key = strings.append(key, *allocator);
          }
        }

        base.addValue(
            key, mapValues, offset + i, mapValues->typeKind(), compareFunction);
      }
    }
  }

  vector_size_t extractValues(
      FlatVector<StringView>& mapKeys,
      FlatVector<S>& mapValues,
      vector_size_t offset) {
    return base.extractValues(mapKeys, mapValues, offset);
  }
};

template <typename K, typename S>
struct AccumulatorTypeTraits {
  using AccumulatorType = Accumulator<K, S>;
};

template <typename S>
struct AccumulatorTypeTraits<StringView, S> {
  using AccumulatorType = StringViewAccumulator<S>;
};

template <typename K, typename S>
class MapUnionMinMaxAggregate : public exec::Aggregate {
 public:
  explicit MapUnionMinMaxAggregate(TypePtr resultType)
      : Aggregate(std::move(resultType)) {}

  using AccumulatorType = typename AccumulatorTypeTraits<K, S>::AccumulatorType;

  int32_t accumulatorFixedWidthSize() const override {
    return sizeof(AccumulatorType);
  }

  bool isFixedSize() const override {
    return false;
  }

  void initializeNewGroups(
      char** groups,
      folly::Range<const vector_size_t*> indices) override {
    setAllNulls(groups, indices);
    for (auto index : indices) {
      new (groups[index] + offset_) AccumulatorType{allocator_};
    }
  }

  void extractValues(char** groups, int32_t numGroups, VectorPtr* result)
      override {
    auto mapVector = (*result)->as<MapVector>();
    VELOX_CHECK(mapVector);
    mapVector->resize(numGroups);

    auto mapKeys = mapVector->mapKeys()->as<FlatVector<K>>();
    auto mapValues = mapVector->mapValues()->as<FlatVector<S>>();

    auto numElements = countElements(groups, numGroups);
    mapKeys->resize(numElements);
    mapValues->resize(numElements);

    auto rawNulls = mapVector->mutableRawNulls();
    vector_size_t offset = 0;
    for (auto i = 0; i < numGroups; ++i) {
      char* group = groups[i];
      if (isNull(group)) {
        bits::setNull(rawNulls, i, true);
        mapVector->setOffsetAndSize(i, 0, 0);
      } else {
        clearNull(rawNulls, i);

        auto mapSize = value<AccumulatorType>(group)->extractValues(
            *mapKeys, *mapValues, offset);
        mapVector->setOffsetAndSize(i, offset, mapSize);
        offset += mapSize;
      }
    }
  }

  void extractAccumulators(char** groups, int32_t numGroups, VectorPtr* result)
      override {
    extractValues(groups, numGroups, result);
  }

  void addRawInput(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    decodedMaps_.decode(*args[0], rows);
    auto mapVector = decodedMaps_.base()->template as<MapVector>();
    auto mapKeys = mapVector->mapKeys()->template as<SimpleVector<K>>();
    auto mapValues = mapVector->mapValues()->template as<SimpleVector<S>>();

    rows.applyToSelected([&](auto row) {
      if (!decodedMaps_.isNullAt(row)) {
        clearNull(groups[row]);
        auto groupMap = value<AccumulatorType>(groups[row]);
        addMap(*groupMap, mapVector, mapKeys, mapValues, row);
      }
    });
  }

  void addSingleGroupRawInput(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /* mayPushdown */) override {
    decodedMaps_.decode(*args[0], rows);
    auto mapVector = decodedMaps_.base()->template as<MapVector>();
    auto mapKeys = mapVector->mapKeys()->template as<SimpleVector<K>>();
    auto mapValues = mapVector->mapValues()->template as<SimpleVector<S>>();

    auto groupMap = value<AccumulatorType>(group);

    rows.applyToSelected([&](auto row) {
      if (!decodedMaps_.isNullAt(row)) {
        clearNull(group);
        addMap(*groupMap, mapVector, mapKeys, mapValues, row);
      }
    });
  }

  void addIntermediateResults(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    addRawInput(groups, rows, args, false);
  }

  void addSingleGroupIntermediateResults(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /* mayPushdown */) override {
    addSingleGroupRawInput(group, rows, args, false);
  }

  void destroy(folly::Range<char**> groups) override {
    if constexpr (std::is_same_v<K, StringView>) {
      for (auto* group : groups) {
        if (!isNull(group)) {
          value<AccumulatorType>(group)->strings.free(*allocator_);
        }
      }
    }
    destroyAccumulators<AccumulatorType>(groups);
  }

 protected:
  DecodedVector decodedMaps_;
  virtual void addMap(
      AccumulatorType& groupMap,
      const MapVector* mapVector,
      const SimpleVector<K>* mapKeys,
      const SimpleVector<S>* mapValues,
      vector_size_t row) const = 0;

 private:
  vector_size_t countElements(char** groups, int32_t numGroups) const {
    vector_size_t size = 0;
    for (int32_t i = 0; i < numGroups; ++i) {
      size += value<AccumulatorType>(groups[i])->size();
    }
    return size;
  }
};

template <typename K, typename S>
class MapUnionMinAggregate : public MapUnionMinMaxAggregate<K, S> {
 public:
  explicit MapUnionMinAggregate(TypePtr resultType)
      : MapUnionMinMaxAggregate<K, S>(resultType) {}

 protected:
  using AccumulatorType = typename AccumulatorTypeTraits<K, S>::AccumulatorType;
  void addMap(
      AccumulatorType& groupMap,
      const MapVector* mapVector,
      const SimpleVector<K>* mapKeys,
      const SimpleVector<S>* mapValues,
      vector_size_t row) const override {
    auto decodedRow = MapUnionMinMaxAggregate<K, S>::decodedMaps_.index(row);
    groupMap.addValues(
        mapVector,
        mapKeys,
        mapValues,
        decodedRow,
        MapUnionMinMaxAggregate<K, S>::allocator_,
        [](S result, S value) -> S { return result < value ? result : value; });
  }
};

template <typename K, typename S>
class MapUnionMaxAggregate : public MapUnionMinMaxAggregate<K, S> {
 public:
  explicit MapUnionMaxAggregate(TypePtr resultType)
      : MapUnionMinMaxAggregate<K, S>(resultType) {}

 protected:
  using AccumulatorType = typename AccumulatorTypeTraits<K, S>::AccumulatorType;
  void addMap(
      AccumulatorType& groupMap,
      const MapVector* mapVector,
      const SimpleVector<K>* mapKeys,
      const SimpleVector<S>* mapValues,
      vector_size_t row) const override {
    auto decodedRow = MapUnionMinMaxAggregate<K, S>::decodedMaps_.index(row);
    groupMap.addValues(
        mapVector,
        mapKeys,
        mapValues,
        decodedRow,
        MapUnionMinMaxAggregate<K, S>::allocator_,
        [](S result, S value) -> S { return result > value ? result : value; });
  }
};

template <template <typename T, typename Y> class TAggregate, typename K>
std::unique_ptr<exec::Aggregate> createMapUnionMinMaxAggregate(
    TypeKind valueKind,
    const TypePtr& resultType) {
  switch (valueKind) {
    case TypeKind::TINYINT:
      return std::make_unique<TAggregate<K, int8_t>>(resultType);
    case TypeKind::SMALLINT:
      return std::make_unique<TAggregate<K, int16_t>>(resultType);
    case TypeKind::INTEGER:
      return std::make_unique<TAggregate<K, int32_t>>(resultType);
    case TypeKind::BIGINT:
      return std::make_unique<TAggregate<K, int64_t>>(resultType);
    case TypeKind::REAL:
      return std::make_unique<TAggregate<K, float>>(resultType);
    case TypeKind::DOUBLE:
      return std::make_unique<TAggregate<K, double>>(resultType);
    default:
      VELOX_UNREACHABLE();
  }
}

template <template <typename T, typename Y> class TAggregate>
exec::AggregateRegistrationResult registerMapUnionMinMax(
    const std::string& name,
    bool withCompanionFunctions,
    bool overwrite) {
  const std::vector<std::string> keyTypes = {
      "tinyint",
      "smallint",
      "integer",
      "bigint",
      "real",
      "double",
      "varchar",
  };
  const std::vector<std::string> valueTypes = {
      "tinyint",
      "smallint",
      "integer",
      "bigint",
      "double",
      "real",
  };

  std::vector<std::shared_ptr<exec::AggregateFunctionSignature>> signatures;
  for (auto keyType : keyTypes) {
    for (auto valueType : valueTypes) {
      auto mapType = fmt::format("map({},{})", keyType, valueType);
      signatures.push_back(exec::AggregateFunctionSignatureBuilder()
                               .returnType(mapType)
                               .intermediateType(mapType)
                               .argumentType(mapType)
                               .build());
    }
  }

  return exec::registerAggregateFunction(
      name,
      std::move(signatures),
      [name](
          core::AggregationNode::Step /*step*/,
          const std::vector<TypePtr>& argTypes,
          const TypePtr& resultType,
          const core::QueryConfig& /*config*/)
          -> std::unique_ptr<exec::Aggregate> {
        VELOX_CHECK_EQ(argTypes.size(), 1);
        VELOX_CHECK(argTypes[0]->isMap());
        auto& mapType = argTypes[0]->asMap();
        auto keyTypeKind = mapType.keyType()->kind();
        auto valueTypeKind = mapType.valueType()->kind();
        switch (keyTypeKind) {
          case TypeKind::TINYINT:
            return createMapUnionMinMaxAggregate<TAggregate, int8_t>(
                valueTypeKind, resultType);
          case TypeKind::SMALLINT:
            return createMapUnionMinMaxAggregate<TAggregate, int16_t>(
                valueTypeKind, resultType);
          case TypeKind::INTEGER:
            return createMapUnionMinMaxAggregate<TAggregate, int32_t>(
                valueTypeKind, resultType);
          case TypeKind::BIGINT:
            return createMapUnionMinMaxAggregate<TAggregate, int64_t>(
                valueTypeKind, resultType);
          case TypeKind::REAL:
            return createMapUnionMinMaxAggregate<TAggregate, float>(
                valueTypeKind, resultType);
          case TypeKind::DOUBLE:
            return createMapUnionMinMaxAggregate<TAggregate, double>(
                valueTypeKind, resultType);
          case TypeKind::VARCHAR:
            return createMapUnionMinMaxAggregate<TAggregate, StringView>(
                valueTypeKind, resultType);
          default:
            VELOX_UNREACHABLE();
        }
      },
      withCompanionFunctions,
      overwrite);
}

} // namespace

void registerMapUnionMinMaxAggregate(
    const std::string& prefix,
    bool withCompanionFunctions,
    bool overwrite) {
  registerMapUnionMinMax<MapUnionMinAggregate>(
      prefix + kMapUnionMin, withCompanionFunctions, overwrite);
  registerMapUnionMinMax<MapUnionMaxAggregate>(
      prefix + kMapUnionMax, withCompanionFunctions, overwrite);
}

} // namespace facebook::velox::aggregate::prestosql
