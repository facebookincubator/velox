/*
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
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

#include "velox/functions/prestosql/aggregates/VectorIndexBuildAggregate.h"

#include <cmath>
#include <limits>

#include "velox/common/vectorindex/LocalVectorIndex.h"
#include "velox/exec/Aggregate.h"
#include "velox/exec/SimpleAggregateAdapter.h"
#include "velox/expression/FunctionSignature.h"
#include "velox/vector/DecodedVector.h"

namespace facebook::velox::aggregate::prestosql {
namespace {

using common::vectorindex::EmbeddingNormalizer;
using common::vectorindex::Embeddings;
using common::vectorindex::LocalVectorIndex;
using common::vectorindex::LocalVectorIndexOptions;
using common::vectorindex::VectorIndexMetric;
using common::vectorindex::VectorIndexMetricName;
using common::vectorindex::VectorIndexMetricValue;
using common::vectorindex::VectorIndexType;
using common::vectorindex::VectorIndexTypeName;
using common::vectorindex::VectorIndexTypeValue;

// Reads an option at 'position' in an intermediate settings array. Positions
// past the end of the array are unset.
std::optional<int32_t> readOption(
    const exec::arg_type<Array<int32_t>>& array,
    int32_t position) {
  if (position >= array.size() || !array[position].has_value()) {
    return std::nullopt;
  }
  return array[position].value();
}

// The settings from the constant arguments, stored in intermediate results as
// an array of integers. Unset options are null. New settings take new
// positions at the end, so an array from before they existed reads them as
// unset.
struct Settings {
  // Positions in the array.
  static constexpr int32_t kIndexType = 0;
  static constexpr int32_t kMetric = 1;
  static constexpr int32_t kNlist = 2;
  static constexpr int32_t kNprobe = 3;
  static constexpr int32_t kM = 4;
  static constexpr int32_t kEfConstruction = 5;
  static constexpr int32_t kEfSearch = 6;
  static constexpr int32_t kSize = 7;
  // The size of the array before any option.
  static constexpr int32_t kMinSize = 2;

  VectorIndexType indexType;
  VectorIndexMetric metric;
  LocalVectorIndexOptions options;

  bool operator==(const Settings&) const = default;

  // Reads settings from an intermediate result, which may come from
  // vector_index_build_merge, so the values are checked as the constant
  // arguments are.
  static Settings read(const exec::arg_type<Array<int32_t>>& array) {
    VELOX_USER_CHECK_GE(
        array.size(),
        kMinSize,
        "Too few entries in intermediate vector index settings");
    VELOX_USER_CHECK_LE(
        array.size(),
        kSize,
        "Too many entries in intermediate vector index settings");
    VELOX_USER_CHECK(
        array[kIndexType].has_value(),
        "Intermediate vector index settings must include the index type");
    VELOX_USER_CHECK(
        array[kMetric].has_value(),
        "Intermediate vector index settings must include the metric");
    Settings settings{
        .indexType = VectorIndexTypeValue::fromValue(array[kIndexType].value()),
        .metric = VectorIndexMetricValue::fromValue(array[kMetric].value()),
        .options =
            {
                .nlist = readOption(array, kNlist),
                .nprobe = readOption(array, kNprobe),
                .m = readOption(array, kM),
                .efConstruction = readOption(array, kEfConstruction),
                .efSearch = readOption(array, kEfSearch),
            },
    };
    settings.options.validate(settings.indexType);
    return settings;
  }

  void write(exec::out_type<Array<int32_t>>& array) const {
    array.add_item() = VectorIndexTypeValue::toValue(indexType);
    array.add_item() = VectorIndexMetricValue::toValue(metric);
    for (const auto& option :
         {options.nlist,
          options.nprobe,
          options.m,
          options.efConstruction,
          options.efSearch}) {
      if (option.has_value()) {
        array.add_item() = *option;
      } else {
        array.add_null();
      }
    }
  }
};

class VectorIndexBuildAggregate {
 public:
  using InputType = Row<
      /*embedding*/ Array<float>,
      /*id*/ int64_t,
      /*index_type*/ Varchar,
      /*metric*/ Varchar,
      /*options*/ Map<Varchar, Varchar>>;

  // Carries the settings from the constant arguments, which steps that merge
  // intermediate results do not receive.
  using IntermediateType = Row<
      /*settings*/ Array<int32_t>,
      /*ids*/ Array<int64_t>,
      /*embeddings, one after another*/ Array<float>>;

  // Positions of the fields of 'IntermediateType'.
  static constexpr int32_t kSettings = 0;
  static constexpr int32_t kIds = 1;
  static constexpr int32_t kEmbeddings = 2;

  using OutputType = Varbinary;

  static constexpr bool is_reducing_ = false;

  // Reads and checks the constant arguments once, before any rows arrive.
  void setConstantInputs(const std::vector<VectorPtr>& constantInputs) {
    VELOX_CHECK_EQ(constantInputs.size(), 5);
    const auto& indexType = constantInputs[2];
    const auto& metric = constantInputs[3];
    const auto& options = constantInputs[4];
    VELOX_USER_CHECK_NOT_NULL(indexType, "Index type must be a constant");
    VELOX_USER_CHECK_NOT_NULL(metric, "Metric must be a constant");
    VELOX_USER_CHECK_NOT_NULL(options, "Options must be a constant");
    VELOX_USER_CHECK(!indexType->isNullAt(0), "Index type must not be null");
    VELOX_USER_CHECK(!metric->isNullAt(0), "Metric must not be null");
    VELOX_USER_CHECK(!options->isNullAt(0), "Options must not be null");

    const auto indexTypeName =
        indexType->as<ConstantVector<StringView>>()->valueAt(0);
    const auto parsedIndexType =
        VectorIndexTypeName::tryToVectorIndexType(indexTypeName);
    VELOX_USER_CHECK(
        parsedIndexType.has_value(),
        "Unsupported vector index type: {}",
        indexTypeName);
    const auto metricName =
        metric->as<ConstantVector<StringView>>()->valueAt(0);
    const auto parsedMetric =
        VectorIndexMetricName::tryToVectorIndexMetric(metricName);
    VELOX_USER_CHECK(
        parsedMetric.has_value(),
        "Unsupported vector index metric: {}",
        metricName);

    DecodedVector decodedOptions(*options);
    const auto* map = decodedOptions.base()->asChecked<MapVector>();
    const auto row = decodedOptions.index(0);
    std::vector<std::pair<std::string_view, std::string_view>> optionPairs;
    if (map->sizeAt(row) > 0) {
      const auto* keys = map->mapKeys()->asChecked<SimpleVector<StringView>>();
      const auto* values =
          map->mapValues()->asChecked<SimpleVector<StringView>>();
      for (auto i = map->offsetAt(row);
           i < map->offsetAt(row) + map->sizeAt(row);
           ++i) {
        VELOX_USER_CHECK(
            !values->isNullAt(i), "Option values must not be null");
        optionPairs.emplace_back(keys->valueAt(i), values->valueAt(i));
      }
    }

    constantSettings_ = Settings{
        .indexType = *parsedIndexType,
        .metric = *parsedMetric,
        .options =
            LocalVectorIndexOptions::parse(*parsedIndexType, optionPairs),
    };
  }

  struct AccumulatorType {
    // The length of every embedding; 0 until the first one arrives.
    int32_t dimension{0};
    std::vector<int64_t, StlAllocator<int64_t>> ids;
    // The embeddings, normalized for the metric, one after another, in the
    // order 'ids' lists them.
    std::vector<float, StlAllocator<float>> embeddings;
    // The group's settings: from the constant arguments for raw input, and
    // from the intermediate results otherwise.
    std::optional<Settings> settings;
    // The function object shared by all groups, which holds the settings from
    // the constant arguments.
    VectorIndexBuildAggregate* fn;

    static constexpr bool is_fixed_size_ = false;
    static constexpr bool use_external_memory_ = true;

    AccumulatorType() = delete;

    explicit AccumulatorType(
        HashStringAllocator* allocator,
        VectorIndexBuildAggregate* fn)
        : ids{StlAllocator<int64_t>(allocator)},
          embeddings{StlAllocator<float>(allocator)},
          fn{fn} {}

    void addInput(
        HashStringAllocator* /*allocator*/,
        exec::arg_type<Array<float>> embedding,
        int64_t id,
        exec::arg_type<Varchar> /*indexType*/,
        exec::arg_type<Varchar> /*metric*/,
        exec::arg_type<Map<Varchar, Varchar>> /*options*/) {
      if (!settings.has_value()) {
        settings = fn->constantSettings();
      }
      setDimension(embedding.size());
      for (auto value : embedding) {
        VELOX_USER_CHECK(value.has_value(), "Embedding must not contain nulls");
        addValue(value.value());
      }
      EmbeddingNormalizer::normalize(
          groupSettings().metric, std::span(embeddings).last(dimension));
      ids.push_back(id);
    }

    void combine(
        HashStringAllocator* /*allocator*/,
        exec::arg_type<IntermediateType> other) {
      VELOX_USER_CHECK(
          other.at<kSettings>().has_value(),
          "Intermediate vector index result must include the settings");
      VELOX_USER_CHECK(
          other.at<kIds>().has_value(),
          "Intermediate vector index result must include the ids");
      VELOX_USER_CHECK(
          other.at<kEmbeddings>().has_value(),
          "Intermediate vector index result must include the embeddings");
      mergeSettings(Settings::read(*other.at<kSettings>()));

      const auto& otherIds = *other.at<kIds>();
      const auto& otherEmbeddings = *other.at<kEmbeddings>();
      VELOX_USER_CHECK_GT(
          otherIds.size(),
          0,
          "Intermediate vector index result must include an embedding");
      VELOX_USER_CHECK_EQ(
          otherEmbeddings.size() % otherIds.size(),
          0,
          "Intermediate vector index result must have the same number of values per embedding");
      setDimension(otherEmbeddings.size() / otherIds.size());
      for (auto id : otherIds) {
        VELOX_USER_CHECK(
            id.has_value(), "Intermediate vector index ids must not be null");
        ids.push_back(id.value());
      }
      for (auto value : otherEmbeddings) {
        VELOX_USER_CHECK(
            value.has_value(),
            "Intermediate vector index embeddings must not contain nulls");
        addValue(value.value());
      }
    }

    void writeIntermediateResult(exec::out_type<IntermediateType>& out) {
      VELOX_USER_CHECK_LE(
          embeddings.size(),
          std::numeric_limits<vector_size_t>::max(),
          "Too many embedding values in a vector index group");
      groupSettings().write(out.get_writer_at<kSettings>());
      auto& idsWriter = out.get_writer_at<kIds>();
      for (auto id : ids) {
        idsWriter.add_item() = id;
      }
      auto& embeddingsWriter = out.get_writer_at<kEmbeddings>();
      for (auto value : embeddings) {
        embeddingsWriter.add_item() = value;
      }
    }

    void writeFinalResult(exec::out_type<OutputType>& out) {
      const auto& finalSettings = groupSettings();
      out.reserve(embeddings.size() * sizeof(float));
      LocalVectorIndex::build(
          finalSettings.indexType,
          finalSettings.metric,
          finalSettings.options,
          Embeddings(embeddings, dimension),
          ids,
          [&](std::string_view chunk) { out.append(chunk); });
    }

   private:
    const Settings& groupSettings() const {
      VELOX_CHECK(settings.has_value());
      return *settings;
    }

    // Takes the settings of the first intermediate result and checks that
    // later ones match.
    void mergeSettings(const Settings& other) {
      if (!settings.has_value()) {
        settings = other;
        return;
      }
      VELOX_USER_CHECK(
          other == *settings,
          "Intermediate vector index results of a group must have the same settings");
    }

    void setDimension(int64_t size) {
      VELOX_USER_CHECK_GT(size, 0, "Embedding must not be empty");
      if (dimension == 0) {
        dimension = size;
        return;
      }
      VELOX_USER_CHECK_EQ(
          size, dimension, "All embeddings must have the same length");
    }

    void addValue(float value) {
      VELOX_USER_CHECK(
          std::isfinite(value), "Embedding must not contain NaN or infinity");
      embeddings.push_back(value);
    }
  };

 private:
  const Settings& constantSettings() const {
    VELOX_CHECK(constantSettings_.has_value());
    return *constantSettings_;
  }

  // Set by setConstantInputs() in steps that read raw input.
  std::optional<Settings> constantSettings_;
};

} // namespace

void registerVectorIndexBuildAggregate(
    const std::vector<std::string>& names,
    bool withCompanionFunctions,
    bool overwrite) {
  std::vector<std::shared_ptr<exec::AggregateFunctionSignature>> signatures{
      exec::AggregateFunctionSignatureBuilder()
          .returnType("varbinary")
          .intermediateType("row(array(integer),array(bigint),array(real))")
          .argumentType("array(real)")
          .argumentType("bigint")
          .constantArgumentType("varchar")
          .constantArgumentType("varchar")
          .constantArgumentType("map(varchar,varchar)")
          .build(),
  };

  exec::registerAggregateFunction(
      names,
      signatures,
      [](core::AggregationNode::Step step,
         const std::vector<TypePtr>& argTypes,
         const TypePtr& resultType,
         const core::QueryConfig& /*config*/)
          -> std::unique_ptr<exec::Aggregate> {
        return std::make_unique<
            exec::SimpleAggregateAdapter<VectorIndexBuildAggregate>>(
            step, argTypes, resultType);
      },
      {.ignoreNullInputs = true},
      withCompanionFunctions,
      overwrite);
}

} // namespace facebook::velox::aggregate::prestosql
