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
#pragma once

#include "velox/common/hyperloglog/KHyperLogLog.h"
#include "velox/expression/VectorFunction.h"

namespace facebook::velox::functions {

using KHyperLogLog = common::hll::KHyperLogLog<int64_t, memory::MemoryPool>;
// hhhh not sure if this is the correct type. i guess since first step is to
// deserialize..?

class KHyperLogLogCardinalityFunction : public exec::VectorFunction {
 public:
  static std::vector<exec::FunctionSignaturePtr> signatures() {
    return {exec::FunctionSignatureBuilder()
                .returnType("bigint")
                .argumentType("khyperloglog")
                .build()};
  }

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& outputType,
      exec::EvalCtx& context,
      VectorPtr& result) const override {
    VELOX_CHECK_EQ(args.size(), 1);

    context.ensureWritable(rows, outputType, result);
    auto* flatResult = result->as<FlatVector<int64_t>>();

    exec::LocalDecodedVector khllDecoder(context, *args[0], rows);
    auto decodedKhll = khllDecoder.get();

    context.applyToSelectedNoThrow(rows, [&](vector_size_t row) {
      if (decodedKhll->isNullAt(row)) {
        flatResult->setNull(row, true);
        return;
      }

      auto khllData = decodedKhll->valueAt<StringView>(row);
      auto khllInstance = KHyperLogLog::deserialize(
          khllData.data(), khllData.size(), context.pool());
      flatResult->set(row, khllInstance->cardinality());
    });
  }
};

class KHyperLogLogIntersectionCardinalityFunction
    : public exec::VectorFunction {
 public:
  static std::vector<exec::FunctionSignaturePtr> signatures() {
    return {exec::FunctionSignatureBuilder()
                .returnType("bigint")
                .argumentType("khyperloglog")
                .argumentType("khyperloglog")
                .build()};
  }

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& outputType,
      exec::EvalCtx& context,
      VectorPtr& result) const override {
    VELOX_CHECK_EQ(args.size(), 2);

    context.ensureWritable(rows, outputType, result);
    auto* flatResult = result->as<FlatVector<int64_t>>();

    exec::LocalDecodedVector khll1Decoder(context, *args[0], rows);
    auto decodedKhll1 = khll1Decoder.get();

    exec::LocalDecodedVector khll2Decoder(context, *args[1], rows);
    auto decodedKhll2 = khll2Decoder.get();

    context.applyToSelectedNoThrow(rows, [&](vector_size_t row) {
      if (decodedKhll1->isNullAt(row) || decodedKhll2->isNullAt(row)) {
        flatResult->setNull(row, true);
        return;
      }

      auto khll1Data = decodedKhll1->valueAt<StringView>(row);
      auto khll2Data = decodedKhll2->valueAt<StringView>(row);

      auto khll1Instance = KHyperLogLog::deserialize(
          khll1Data.data(), khll1Data.size(), context.pool());
      auto khll2Instance = KHyperLogLog::deserialize(
          khll2Data.data(), khll2Data.size(), context.pool());

      // If both khlls are exact, return the exact intersection cardinality.
      if (khll1Instance->isExact() && khll2Instance->isExact()) {
        flatResult->set(
            row,
            KHyperLogLog::exactIntersectionCardinality(
                *khll1Instance, *khll2Instance));
        return;
      }

      // If either of the khlls are not exact, return an approximation of the
      // intersection cardinality using the Jaccard Index like a similarity
      // index between the 2 key sets.
      int64_t lowestCardinality =
          std::min(khll1Instance->cardinality(), khll2Instance->cardinality());
      double jaccard =
          KHyperLogLog::jaccardIndex(*khll1Instance, *khll2Instance);
      auto setUnion = KHyperLogLog::merge(*khll1Instance, *khll2Instance);
      int64_t computedResult =
          static_cast<int64_t>(std::round(jaccard * setUnion->cardinality()));

      // In a special case where one set is much smaller and almost a true
      // subset of the other, return the size of the smaller set. For example:
      // Set1 = {1,2,3,4,5,6}
      // Set2 = {1,2}
      // Jaccard Index = 1
      // Approximated intersection cardinality = 6
      // This result does not make sense as Set2 does not even have 6
      // elements. Thus return 2 instead.
      flatResult->set(row, std::min(computedResult, lowestCardinality));
    });
  }
};

class KHyperLogLogJaccardIndexFunction : public exec::VectorFunction {
 public:
  static std::vector<exec::FunctionSignaturePtr> signatures() {
    return {exec::FunctionSignatureBuilder()
                .returnType("double")
                .argumentType("khyperloglog")
                .argumentType("khyperloglog")
                .build()};
  }

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& outputType,
      exec::EvalCtx& context,
      VectorPtr& result) const override {
    VELOX_CHECK_EQ(args.size(), 2);

    context.ensureWritable(rows, outputType, result);
    auto* flatResult = result->as<FlatVector<double>>();

    exec::LocalDecodedVector khll1Decoder(context, *args[0], rows);
    auto decodedKhll1 = khll1Decoder.get();

    exec::LocalDecodedVector khll2Decoder(context, *args[1], rows);
    auto decodedKhll2 = khll2Decoder.get();

    context.applyToSelectedNoThrow(rows, [&](vector_size_t row) {
      if (decodedKhll1->isNullAt(row) || decodedKhll2->isNullAt(row)) {
        flatResult->setNull(row, true);
        return;
      }

      auto khll1Data = decodedKhll1->valueAt<StringView>(row);
      auto khll2Data = decodedKhll2->valueAt<StringView>(row);

      auto khll1Instance = KHyperLogLog::deserialize(
          khll1Data.data(), khll1Data.size(), context.pool());
      auto khll2Instance = KHyperLogLog::deserialize(
          khll2Data.data(), khll2Data.size(), context.pool());

      flatResult->set(
          row, KHyperLogLog::jaccardIndex(*khll1Instance, *khll2Instance));
    });
  }
};

class KHyperLogLogReidentificationPotentialFunction
    : public exec::VectorFunction {
 public:
  static std::vector<exec::FunctionSignaturePtr> signatures() {
    return {exec::FunctionSignatureBuilder()
                .returnType("double")
                .argumentType("khyperloglog")
                .argumentType("bigint")
                .build()};
  }

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& outputType,
      exec::EvalCtx& context,
      VectorPtr& result) const override {
    VELOX_CHECK_EQ(args.size(), 2);

    context.ensureWritable(rows, outputType, result);
    auto* flatResult = result->as<FlatVector<double>>();

    exec::LocalDecodedVector khllDecoder(context, *args[0], rows);
    auto decodedKhll = khllDecoder.get();

    exec::LocalDecodedVector thresholdDecoder(context, *args[1], rows);
    auto decodedThreshold = thresholdDecoder.get();

    context.applyToSelectedNoThrow(rows, [&](vector_size_t row) {
      if (decodedKhll->isNullAt(row) || decodedThreshold->isNullAt(row)) {
        flatResult->setNull(row, true);
        return;
      }

      auto khllData = decodedKhll->valueAt<StringView>(row);
      auto threshold = decodedThreshold->valueAt<int64_t>(row);

      auto khllInstance = KHyperLogLog::deserialize(
          khllData.data(), khllData.size(), context.pool());
      flatResult->set(row, khllInstance->reidentificationPotential(threshold));
    });
  }
};

class KHyperLogLogUniquenessDistributionFunction : public exec::VectorFunction {
 public:
  static std::vector<exec::FunctionSignaturePtr> signatures() {
    return {
        exec::FunctionSignatureBuilder()
            .returnType("map(bigint,double)")
            .argumentType("khyperloglog")
            .build(),
        exec::FunctionSignatureBuilder()
            .returnType("map(bigint,double)")
            .argumentType("khyperloglog")
            .argumentType("bigint")
            .build()};
  }

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& outputType,
      exec::EvalCtx& context,
      VectorPtr& result) const override {
    VELOX_CHECK(args.size() == 1 || args.size() == 2);

    context.ensureWritable(rows, outputType, result);
    auto* mapResult = result->as<MapVector>();

    exec::LocalDecodedVector khllDecoder(context, *args[0], rows);
    auto decodedKhll = khllDecoder.get();

    // Histogram size is an optional second parameter to the function, with a
    // default value of 256.
    std::unique_ptr<exec::LocalDecodedVector> histogramSizeDecoder;
    DecodedVector* decodedHistogramSize = nullptr;
    if (args.size() == 2) {
      histogramSizeDecoder =
          std::make_unique<exec::LocalDecodedVector>(context, *args[1], rows);
      decodedHistogramSize = histogramSizeDecoder->get();
    }

    // Prepare key and value vectors
    auto mapKeys = mapResult->mapKeys()->asFlatVector<int64_t>();
    auto mapValues = mapResult->mapValues()->asFlatVector<double>();

    vector_size_t currentOffset = 0;

    context.applyToSelectedNoThrow(rows, [&](vector_size_t row) {
      if (decodedKhll->isNullAt(row)) {
        mapResult->setNull(row, true);
        mapResult->setOffsetAndSize(row, 0, 0);
        return;
      }

      auto khllData = decodedKhll->valueAt<StringView>(row);
      auto khllInstance = KHyperLogLog::deserialize(
          khllData.data(), khllData.size(), context.pool());

      int64_t histogramSize =
          (decodedHistogramSize && !decodedHistogramSize->isNullAt(row))
          ? decodedHistogramSize->valueAt<int64_t>(row)
          : KHyperLogLog::kDefaultHistogramSize;

      auto distribution = khllInstance->uniquenessDistribution(histogramSize);

      // Sort the distribution by key to ensure sorted output
      std::vector<std::pair<int64_t, double>> sortedDistribution(
          distribution.begin(), distribution.end());
      std::sort(
          sortedDistribution.begin(),
          sortedDistribution.end(),
          [](const auto& a, const auto& b) { return a.first < b.first; });

      // Calculate required size
      vector_size_t mapSize = sortedDistribution.size();
      vector_size_t newOffset = currentOffset + mapSize;

      // Resize vectors to accommodate new elements
      mapKeys->resize(newOffset);
      mapValues->resize(newOffset);

      // Set the offset and size for this row
      mapResult->setOffsetAndSize(row, currentOffset, mapSize);

      // Fill in the keys and values
      vector_size_t i = 0;
      for (const auto& [key, value] : sortedDistribution) {
        mapKeys->set(currentOffset + i, key);
        mapValues->set(currentOffset + i, value);
        ++i;
      }

      currentOffset = newOffset;
    });
  }
};

class MergeKHyperLogLogFunction : public exec::VectorFunction {
 public:
  static std::vector<exec::FunctionSignaturePtr> signatures() {
    return {exec::FunctionSignatureBuilder()
                .returnType("khyperloglog")
                .argumentType("array(KHYPERLOGLOG)")
                .build()};
  }

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& outputType,
      exec::EvalCtx& context,
      VectorPtr& result) const override {
    VELOX_CHECK_EQ(args.size(), 1);

    context.ensureWritable(rows, outputType, result);
    auto* flatResult = result->as<FlatVector<StringView>>();
    exec::LocalDecodedVector arrayDecoder(context, *args[0], rows);
    auto decodedArray = arrayDecoder.get();
    auto baseArray = decodedArray->base()->as<ArrayVector>();
    auto rawSizes = baseArray->rawSizes();
    auto rawOffsets = baseArray->rawOffsets();
    auto indices = decodedArray->indices();
    auto arrayElements = baseArray->elements();

    exec::LocalSelectivityVector allElementsRows(
        context, arrayElements->size());
    allElementsRows->setAll();
    exec::LocalDecodedVector elementsDecoder(
        context, *arrayElements, *allElementsRows);
    auto decodedElements = elementsDecoder.get();

    context.applyToSelectedNoThrow(rows, [&](vector_size_t row) {
      if (decodedArray->isNullAt(row)) {
        flatResult->setNull(row, true);
        return;
      }

      auto arraySize = rawSizes[indices[row]];
      auto arrayOffset = rawOffsets[indices[row]];

      std::unique_ptr<KHyperLogLog> merged = nullptr;

      for (vector_size_t i = 0; i < arraySize; ++i) {
        auto elementIndex = arrayOffset + i;
        if (decodedElements->isNullAt(elementIndex)) {
          continue;
        }

        auto khllData = decodedElements->valueAt<StringView>(elementIndex);
        if (khllData.empty()) {
          continue;
        }
        if (!merged) {
          // Initialize with first non-null element.
          merged = KHyperLogLog::deserialize(
              khllData.data(), khllData.size(), context.pool());
        } else {
          auto currentKhll = KHyperLogLog::deserialize(
              khllData.data(), khllData.size(), context.pool());
          merged->mergeWith(*currentKhll);
        }
      }

      // Return null if all elements were null.
      if (!merged) {
        flatResult->setNull(row, true);
        return;
      }

      std::string serialized(merged->estimatedSerializedSize(), '\0');
      merged->serialize(serialized.data());

      char* buffer = flatResult->getRawStringBufferWithSpace(serialized.size());
      memcpy(buffer, serialized.data(), serialized.size());
      flatResult->setNoCopy(row, StringView(buffer, serialized.size()));
    });
  }
};

} // namespace facebook::velox::functions
