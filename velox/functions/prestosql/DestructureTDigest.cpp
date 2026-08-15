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
#include <algorithm>
#include "velox/expression/DecodedArgs.h"
#include "velox/expression/VectorFunction.h"
#include "velox/functions/lib/TDigest.h"

namespace facebook::velox::functions {

namespace {

class DestructureTDigestFunction : public exec::VectorFunction {
 public:
  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& outputType,
      exec::EvalCtx& context,
      VectorPtr& result) const override {
    const exec::DecodedArgs decodedArgs(rows, args, context);
    DecodedVector* input = decodedArgs.at(0);
    auto* pool = context.pool();
    const auto numRows = rows.end();

    auto compression =
        BaseVector::create<FlatVector<double>>(DOUBLE(), numRows, pool);
    auto minValues =
        BaseVector::create<FlatVector<double>>(DOUBLE(), numRows, pool);
    auto maxValues =
        BaseVector::create<FlatVector<double>>(DOUBLE(), numRows, pool);
    auto sumValues =
        BaseVector::create<FlatVector<double>>(DOUBLE(), numRows, pool);
    auto counts =
        BaseVector::create<FlatVector<int64_t>>(BIGINT(), numRows, pool);

    // Means and weights share offsets/sizes (identical per-row counts).
    BufferPtr offsets = allocateOffsets(numRows, pool);
    BufferPtr sizes = allocateSizes(numRows, pool);
    auto* rawOffsets = offsets->asMutable<vector_size_t>();
    auto* rawSizes = sizes->asMutable<vector_size_t>();

    std::vector<double> meansElements;
    std::vector<int32_t> weightsElements;
    vector_size_t elementOffset{0};
    rows.applyToSelected([&](vector_size_t row) {
      auto serialized = input->valueAt<StringView>(row);
      auto digest = TDigest<>::fromSerialized(serialized.data());
      compression->set(row, digest.compression());
      minValues->set(row, digest.min());
      maxValues->set(row, digest.max());
      sumValues->set(row, digest.sum());

      const double* means = digest.means();
      const double* weights = digest.weights();
      const auto numCentroids = static_cast<vector_size_t>(digest.size());
      rawOffsets[row] = elementOffset;
      rawSizes[row] = numCentroids;
      int64_t count{0};
      for (vector_size_t i = 0; i < numCentroids; ++i) {
        meansElements.push_back(means[i]);
        weightsElements.push_back(static_cast<int32_t>(weights[i]));
        count += static_cast<int64_t>(weights[i]);
      }
      elementOffset += numCentroids;
      counts->set(row, count);
    });

    auto meansVector =
        BaseVector::create<FlatVector<double>>(DOUBLE(), elementOffset, pool);
    std::copy(
        meansElements.begin(),
        meansElements.end(),
        meansVector->mutableRawValues());
    auto weightsVector =
        BaseVector::create<FlatVector<int32_t>>(INTEGER(), elementOffset, pool);
    std::copy(
        weightsElements.begin(),
        weightsElements.end(),
        weightsVector->mutableRawValues());

    auto centroidMeans = std::make_shared<ArrayVector>(
        pool,
        ARRAY(DOUBLE()),
        BufferPtr(nullptr),
        numRows,
        offsets,
        sizes,
        meansVector);
    auto centroidWeights = std::make_shared<ArrayVector>(
        pool,
        ARRAY(INTEGER()),
        BufferPtr(nullptr),
        numRows,
        offsets,
        sizes,
        weightsVector);

    auto rowResult = std::make_shared<RowVector>(
        pool,
        outputType,
        BufferPtr(nullptr),
        numRows,
        std::vector<VectorPtr>{
            centroidMeans,
            centroidWeights,
            compression,
            minValues,
            maxValues,
            sumValues,
            counts,
        });
    context.moveOrCopyResult(rowResult, rows, result);
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {exec::FunctionSignatureBuilder()
                .returnType(
                    "row(centroid_means array(double), "
                    "centroid_weights array(integer), compression double, "
                    "min double, max double, sum double, count bigint)")
                .argumentType("tdigest(double)")
                .build()};
  }
};

} // namespace

VELOX_DECLARE_VECTOR_FUNCTION(
    udf_destructure_tdigest,
    DestructureTDigestFunction::signatures(),
    std::make_unique<DestructureTDigestFunction>());

} // namespace facebook::velox::functions
