/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

#include <cstdint>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <folly/Executor.h>
#include <folly/container/F14Map.h>

#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Constants.h"
#include "velox/dwio/nimble/compression/CompressionPolicy.h"
#include "velox/dwio/nimble/encodings/common/EncodingType.h"
#include "velox/dwio/nimble/serializer/Options.h"
#include "velox/dwio/nimble/writer/EncodingLayoutTree.h"
#include "velox/vector/BaseVector.h"

namespace facebook::nimble::selection {

/// Weights applied to normalized encoding metrics during candidate ranking.
struct ScoreWeights {
  double encodeSize{1.0};
  double encodeTime{0.0};
  double decodeTime{0.0};
};

/// Configuration for an encoding evaluation.
struct EvaluationOptions {
  int32_t iterations{3};
  ScoreWeights weights;
  std::optional<nimble::CompressionOptions> compressionOptions;
  uint16_t blockBitPackingBlockSize{nimble::kBlockBitPackingBlockSize};
  bool fixedBitWidthUseExactBits{true};
  std::vector<std::pair<nimble::EncodingType, float>> readFactors;
  std::optional<std::vector<std::pair<nimble::EncodingType, float>>>
      nestedReadFactors;
  folly::F14FastMap<std::string, std::set<std::string>> flatMapColumns;
  size_t parallelism{1};
};

/// One candidate encoding to evaluate.
struct CandidateEncoding {
  nimble::EncodingType type{};
  nimble::EncodingLayoutTree tree;
};

/// Measurement plus normalized ranking metrics for one candidate.
struct EvaluationResult {
  nimble::EncodingType type{};
  uint64_t encodedBytes{};
  uint64_t encodeNanos{};
  uint64_t decodeNanos{};
  double sizeRatio{};
  double encodeRatio{};
  double decodeRatio{};
  double score{};
};

/// Measures each candidate against the given vectors. Nullopt slot means the
/// encoding was incompatible with the data. When executor is set and vectors
/// has more than one element, per-candidate measurement fans out across
/// vector chunks on the executor.
std::vector<std::optional<EvaluationResult>> evaluateCandidates(
    const std::vector<velox::VectorPtr>& vectors,
    const std::vector<CandidateEncoding>& candidates,
    const EvaluationOptions& opts,
    velox::memory::MemoryPool* pool,
    folly::Executor* executor = nullptr);

/// Normalizes results against the Trivial baseline, applies weights, and
/// returns them sorted best-first.
std::vector<EvaluationResult> rankResults(
    const std::vector<std::optional<EvaluationResult>>& results,
    const ScoreWeights& weights);

/// Evaluates the candidates and returns the winning EncodingLayoutTree.
nimble::EncodingLayoutTree getOptimalEncoding(
    const std::vector<velox::VectorPtr>& vectors,
    const std::vector<CandidateEncoding>& candidates,
    const EvaluationOptions& opts,
    velox::memory::MemoryPool* pool,
    folly::Executor* executor = nullptr);

/// Builds one CandidateEncoding per encoding type.
std::vector<CandidateEncoding> buildEncodingCandidates(
    const std::vector<nimble::EncodingType>& encodings);

} // namespace facebook::nimble::selection
