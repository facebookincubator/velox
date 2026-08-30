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

#include "velox/dwio/nimble/tools/encoding/EncodingEvaluation.h"

#include <algorithm>
#include <limits>
#include <utility>

#include <fmt/core.h>
#include <folly/futures/Future.h>

#include "velox/common/time/CpuWallTimer.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/NimbleException.h"
#include "velox/dwio/nimble/encodings/common/EncodingLayout.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"
#include "velox/dwio/nimble/serializer/Deserializer.h"
#include "velox/dwio/nimble/serializer/Serializer.h"
#include "velox/dwio/nimble/velox/OrderedRanges.h"
#include "velox/dwio/nimble/velox/SchemaSerialization.h"

namespace facebook::nimble::selection {
namespace {

struct ChunkMeasurement {
  uint64_t encodedBytes{};
  uint64_t encodeNanos{};
  uint64_t decodeNanos{};
};

uint8_t nestedChildrenCount(nimble::EncodingType encodingType) {
  switch (encodingType) {
    case nimble::EncodingType::Delta:
    case nimble::EncodingType::BlockBitPacking:
    case nimble::EncodingType::FOR:
      return 3;
    case nimble::EncodingType::RLE:
    case nimble::EncodingType::Dictionary:
    case nimble::EncodingType::MainlyConstant:
    case nimble::EncodingType::MainlyConstantV2:
    case nimble::EncodingType::PFOR:
      return 2;
    case nimble::EncodingType::SparseBool:
    case nimble::EncodingType::Trivial:
      return 1;
    default:
      return 0;
  }
}

nimble::EncodingLayoutTree buildScalarOverrideTree(nimble::EncodingType type) {
  using StreamIds = nimble::EncodingLayoutTree::StreamIdentifiers;
  std::vector<std::optional<const nimble::EncodingLayout>> nestedChildren(
      nestedChildrenCount(type), std::nullopt);
  return nimble::EncodingLayoutTree(
      nimble::Kind::Scalar,
      {{StreamIds::Scalar::ScalarStream,
        nimble::EncodingLayout(
            type,
            {},
            nimble::CompressionType::Uncompressed,
            std::move(nestedChildren))}},
      "");
}

nimble::SerializerOptions buildSerializerOptions(
    const CandidateEncoding& candidate,
    const EvaluationOptions& opts) {
  nimble::SerializerOptions options;
  options.version = nimble::SerializationVersion::kSerialization;
  options.encodingOptions.blockBitPackingBlockSize =
      opts.blockBitPackingBlockSize;
  options.encodingOptions.fixedBitWidthUseExactBits =
      opts.fixedBitWidthUseExactBits;
  options.compressionOptions = opts.compressionOptions;
  options.flatMapColumns = opts.flatMapColumns;

  auto readFactors = opts.readFactors;
  if (readFactors.empty()) {
    readFactors = nimble::ManualEncodingSelectionPolicyFactory::
        defaultEncodingReadFactors();
  }
  nimble::ManualEncodingSelectionPolicyFactory factory{
      std::move(readFactors), opts.compressionOptions, opts.nestedReadFactors};
  options.encodingSelectionPolicyCreator =
      [factory = std::move(factory)](nimble::DataType dataType)
      -> std::unique_ptr<nimble::EncodingSelectionPolicyBase> {
    return factory.createPolicy(dataType);
  };

  options.encodingLayoutTree.emplace(candidate.tree);
  return options;
}

ChunkMeasurement evaluateCandidateEncodingOnce(
    const CandidateEncoding& candidate,
    const std::vector<velox::VectorPtr>& vectors,
    const EvaluationOptions& opts,
    velox::memory::MemoryPool* pool) {
  auto serializerOptions = buildSerializerOptions(candidate, opts);
  nimble::Serializer serializer{
      serializerOptions, vectors.front()->type(), pool};

  velox::CpuWallTiming encodeTiming;
  std::vector<std::string> serializedBuffers;
  serializedBuffers.reserve(vectors.size());
  uint64_t chunkBytes{0};

  for (const auto& vector : vectors) {
    nimble::OrderedRanges ranges;
    ranges.add(0, vector->size());
    std::string_view serialized;
    {
      velox::CpuWallTimer timer(encodeTiming);
      serialized = serializer.serialize(vector, ranges);
    }
    serializedBuffers.emplace_back(serialized);
    chunkBytes += serialized.size();
  }

  nimble::SchemaSerializer schemaSerializer;
  const auto serializedSchema =
      schemaSerializer.serialize(serializer.schemaBuilder());
  auto schema = nimble::SchemaDeserializer::deserialize(serializedSchema);
  nimble::DeserializerOptions deserializerOptions{.hasHeader = true};
  nimble::Deserializer deserializer(schema, pool, deserializerOptions);

  velox::CpuWallTiming decodeTiming;
  for (const auto& buffer : serializedBuffers) {
    velox::CpuWallTimer timer(decodeTiming);
    velox::VectorPtr deserialized;
    deserializer.deserialize(buffer, deserialized);
  }

  return {chunkBytes, encodeTiming.wallNanos, decodeTiming.wallNanos};
}

ChunkMeasurement evaluateCandidateEncodingParallel(
    const CandidateEncoding& candidate,
    const std::vector<velox::VectorPtr>& vectors,
    const EvaluationOptions& opts,
    velox::memory::MemoryPool* pool,
    folly::Executor* executor) {
  const auto parallelism =
      std::min<size_t>(vectors.size(), std::max<size_t>(opts.parallelism, 1));
  const auto chunkSize = (vectors.size() + parallelism - 1) / parallelism;
  std::vector<folly::Future<ChunkMeasurement>> futures;
  for (size_t t = 0; t < parallelism; ++t) {
    const auto start = t * chunkSize;
    const auto end = std::min(start + chunkSize, vectors.size());
    if (start >= end) {
      break;
    }
    std::vector<velox::VectorPtr> chunkVectors(
        vectors.begin() + start, vectors.begin() + end);
    auto chunkPool = pool->addLeafChild(fmt::format("chunk_{}", t));
    futures.emplace_back(
        folly::via(
            executor,
            [&candidate,
             &opts,
             chunkVectors = std::move(chunkVectors),
             chunkPool = std::move(chunkPool)]() {
              return evaluateCandidateEncodingOnce(
                  candidate, chunkVectors, opts, chunkPool.get());
            }));
  }
  auto perChunk = folly::collectAll(std::move(futures)).get();
  ChunkMeasurement total;
  for (auto& chunkResult : perChunk) {
    const auto& measurement = chunkResult.value();
    total.encodedBytes += measurement.encodedBytes;
    total.encodeNanos += measurement.encodeNanos;
    total.decodeNanos += measurement.decodeNanos;
  }
  return total;
}

std::optional<EvaluationResult> evaluateCandidateEncoding(
    const CandidateEncoding& candidate,
    const std::vector<velox::VectorPtr>& vectors,
    const EvaluationOptions& opts,
    velox::memory::MemoryPool* pool,
    folly::Executor* executor) {
  uint64_t encodeNanos{std::numeric_limits<uint64_t>::max()};
  uint64_t decodeNanos{std::numeric_limits<uint64_t>::max()};
  uint64_t encodedBytes{0};

  for (int32_t iter = 0; iter < opts.iterations; ++iter) {
    try {
      const auto chunk =
          (executor != nullptr && opts.parallelism > 1 && vectors.size() > 1)
          ? evaluateCandidateEncodingParallel(
                candidate, vectors, opts, pool, executor)
          : evaluateCandidateEncodingOnce(candidate, vectors, opts, pool);
      encodeNanos = std::min(encodeNanos, chunk.encodeNanos);
      decodeNanos = std::min(decodeNanos, chunk.decodeNanos);
      encodedBytes = chunk.encodedBytes;
    } catch (const nimble::NimbleUserError& error) {
      if (error.errorCode() != nimble::error_code::IncompatibleEncoding) {
        throw;
      }
      return std::nullopt;
    }
  }

  return EvaluationResult{
      .type = candidate.type,
      .encodedBytes = encodedBytes,
      .encodeNanos = encodeNanos,
      .decodeNanos = decodeNanos,
  };
}

const EvaluationResult& findTrivialEncodingResult(
    const std::vector<std::optional<EvaluationResult>>& results) {
  for (const auto& result : results) {
    if (result.has_value() && result->type == nimble::EncodingType::Trivial) {
      return *result;
    }
  }
  NIMBLE_USER_CHECK(
      false, "Trivial encoding required for result normalization.");
}

} // namespace

std::vector<std::optional<EvaluationResult>> evaluateCandidates(
    const std::vector<velox::VectorPtr>& vectors,
    const std::vector<CandidateEncoding>& candidates,
    const EvaluationOptions& opts,
    velox::memory::MemoryPool* pool,
    folly::Executor* executor) {
  NIMBLE_USER_CHECK(
      !vectors.empty(), "No data provided for encoding evaluation.");
  NIMBLE_USER_CHECK(
      opts.iterations > 0, "EvaluationOptions.iterations must be positive.");
  NIMBLE_USER_CHECK(
      std::any_of(
          candidates.begin(),
          candidates.end(),
          [](const auto& c) {
            return c.type == nimble::EncodingType::Trivial;
          }),
      "Trivial encoding required for result normalization.");

  std::vector<std::optional<EvaluationResult>> results;
  results.reserve(candidates.size());
  for (const auto& candidate : candidates) {
    results.emplace_back(
        evaluateCandidateEncoding(candidate, vectors, opts, pool, executor));
  }
  return results;
}

std::vector<EvaluationResult> rankResults(
    const std::vector<std::optional<EvaluationResult>>& results,
    const ScoreWeights& weights) {
  const auto& baseline = findTrivialEncodingResult(results);
  NIMBLE_CHECK(
      baseline.encodedBytes > 0,
      "Trivial baseline encodedBytes must be positive.");
  NIMBLE_CHECK(
      baseline.encodeNanos > 0,
      "Trivial baseline encodeNanos must be positive.");
  NIMBLE_CHECK(
      baseline.decodeNanos > 0,
      "Trivial baseline decodeNanos must be positive.");

  std::vector<EvaluationResult> ranked;
  ranked.reserve(results.size());
  for (const auto& result : results) {
    if (!result.has_value()) {
      continue;
    }
    EvaluationResult copy = *result;
    copy.sizeRatio =
        static_cast<double>(copy.encodedBytes) / baseline.encodedBytes;
    copy.encodeRatio =
        static_cast<double>(copy.encodeNanos) / baseline.encodeNanos;
    copy.decodeRatio =
        static_cast<double>(copy.decodeNanos) / baseline.decodeNanos;
    copy.score = weights.encodeSize * copy.sizeRatio +
        weights.encodeTime * copy.encodeRatio +
        weights.decodeTime * copy.decodeRatio;
    ranked.emplace_back(std::move(copy));
  }

  std::sort(ranked.begin(), ranked.end(), [](const auto& lhs, const auto& rhs) {
    return lhs.score < rhs.score;
  });
  return ranked;
}

nimble::EncodingLayoutTree getOptimalEncoding(
    const std::vector<velox::VectorPtr>& vectors,
    const std::vector<CandidateEncoding>& candidates,
    const EvaluationOptions& opts,
    velox::memory::MemoryPool* pool,
    folly::Executor* executor) {
  auto rawResults =
      evaluateCandidates(vectors, candidates, opts, pool, executor);
  auto ranked = rankResults(rawResults, opts.weights);
  NIMBLE_USER_CHECK(!ranked.empty());
  const auto best = ranked.front().type;
  for (const auto& candidate : candidates) {
    if (candidate.type == best) {
      return candidate.tree;
    }
  }
  NIMBLE_UNREACHABLE();
}

std::vector<CandidateEncoding> buildEncodingCandidates(
    const std::vector<nimble::EncodingType>& encodings) {
  std::vector<CandidateEncoding> candidates;
  candidates.reserve(encodings.size());
  for (const auto encoding : encodings) {
    candidates.push_back({encoding, buildScalarOverrideTree(encoding)});
  }
  return candidates;
}

} // namespace facebook::nimble::selection
