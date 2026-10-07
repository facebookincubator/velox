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

namespace facebook::nimble {
/// WARNING: These values have been derived experimentally.

/// Attempt to compress a data stream only if the data size is equal or greater
/// than this threshold.
constexpr uint32_t kMetaInternalMinCompressionSize{40};
constexpr uint32_t kZstdMinCompressionSize{25};
constexpr uint32_t kLz4MinCompressionSize{12};
constexpr uint32_t kOpenZLMinCompressionSize{40};

/// Default options for the ChunkFlushPolicy.
/// Threshold to trigger chunking to relieve memory pressure
constexpr uint64_t kChunkingWriterMemoryHighThreshold{2ULL << 30}; // 2 GB
/// Threshold below which chunking stops.
constexpr uint64_t kChunkingWriterMemoryLowThreshold{1ULL << 30}; // 1 GB
/// Target size for encoded stripes.
constexpr uint64_t kChunkingWriterTargetStripeStorageSize{100 << 20}; // 100MB
/// Expected ratio of raw to encoded data.
constexpr double kChunkingWriterEstimatedCompressionFactor{3.7};
/// When flushing data streams into chunks, streams with raw data size smaller
/// than this threshold will not be flushed.
/// Note: this threshold is ignored when it is time to flush a stripe.
constexpr uint64_t kChunkingWriterMinChunkSize{512 << 10}; // 512KB
/// When flushing data streams into chunks, streams with raw data size larger
/// than this threshold will be broken down into multiple smaller chunks. Each
/// chunk will be at most this size.
constexpr uint64_t kChunkingWriterMaxChunkSize{20 << 20}; // 20MB
/// Used in place of kChunkingWriterMaxChunkSize for tables with large schemas.
constexpr uint64_t kChunkingWriterWideSchemaMaxChunkSize{2 << 20}; // 2MB

/// Default block size for BlockBitPacking encoding and its statistics.
constexpr uint16_t kBlockBitPackingBlockSize{1024};

} // namespace facebook::nimble
