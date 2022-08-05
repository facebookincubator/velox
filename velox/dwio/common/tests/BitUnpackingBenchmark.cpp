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

#include "velox/common/base/BitUtil.h"
#include "velox/dwio/common/BitUnpacking.h"
#include "velox/dwio/common/tests/Lemire/bmipacking32.h"
#include "velox/external/duckdb/duckdb-fastpforlib.hpp"
#include "velox/external/duckdb/parquet-amalgamation.hpp"

#include <arrow/util/rle_encoding.h> // @manual
#include <folly/Benchmark.h>
#include <glog/logging.h>

#include <random>

using namespace folly;
using namespace facebook::velox;

#define BYTES(bitWidth) (kNumValues * bitWidth + 7) / 8
#define INPUT_BUFFER(bitWidth) inputBuffer##bitWidth

static const uint64_t kNumValues = 1024768 * 8;

// We run benchmarks for bitWidth up to 16.
std::vector<uint8_t> input(BYTES(16), 0);
const uint8_t* inputBuffer = input.data();

// We run benchmarks for output size up to 4 bytes (uint32_t)
std::vector<uint8_t> output(kNumValues * sizeof(uint32_t), 0);
uint8_t* outputBuffer = output.data();

std::vector<uint32_t> rows(kNumValues);

// BENCHMARK_NAMED_PARAM was not used to reduce the noise caused by not
// inlining.

BENCHMARK(duckdb_1_8) {
  duckdb::ByteBuffer duckInputBuffer(
      reinterpret_cast<char*>(const_cast<uint8_t*>((inputBuffer))), BYTES(16));
  uint8_t bitpack_pos = 0;
  duckdb::ParquetDecodeUtils::BitUnpack<uint8_t>(
      duckInputBuffer, bitpack_pos, outputBuffer, kNumValues, 1);
}

BENCHMARK(duckdb_2_8) {
  duckdb::ByteBuffer duckInputBuffer(
      reinterpret_cast<char*>(const_cast<uint8_t*>((inputBuffer))), BYTES(16));
  uint8_t bitpack_pos = 0;
  duckdb::ParquetDecodeUtils::BitUnpack<uint8_t>(
      duckInputBuffer, bitpack_pos, outputBuffer, kNumValues, 2);
}

BENCHMARK(duckdb_3_8) {
  duckdb::ByteBuffer duckInputBuffer(
      reinterpret_cast<char*>(const_cast<uint8_t*>((inputBuffer))), BYTES(16));
  uint8_t bitpack_pos = 0;
  duckdb::ParquetDecodeUtils::BitUnpack<uint8_t>(
      duckInputBuffer, bitpack_pos, outputBuffer, kNumValues, 3);
}

BENCHMARK(duckdb_4_8) {
  duckdb::ByteBuffer duckInputBuffer(
      reinterpret_cast<char*>(const_cast<uint8_t*>((inputBuffer))), BYTES(16));
  uint8_t bitpack_pos = 0;
  duckdb::ParquetDecodeUtils::BitUnpack<uint8_t>(
      duckInputBuffer, bitpack_pos, outputBuffer, kNumValues, 4);
}

BENCHMARK(duckdb_5_8) {
  duckdb::ByteBuffer duckInputBuffer(
      reinterpret_cast<char*>(const_cast<uint8_t*>((inputBuffer))), BYTES(16));
  uint8_t bitpack_pos = 0;
  duckdb::ParquetDecodeUtils::BitUnpack<uint8_t>(
      duckInputBuffer, bitpack_pos, outputBuffer, kNumValues, 5);
}

BENCHMARK(duckdb_6_8) {
  duckdb::ByteBuffer duckInputBuffer(
      reinterpret_cast<char*>(const_cast<uint8_t*>((inputBuffer))), BYTES(16));
  uint8_t bitpack_pos = 0;
  duckdb::ParquetDecodeUtils::BitUnpack<uint8_t>(
      duckInputBuffer, bitpack_pos, outputBuffer, kNumValues, 6);
}

BENCHMARK(duckdb_7_8) {
  duckdb::ByteBuffer duckInputBuffer(
      reinterpret_cast<char*>(const_cast<uint8_t*>((inputBuffer))), BYTES(16));
  uint8_t bitpack_pos = 0;
  duckdb::ParquetDecodeUtils::BitUnpack<uint8_t>(
      duckInputBuffer, bitpack_pos, outputBuffer, kNumValues, 7);
}

BENCHMARK(duckdb_8_8) {
  duckdb::ByteBuffer duckInputBuffer(
      reinterpret_cast<char*>(const_cast<uint8_t*>((inputBuffer))), BYTES(16));
  uint8_t bitpack_pos = 0;
  duckdb::ParquetDecodeUtils::BitUnpack<uint8_t>(
      duckInputBuffer, bitpack_pos, outputBuffer, kNumValues, 8);
}

BENCHMARK(duckdb_1_16) {
  duckdb::ByteBuffer duckInputBuffer(
      reinterpret_cast<char*>(const_cast<uint8_t*>((inputBuffer))), BYTES(16));
  uint8_t bitpack_pos = 0;
  uint16_t* duckOutputBuffer = reinterpret_cast<uint16_t*>(outputBuffer);
  duckdb::ParquetDecodeUtils::BitUnpack<uint16_t>(
      duckInputBuffer, bitpack_pos, duckOutputBuffer, kNumValues, 1);
}

BENCHMARK(duckdb_2_16) {
  duckdb::ByteBuffer duckInputBuffer(
      reinterpret_cast<char*>(const_cast<uint8_t*>((inputBuffer))), BYTES(16));
  uint8_t bitpack_pos = 0;
  uint16_t* duckOutputBuffer = reinterpret_cast<uint16_t*>(outputBuffer);
  duckdb::ParquetDecodeUtils::BitUnpack<uint16_t>(
      duckInputBuffer, bitpack_pos, duckOutputBuffer, kNumValues, 2);
}

BENCHMARK(duckdb_3_16) {
  duckdb::ByteBuffer duckInputBuffer(
      reinterpret_cast<char*>(const_cast<uint8_t*>((inputBuffer))), BYTES(16));
  uint8_t bitpack_pos = 0;
  uint16_t* duckOutputBuffer = reinterpret_cast<uint16_t*>(outputBuffer);
  duckdb::ParquetDecodeUtils::BitUnpack<uint16_t>(
      duckInputBuffer, bitpack_pos, duckOutputBuffer, kNumValues, 3);
}

BENCHMARK(duckdb_4_16) {
  duckdb::ByteBuffer duckInputBuffer(
      reinterpret_cast<char*>(const_cast<uint8_t*>((inputBuffer))), BYTES(16));
  uint8_t bitpack_pos = 0;
  uint16_t* duckOutputBuffer = reinterpret_cast<uint16_t*>(outputBuffer);
  duckdb::ParquetDecodeUtils::BitUnpack<uint16_t>(
      duckInputBuffer, bitpack_pos, duckOutputBuffer, kNumValues, 4);
}

BENCHMARK(duckdb_5_16) {
  duckdb::ByteBuffer duckInputBuffer(
      reinterpret_cast<char*>(const_cast<uint8_t*>((inputBuffer))), BYTES(16));
  uint8_t bitpack_pos = 0;
  uint16_t* duckOutputBuffer = reinterpret_cast<uint16_t*>(outputBuffer);
  duckdb::ParquetDecodeUtils::BitUnpack<uint16_t>(
      duckInputBuffer, bitpack_pos, duckOutputBuffer, kNumValues, 5);
}

BENCHMARK(duckdb_6_16) {
  duckdb::ByteBuffer duckInputBuffer(
      reinterpret_cast<char*>(const_cast<uint8_t*>((inputBuffer))), BYTES(16));
  uint8_t bitpack_pos = 0;
  uint16_t* duckOutputBuffer = reinterpret_cast<uint16_t*>(outputBuffer);
  duckdb::ParquetDecodeUtils::BitUnpack<uint16_t>(
      duckInputBuffer, bitpack_pos, duckOutputBuffer, kNumValues, 6);
}

BENCHMARK(duckdb_7_16) {
  duckdb::ByteBuffer duckInputBuffer(
      reinterpret_cast<char*>(const_cast<uint8_t*>((inputBuffer))), BYTES(16));
  uint8_t bitpack_pos = 0;
  uint16_t* duckOutputBuffer = reinterpret_cast<uint16_t*>(outputBuffer);

  duckdb::ParquetDecodeUtils::BitUnpack<uint16_t>(
      duckInputBuffer, bitpack_pos, duckOutputBuffer, kNumValues, 7);
}

BENCHMARK(duckdb_8_16) {
  duckdb::ByteBuffer duckInputBuffer(
      reinterpret_cast<char*>(const_cast<uint8_t*>((inputBuffer))), BYTES(16));
  uint8_t bitpack_pos = 0;
  uint16_t* duckOutputBuffer = reinterpret_cast<uint16_t*>(outputBuffer);

  duckdb::ParquetDecodeUtils::BitUnpack<uint16_t>(
      duckInputBuffer, bitpack_pos, duckOutputBuffer, kNumValues, 8);
}

BENCHMARK(duckdb_9_16) {
  duckdb::ByteBuffer duckInputBuffer(
      reinterpret_cast<char*>(const_cast<uint8_t*>((inputBuffer))), BYTES(16));
  uint8_t bitpack_pos = 0;
  uint16_t* duckOutputBuffer = reinterpret_cast<uint16_t*>(outputBuffer);
  duckdb::ParquetDecodeUtils::BitUnpack<uint16_t>(
      duckInputBuffer, bitpack_pos, duckOutputBuffer, kNumValues, 9);
}

BENCHMARK(duckdb_10_16) {
  duckdb::ByteBuffer duckInputBuffer(
      reinterpret_cast<char*>(const_cast<uint8_t*>((inputBuffer))), BYTES(16));
  uint8_t bitpack_pos = 0;
  uint16_t* duckOutputBuffer = reinterpret_cast<uint16_t*>(outputBuffer);
  duckdb::ParquetDecodeUtils::BitUnpack<uint16_t>(
      duckInputBuffer, bitpack_pos, duckOutputBuffer, kNumValues, 10);
}

BENCHMARK(duckdb_11_16) {
  duckdb::ByteBuffer duckInputBuffer(
      reinterpret_cast<char*>(const_cast<uint8_t*>((inputBuffer))), BYTES(16));
  uint8_t bitpack_pos = 0;
  uint16_t* duckOutputBuffer = reinterpret_cast<uint16_t*>(outputBuffer);
  duckdb::ParquetDecodeUtils::BitUnpack<uint16_t>(
      duckInputBuffer, bitpack_pos, duckOutputBuffer, kNumValues, 11);
}

BENCHMARK(duckdb_12_16) {
  duckdb::ByteBuffer duckInputBuffer(
      reinterpret_cast<char*>(const_cast<uint8_t*>((inputBuffer))), BYTES(16));
  uint8_t bitpack_pos = 0;
  uint16_t* duckOutputBuffer = reinterpret_cast<uint16_t*>(outputBuffer);
  duckdb::ParquetDecodeUtils::BitUnpack<uint16_t>(
      duckInputBuffer, bitpack_pos, duckOutputBuffer, kNumValues, 12);
}

BENCHMARK(duckdb_13_16) {
  duckdb::ByteBuffer duckInputBuffer(
      reinterpret_cast<char*>(const_cast<uint8_t*>((inputBuffer))), BYTES(16));
  uint8_t bitpack_pos = 0;
  uint16_t* duckOutputBuffer = reinterpret_cast<uint16_t*>(outputBuffer);
  duckdb::ParquetDecodeUtils::BitUnpack<uint16_t>(
      duckInputBuffer, bitpack_pos, duckOutputBuffer, kNumValues, 13);
}

BENCHMARK(duckdb_14_16) {
  duckdb::ByteBuffer duckInputBuffer(
      reinterpret_cast<char*>(const_cast<uint8_t*>((inputBuffer))), BYTES(16));
  uint8_t bitpack_pos = 0;
  uint16_t* duckOutputBuffer = reinterpret_cast<uint16_t*>(outputBuffer);
  duckdb::ParquetDecodeUtils::BitUnpack<uint16_t>(
      duckInputBuffer, bitpack_pos, duckOutputBuffer, kNumValues, 14);
}

BENCHMARK(duckdb_15_16) {
  duckdb::ByteBuffer duckInputBuffer(
      reinterpret_cast<char*>(const_cast<uint8_t*>((inputBuffer))), BYTES(16));
  uint8_t bitpack_pos = 0;
  uint16_t* duckOutputBuffer = reinterpret_cast<uint16_t*>(outputBuffer);

  duckdb::ParquetDecodeUtils::BitUnpack<uint16_t>(
      duckInputBuffer, bitpack_pos, duckOutputBuffer, kNumValues, 15);
}

BENCHMARK(duckdb_16_16) {
  duckdb::ByteBuffer duckInputBuffer(
      reinterpret_cast<char*>(const_cast<uint8_t*>((inputBuffer))), BYTES(16));
  uint8_t bitpack_pos = 0;
  uint16_t* duckOutputBuffer = reinterpret_cast<uint16_t*>(outputBuffer);

  duckdb::ParquetDecodeUtils::BitUnpack<uint16_t>(
      duckInputBuffer, bitpack_pos, duckOutputBuffer, kNumValues, 16);
}

BENCHMARK(duckdb_1_32) {
  duckdb::ByteBuffer duckInputBuffer(
      reinterpret_cast<char*>(const_cast<uint8_t*>((inputBuffer))), BYTES(32));
  uint8_t bitpack_pos = 0;
  uint32_t* duckOutputBuffer = reinterpret_cast<uint32_t*>(outputBuffer);
  duckdb::ParquetDecodeUtils::BitUnpack<uint32_t>(
      duckInputBuffer, bitpack_pos, duckOutputBuffer, kNumValues, 1);
}

BENCHMARK(duckdb_2_32) {
  duckdb::ByteBuffer duckInputBuffer(
      reinterpret_cast<char*>(const_cast<uint8_t*>((inputBuffer))), BYTES(32));
  uint8_t bitpack_pos = 0;
  uint32_t* duckOutputBuffer = reinterpret_cast<uint32_t*>(outputBuffer);
  duckdb::ParquetDecodeUtils::BitUnpack<uint32_t>(
      duckInputBuffer, bitpack_pos, duckOutputBuffer, kNumValues, 2);
}

BENCHMARK(duckdb_3_32) {
  duckdb::ByteBuffer duckInputBuffer(
      reinterpret_cast<char*>(const_cast<uint8_t*>((inputBuffer))), BYTES(32));
  uint8_t bitpack_pos = 0;
  uint32_t* duckOutputBuffer = reinterpret_cast<uint32_t*>(outputBuffer);
  duckdb::ParquetDecodeUtils::BitUnpack<uint32_t>(
      duckInputBuffer, bitpack_pos, duckOutputBuffer, kNumValues, 3);
}

BENCHMARK(duckdb_4_32) {
  duckdb::ByteBuffer duckInputBuffer(
      reinterpret_cast<char*>(const_cast<uint8_t*>((inputBuffer))), BYTES(32));
  uint8_t bitpack_pos = 0;
  uint32_t* duckOutputBuffer = reinterpret_cast<uint32_t*>(outputBuffer);
  duckdb::ParquetDecodeUtils::BitUnpack<uint32_t>(
      duckInputBuffer, bitpack_pos, duckOutputBuffer, kNumValues, 4);
}

BENCHMARK(duckdb_5_32) {
  duckdb::ByteBuffer duckInputBuffer(
      reinterpret_cast<char*>(const_cast<uint8_t*>((inputBuffer))), BYTES(32));
  uint8_t bitpack_pos = 0;
  uint32_t* duckOutputBuffer = reinterpret_cast<uint32_t*>(outputBuffer);
  duckdb::ParquetDecodeUtils::BitUnpack<uint32_t>(
      duckInputBuffer, bitpack_pos, duckOutputBuffer, kNumValues, 5);
}

BENCHMARK(duckdb_6_32) {
  duckdb::ByteBuffer duckInputBuffer(
      reinterpret_cast<char*>(const_cast<uint8_t*>((inputBuffer))), BYTES(32));
  uint8_t bitpack_pos = 0;
  uint32_t* duckOutputBuffer = reinterpret_cast<uint32_t*>(outputBuffer);
  duckdb::ParquetDecodeUtils::BitUnpack<uint32_t>(
      duckInputBuffer, bitpack_pos, duckOutputBuffer, kNumValues, 6);
}

BENCHMARK(duckdb_7_32) {
  duckdb::ByteBuffer duckInputBuffer(
      reinterpret_cast<char*>(const_cast<uint8_t*>((inputBuffer))), BYTES(32));
  uint8_t bitpack_pos = 0;
  uint32_t* duckOutputBuffer = reinterpret_cast<uint32_t*>(outputBuffer);

  duckdb::ParquetDecodeUtils::BitUnpack<uint32_t>(
      duckInputBuffer, bitpack_pos, duckOutputBuffer, kNumValues, 7);
}

BENCHMARK(duckdb_8_32) {
  duckdb::ByteBuffer duckInputBuffer(
      reinterpret_cast<char*>(const_cast<uint8_t*>((inputBuffer))), BYTES(32));
  uint8_t bitpack_pos = 0;
  uint32_t* duckOutputBuffer = reinterpret_cast<uint32_t*>(outputBuffer);

  duckdb::ParquetDecodeUtils::BitUnpack<uint32_t>(
      duckInputBuffer, bitpack_pos, duckOutputBuffer, kNumValues, 8);
}

BENCHMARK(duckdb_9_32) {
  duckdb::ByteBuffer duckInputBuffer(
      reinterpret_cast<char*>(const_cast<uint8_t*>((inputBuffer))), BYTES(32));
  uint8_t bitpack_pos = 0;
  uint32_t* duckOutputBuffer = reinterpret_cast<uint32_t*>(outputBuffer);
  duckdb::ParquetDecodeUtils::BitUnpack<uint32_t>(
      duckInputBuffer, bitpack_pos, duckOutputBuffer, kNumValues, 9);
}

BENCHMARK(duckdb_10_32) {
  duckdb::ByteBuffer duckInputBuffer(
      reinterpret_cast<char*>(const_cast<uint8_t*>((inputBuffer))), BYTES(32));
  uint8_t bitpack_pos = 0;
  uint32_t* duckOutputBuffer = reinterpret_cast<uint32_t*>(outputBuffer);
  duckdb::ParquetDecodeUtils::BitUnpack<uint32_t>(
      duckInputBuffer, bitpack_pos, duckOutputBuffer, kNumValues, 10);
}

BENCHMARK(duckdb_11_32) {
  duckdb::ByteBuffer duckInputBuffer(
      reinterpret_cast<char*>(const_cast<uint8_t*>((inputBuffer))), BYTES(32));
  uint8_t bitpack_pos = 0;
  uint32_t* duckOutputBuffer = reinterpret_cast<uint32_t*>(outputBuffer);
  duckdb::ParquetDecodeUtils::BitUnpack<uint32_t>(
      duckInputBuffer, bitpack_pos, duckOutputBuffer, kNumValues, 11);
}

BENCHMARK(duckdb_12_32) {
  duckdb::ByteBuffer duckInputBuffer(
      reinterpret_cast<char*>(const_cast<uint8_t*>((inputBuffer))), BYTES(32));
  uint8_t bitpack_pos = 0;
  uint32_t* duckOutputBuffer = reinterpret_cast<uint32_t*>(outputBuffer);
  duckdb::ParquetDecodeUtils::BitUnpack<uint32_t>(
      duckInputBuffer, bitpack_pos, duckOutputBuffer, kNumValues, 12);
}

BENCHMARK(duckdb_13_32) {
  duckdb::ByteBuffer duckInputBuffer(
      reinterpret_cast<char*>(const_cast<uint8_t*>((inputBuffer))), BYTES(32));
  uint8_t bitpack_pos = 0;
  uint32_t* duckOutputBuffer = reinterpret_cast<uint32_t*>(outputBuffer);
  duckdb::ParquetDecodeUtils::BitUnpack<uint32_t>(
      duckInputBuffer, bitpack_pos, duckOutputBuffer, kNumValues, 13);
}

BENCHMARK(duckdb_14_32) {
  duckdb::ByteBuffer duckInputBuffer(
      reinterpret_cast<char*>(const_cast<uint8_t*>((inputBuffer))), BYTES(32));
  uint8_t bitpack_pos = 0;
  uint32_t* duckOutputBuffer = reinterpret_cast<uint32_t*>(outputBuffer);
  duckdb::ParquetDecodeUtils::BitUnpack<uint32_t>(
      duckInputBuffer, bitpack_pos, duckOutputBuffer, kNumValues, 14);
}

BENCHMARK(duckdb_15_32) {
  duckdb::ByteBuffer duckInputBuffer(
      reinterpret_cast<char*>(const_cast<uint8_t*>((inputBuffer))), BYTES(32));
  uint8_t bitpack_pos = 0;
  uint32_t* duckOutputBuffer = reinterpret_cast<uint32_t*>(outputBuffer);

  duckdb::ParquetDecodeUtils::BitUnpack<uint32_t>(
      duckInputBuffer, bitpack_pos, duckOutputBuffer, kNumValues, 15);
}

BENCHMARK(duckdb_16_32) {
  duckdb::ByteBuffer duckInputBuffer(
      reinterpret_cast<char*>(const_cast<uint8_t*>((inputBuffer))), BYTES(32));
  uint8_t bitpack_pos = 0;
  uint32_t* duckOutputBuffer = reinterpret_cast<uint32_t*>(outputBuffer);

  duckdb::ParquetDecodeUtils::BitUnpack<uint32_t>(
      duckInputBuffer, bitpack_pos, duckOutputBuffer, kNumValues, 16);
}
BENCHMARK(duckdb_17_32) {
  duckdb::ByteBuffer duckInputBuffer(
      reinterpret_cast<char*>(const_cast<uint8_t*>((inputBuffer))), BYTES(32));
  uint8_t bitpack_pos = 0;
  uint32_t* duckOutputBuffer = reinterpret_cast<uint32_t*>(outputBuffer);
  duckdb::ParquetDecodeUtils::BitUnpack<uint32_t>(
      duckInputBuffer, bitpack_pos, duckOutputBuffer, kNumValues, 17);
}

BENCHMARK(duckdb_18_32) {
  duckdb::ByteBuffer duckInputBuffer(
      reinterpret_cast<char*>(const_cast<uint8_t*>((inputBuffer))), BYTES(32));
  uint8_t bitpack_pos = 0;
  uint32_t* duckOutputBuffer = reinterpret_cast<uint32_t*>(outputBuffer);
  duckdb::ParquetDecodeUtils::BitUnpack<uint32_t>(
      duckInputBuffer, bitpack_pos, duckOutputBuffer, kNumValues, 18);
}

BENCHMARK(duckdb_19_32) {
  duckdb::ByteBuffer duckInputBuffer(
      reinterpret_cast<char*>(const_cast<uint8_t*>((inputBuffer))), BYTES(32));
  uint8_t bitpack_pos = 0;
  uint32_t* duckOutputBuffer = reinterpret_cast<uint32_t*>(outputBuffer);
  duckdb::ParquetDecodeUtils::BitUnpack<uint32_t>(
      duckInputBuffer, bitpack_pos, duckOutputBuffer, kNumValues, 19);
}

BENCHMARK(duckdb_20_32) {
  duckdb::ByteBuffer duckInputBuffer(
      reinterpret_cast<char*>(const_cast<uint8_t*>((inputBuffer))), BYTES(32));
  uint8_t bitpack_pos = 0;
  uint32_t* duckOutputBuffer = reinterpret_cast<uint32_t*>(outputBuffer);
  duckdb::ParquetDecodeUtils::BitUnpack<uint32_t>(
      duckInputBuffer, bitpack_pos, duckOutputBuffer, kNumValues, 20);
}

BENCHMARK(duckdb_21_32) {
  duckdb::ByteBuffer duckInputBuffer(
      reinterpret_cast<char*>(const_cast<uint8_t*>((inputBuffer))), BYTES(32));
  uint8_t bitpack_pos = 0;
  uint32_t* duckOutputBuffer = reinterpret_cast<uint32_t*>(outputBuffer);
  duckdb::ParquetDecodeUtils::BitUnpack<uint32_t>(
      duckInputBuffer, bitpack_pos, duckOutputBuffer, kNumValues, 21);
}

BENCHMARK(duckdb_22_32) {
  duckdb::ByteBuffer duckInputBuffer(
      reinterpret_cast<char*>(const_cast<uint8_t*>((inputBuffer))), BYTES(32));
  uint8_t bitpack_pos = 0;
  uint32_t* duckOutputBuffer = reinterpret_cast<uint32_t*>(outputBuffer);
  duckdb::ParquetDecodeUtils::BitUnpack<uint32_t>(
      duckInputBuffer, bitpack_pos, duckOutputBuffer, kNumValues, 22);
}

BENCHMARK(duckdb_23_32) {
  duckdb::ByteBuffer duckInputBuffer(
      reinterpret_cast<char*>(const_cast<uint8_t*>((inputBuffer))), BYTES(32));
  uint8_t bitpack_pos = 0;
  uint32_t* duckOutputBuffer = reinterpret_cast<uint32_t*>(outputBuffer);

  duckdb::ParquetDecodeUtils::BitUnpack<uint32_t>(
      duckInputBuffer, bitpack_pos, duckOutputBuffer, kNumValues, 23);
}

BENCHMARK(duckdb_24_32) {
  duckdb::ByteBuffer duckInputBuffer(
      reinterpret_cast<char*>(const_cast<uint8_t*>((inputBuffer))), BYTES(32));
  uint8_t bitpack_pos = 0;
  uint32_t* duckOutputBuffer = reinterpret_cast<uint32_t*>(outputBuffer);

  duckdb::ParquetDecodeUtils::BitUnpack<uint32_t>(
      duckInputBuffer, bitpack_pos, duckOutputBuffer, kNumValues, 24);
}

BENCHMARK(duckdb_25_32) {
  duckdb::ByteBuffer duckInputBuffer(
      reinterpret_cast<char*>(const_cast<uint8_t*>((inputBuffer))), BYTES(32));
  uint8_t bitpack_pos = 0;
  uint32_t* duckOutputBuffer = reinterpret_cast<uint32_t*>(outputBuffer);
  duckdb::ParquetDecodeUtils::BitUnpack<uint32_t>(
      duckInputBuffer, bitpack_pos, duckOutputBuffer, kNumValues, 25);
}

BENCHMARK(duckdb_26_32) {
  duckdb::ByteBuffer duckInputBuffer(
      reinterpret_cast<char*>(const_cast<uint8_t*>((inputBuffer))), BYTES(32));
  uint8_t bitpack_pos = 0;
  uint32_t* duckOutputBuffer = reinterpret_cast<uint32_t*>(outputBuffer);
  duckdb::ParquetDecodeUtils::BitUnpack<uint32_t>(
      duckInputBuffer, bitpack_pos, duckOutputBuffer, kNumValues, 26);
}

BENCHMARK(duckdb_27_32) {
  duckdb::ByteBuffer duckInputBuffer(
      reinterpret_cast<char*>(const_cast<uint8_t*>((inputBuffer))), BYTES(32));
  uint8_t bitpack_pos = 0;
  uint32_t* duckOutputBuffer = reinterpret_cast<uint32_t*>(outputBuffer);
  duckdb::ParquetDecodeUtils::BitUnpack<uint32_t>(
      duckInputBuffer, bitpack_pos, duckOutputBuffer, kNumValues, 27);
}

BENCHMARK(duckdb_28_32) {
  duckdb::ByteBuffer duckInputBuffer(
      reinterpret_cast<char*>(const_cast<uint8_t*>((inputBuffer))), BYTES(32));
  uint8_t bitpack_pos = 0;
  uint32_t* duckOutputBuffer = reinterpret_cast<uint32_t*>(outputBuffer);
  duckdb::ParquetDecodeUtils::BitUnpack<uint32_t>(
      duckInputBuffer, bitpack_pos, duckOutputBuffer, kNumValues, 28);
}

BENCHMARK(duckdb_29_32) {
  duckdb::ByteBuffer duckInputBuffer(
      reinterpret_cast<char*>(const_cast<uint8_t*>((inputBuffer))), BYTES(32));
  uint8_t bitpack_pos = 0;
  uint32_t* duckOutputBuffer = reinterpret_cast<uint32_t*>(outputBuffer);
  duckdb::ParquetDecodeUtils::BitUnpack<uint32_t>(
      duckInputBuffer, bitpack_pos, duckOutputBuffer, kNumValues, 29);
}

BENCHMARK(duckdb_30_32) {
  duckdb::ByteBuffer duckInputBuffer(
      reinterpret_cast<char*>(const_cast<uint8_t*>((inputBuffer))), BYTES(32));
  uint8_t bitpack_pos = 0;
  uint32_t* duckOutputBuffer = reinterpret_cast<uint32_t*>(outputBuffer);
  duckdb::ParquetDecodeUtils::BitUnpack<uint32_t>(
      duckInputBuffer, bitpack_pos, duckOutputBuffer, kNumValues, 30);
}

BENCHMARK(duckdb_31_32) {
  duckdb::ByteBuffer duckInputBuffer(
      reinterpret_cast<char*>(const_cast<uint8_t*>((inputBuffer))), BYTES(32));
  uint8_t bitpack_pos = 0;
  uint32_t* duckOutputBuffer = reinterpret_cast<uint32_t*>(outputBuffer);

  duckdb::ParquetDecodeUtils::BitUnpack<uint32_t>(
      duckInputBuffer, bitpack_pos, duckOutputBuffer, kNumValues, 31);
}

BENCHMARK(duckdb_32_32) {
  duckdb::ByteBuffer duckInputBuffer(
      reinterpret_cast<char*>(const_cast<uint8_t*>((inputBuffer))), BYTES(32));
  uint8_t bitpack_pos = 0;
  uint32_t* duckOutputBuffer = reinterpret_cast<uint32_t*>(outputBuffer);

  duckdb::ParquetDecodeUtils::BitUnpack<uint32_t>(
      duckInputBuffer, bitpack_pos, duckOutputBuffer, kNumValues, 32);
}

BENCHMARK(arrow_1_8) {
  arrow::bit_util::BitReader bitReader(inputBuffer, BYTES(8));
  bitReader.GetBatch(1, outputBuffer, kNumValues);
}

BENCHMARK(arrow_2_8) {
  arrow::bit_util::BitReader bitReader(inputBuffer, BYTES(8));
  bitReader.GetBatch(2, outputBuffer, kNumValues);
}

BENCHMARK(arrow_3_8) {
  arrow::bit_util::BitReader bitReader(inputBuffer, BYTES(8));
  bitReader.GetBatch(3, outputBuffer, kNumValues);
}

BENCHMARK(arrow_4_8) {
  arrow::bit_util::BitReader bitReader(inputBuffer, BYTES(8));
  bitReader.GetBatch(4, outputBuffer, kNumValues);
}

BENCHMARK(arrow_5_8) {
  arrow::bit_util::BitReader bitReader(inputBuffer, BYTES(8));
  bitReader.GetBatch(5, outputBuffer, kNumValues);
}

BENCHMARK(arrow_6_8) {
  arrow::bit_util::BitReader bitReader(inputBuffer, BYTES(8));
  bitReader.GetBatch(6, outputBuffer, kNumValues);
}

BENCHMARK(arrow_7_8) {
  arrow::bit_util::BitReader bitReader(inputBuffer, BYTES(8));
  bitReader.GetBatch(7, outputBuffer, kNumValues);
}

BENCHMARK(arrow_8_8) {
  arrow::bit_util::BitReader bitReader(inputBuffer, BYTES(8));
  bitReader.GetBatch(8, outputBuffer, kNumValues);
}

BENCHMARK(arrow_1_16) {
  arrow::bit_util::BitReader bitReader(inputBuffer, BYTES(16));
  bitReader.GetBatch(1, outputBuffer, kNumValues);
}

BENCHMARK(arrow_2_16) {
  arrow::bit_util::BitReader bitReader(inputBuffer, BYTES(16));
  bitReader.GetBatch(2, outputBuffer, kNumValues);
}

BENCHMARK(arrow_3_16) {
  arrow::bit_util::BitReader bitReader(inputBuffer, BYTES(16));
  bitReader.GetBatch(3, outputBuffer, kNumValues);
}

BENCHMARK(arrow_4_16) {
  arrow::bit_util::BitReader bitReader(inputBuffer, BYTES(16));
  bitReader.GetBatch(4, outputBuffer, kNumValues);
}

BENCHMARK(arrow_5_16) {
  arrow::bit_util::BitReader bitReader(inputBuffer, BYTES(16));
  bitReader.GetBatch(5, outputBuffer, kNumValues);
}

BENCHMARK(arrow_6_16) {
  arrow::bit_util::BitReader bitReader(inputBuffer, BYTES(16));
  bitReader.GetBatch(6, outputBuffer, kNumValues);
}

BENCHMARK(arrow_7_16) {
  arrow::bit_util::BitReader bitReader(inputBuffer, BYTES(16));
  bitReader.GetBatch(7, outputBuffer, kNumValues);
}

BENCHMARK(arrow_8_16) {
  arrow::bit_util::BitReader bitReader(inputBuffer, BYTES(16));
  bitReader.GetBatch(8, outputBuffer, kNumValues);
}

BENCHMARK(arrow_9_16) {
  arrow::bit_util::BitReader bitReader(inputBuffer, BYTES(16));
  bitReader.GetBatch(9, outputBuffer, kNumValues);
}

BENCHMARK(arrow_10_16) {
  arrow::bit_util::BitReader bitReader(inputBuffer, BYTES(16));
  bitReader.GetBatch(10, outputBuffer, kNumValues);
}

BENCHMARK(arrow_11_16) {
  arrow::bit_util::BitReader bitReader(inputBuffer, BYTES(16));
  bitReader.GetBatch(11, outputBuffer, kNumValues);
}

BENCHMARK(arrow_12_16) {
  arrow::bit_util::BitReader bitReader(inputBuffer, BYTES(16));
  bitReader.GetBatch(12, outputBuffer, kNumValues);
}

BENCHMARK(arrow_13_16) {
  arrow::bit_util::BitReader bitReader(inputBuffer, BYTES(16));
  bitReader.GetBatch(13, outputBuffer, kNumValues);
}

BENCHMARK(arrow_14_16) {
  arrow::bit_util::BitReader bitReader(inputBuffer, BYTES(16));
  bitReader.GetBatch(14, outputBuffer, kNumValues);
}

BENCHMARK(arrow_15_16) {
  arrow::bit_util::BitReader bitReader(inputBuffer, BYTES(16));
  bitReader.GetBatch(15, outputBuffer, kNumValues);
}

BENCHMARK(arrow_16_16) {
  arrow::bit_util::BitReader bitReader(inputBuffer, BYTES(16));
  bitReader.GetBatch(16, outputBuffer, kNumValues);
}

BENCHMARK(arrow_1_32) {
  arrow::bit_util::BitReader bitReader(inputBuffer, BYTES(32));
  bitReader.GetBatch(1, outputBuffer, kNumValues);
}

BENCHMARK(arrow_2_32) {
  arrow::bit_util::BitReader bitReader(inputBuffer, BYTES(32));
  bitReader.GetBatch(2, outputBuffer, kNumValues);
}

BENCHMARK(arrow_3_32) {
  arrow::bit_util::BitReader bitReader(inputBuffer, BYTES(32));
  bitReader.GetBatch(3, outputBuffer, kNumValues);
}

BENCHMARK(arrow_4_32) {
  arrow::bit_util::BitReader bitReader(inputBuffer, BYTES(32));
  bitReader.GetBatch(4, outputBuffer, kNumValues);
}

BENCHMARK(arrow_5_32) {
  arrow::bit_util::BitReader bitReader(inputBuffer, BYTES(32));
  bitReader.GetBatch(5, outputBuffer, kNumValues);
}

BENCHMARK(arrow_6_32) {
  arrow::bit_util::BitReader bitReader(inputBuffer, BYTES(32));
  bitReader.GetBatch(6, outputBuffer, kNumValues);
}

BENCHMARK(arrow_7_32) {
  arrow::bit_util::BitReader bitReader(inputBuffer, BYTES(32));
  bitReader.GetBatch(7, outputBuffer, kNumValues);
}

BENCHMARK(arrow_8_32) {
  arrow::bit_util::BitReader bitReader(inputBuffer, BYTES(32));
  bitReader.GetBatch(8, outputBuffer, kNumValues);
}

BENCHMARK(arrow_9_32) {
  arrow::bit_util::BitReader bitReader(inputBuffer, BYTES(32));
  bitReader.GetBatch(9, outputBuffer, kNumValues);
}

BENCHMARK(arrow_10_32) {
  arrow::bit_util::BitReader bitReader(inputBuffer, BYTES(32));
  bitReader.GetBatch(10, outputBuffer, kNumValues);
}

BENCHMARK(arrow_11_32) {
  arrow::bit_util::BitReader bitReader(inputBuffer, BYTES(32));
  bitReader.GetBatch(11, outputBuffer, kNumValues);
}

BENCHMARK(arrow_12_32) {
  arrow::bit_util::BitReader bitReader(inputBuffer, BYTES(32));
  bitReader.GetBatch(12, outputBuffer, kNumValues);
}

BENCHMARK(arrow_13_32) {
  arrow::bit_util::BitReader bitReader(inputBuffer, BYTES(32));
  bitReader.GetBatch(13, outputBuffer, kNumValues);
}

BENCHMARK(arrow_14_32) {
  arrow::bit_util::BitReader bitReader(inputBuffer, BYTES(32));
  bitReader.GetBatch(14, outputBuffer, kNumValues);
}

BENCHMARK(arrow_15_32) {
  arrow::bit_util::BitReader bitReader(inputBuffer, BYTES(32));
  bitReader.GetBatch(15, outputBuffer, kNumValues);
}

BENCHMARK(arrow_16_32) {
  arrow::bit_util::BitReader bitReader(inputBuffer, BYTES(32));
  bitReader.GetBatch(16, outputBuffer, kNumValues);
}

BENCHMARK(arrow_17_32) {
  arrow::bit_util::BitReader bitReader(inputBuffer, BYTES(32));
  bitReader.GetBatch(17, outputBuffer, kNumValues);
}

BENCHMARK(arrow_18_32) {
  arrow::bit_util::BitReader bitReader(inputBuffer, BYTES(32));
  bitReader.GetBatch(18, outputBuffer, kNumValues);
}

BENCHMARK(arrow_19_32) {
  arrow::bit_util::BitReader bitReader(inputBuffer, BYTES(32));
  bitReader.GetBatch(19, outputBuffer, kNumValues);
}

BENCHMARK(arrow_20_32) {
  arrow::bit_util::BitReader bitReader(inputBuffer, BYTES(32));
  bitReader.GetBatch(20, outputBuffer, kNumValues);
}

BENCHMARK(arrow_21_32) {
  arrow::bit_util::BitReader bitReader(inputBuffer, BYTES(32));
  bitReader.GetBatch(21, outputBuffer, kNumValues);
}

BENCHMARK(arrow_22_32) {
  arrow::bit_util::BitReader bitReader(inputBuffer, BYTES(32));
  bitReader.GetBatch(22, outputBuffer, kNumValues);
}

BENCHMARK(arrow_23_32) {
  arrow::bit_util::BitReader bitReader(inputBuffer, BYTES(32));
  bitReader.GetBatch(23, outputBuffer, kNumValues);
}

BENCHMARK(arrow_24_32) {
  arrow::bit_util::BitReader bitReader(inputBuffer, BYTES(32));
  bitReader.GetBatch(24, outputBuffer, kNumValues);
}

BENCHMARK(arrow_25_32) {
  arrow::bit_util::BitReader bitReader(inputBuffer, BYTES(32));
  bitReader.GetBatch(25, outputBuffer, kNumValues);
}

BENCHMARK(arrow_26_32) {
  arrow::bit_util::BitReader bitReader(inputBuffer, BYTES(32));
  bitReader.GetBatch(26, outputBuffer, kNumValues);
}

BENCHMARK(arrow_27_32) {
  arrow::bit_util::BitReader bitReader(inputBuffer, BYTES(32));
  bitReader.GetBatch(27, outputBuffer, kNumValues);
}

BENCHMARK(arrow_28_32) {
  arrow::bit_util::BitReader bitReader(inputBuffer, BYTES(32));
  bitReader.GetBatch(28, outputBuffer, kNumValues);
}

BENCHMARK(arrow_29_32) {
  arrow::bit_util::BitReader bitReader(inputBuffer, BYTES(32));
  bitReader.GetBatch(29, outputBuffer, kNumValues);
}

BENCHMARK(arrow_30_32) {
  arrow::bit_util::BitReader bitReader(inputBuffer, BYTES(32));
  bitReader.GetBatch(30, outputBuffer, kNumValues);
}

BENCHMARK(arrow_31_32) {
  arrow::bit_util::BitReader bitReader(inputBuffer, BYTES(32));
  bitReader.GetBatch(31, outputBuffer, kNumValues);
}

BENCHMARK(arrow_32_32) {
  arrow::bit_util::BitReader bitReader(inputBuffer, BYTES(32));
  bitReader.GetBatch(32, outputBuffer, kNumValues);
}

BENCHMARK(lemirefastpforlib_1_8) {
  uint64_t numBatches = kNumValues / 32;
  for (auto i = 0; i < numBatches; i++) {
    // Read 4 bytes and unpack 32 values
    duckdb_fastpforlib::fastunpack(
        inputBuffer + i * 4, outputBuffer + i * 32, 1);
  }
}

BENCHMARK(lemirefastpforlib_2_8) {
  uint64_t numBatches = kNumValues / 32;
  for (auto i = 0; i < numBatches; i++) {
    // Read 8 bytes and unpack 32 values
    duckdb_fastpforlib::fastunpack(
        inputBuffer + i * 8, outputBuffer + i * 32, 2);
  }
}

BENCHMARK(lemirefastpforlib_3_8) {
  uint64_t numBatches = kNumValues / 32;
  for (auto i = 0; i < numBatches; i++) {
    // Read 12 bytes and unpack 32 values
    duckdb_fastpforlib::fastunpack(
        inputBuffer + i * 12, outputBuffer + i * 32, 3);
  }
}

BENCHMARK(lemirefastpforlib_4_8) {
  uint64_t numBatches = kNumValues / 32;
  for (auto i = 0; i < numBatches; i++) {
    // Read 16 bytes and unpack 32 values
    duckdb_fastpforlib::fastunpack(
        inputBuffer + i * 16, outputBuffer + i * 32, 4);
  }
}

BENCHMARK(lemirefastpforlib_5_8) {
  uint64_t numBatches = kNumValues / 32;
  for (auto i = 0; i < numBatches; i++) {
    // Read 20 bytes and unpack 32 values
    duckdb_fastpforlib::fastunpack(
        inputBuffer + i * 20, outputBuffer + i * 32, 5);
  }
}

BENCHMARK(lemirefastpforlib_6_8) {
  uint64_t numBatches = kNumValues / 32;
  for (auto i = 0; i < numBatches; i++) {
    // Read 24 bytes and unpack 32 values
    duckdb_fastpforlib::fastunpack(
        inputBuffer + i * 24, outputBuffer + i * 32, 6);
  }
}

BENCHMARK(lemirefastpforlib_7_8) {
  uint64_t numBatches = kNumValues / 32;
  for (auto i = 0; i < numBatches; i++) {
    // Read 28 bytes and unpack 32 values
    duckdb_fastpforlib::fastunpack(
        inputBuffer + i * 28, outputBuffer + i * 32, 7);
  }
}

BENCHMARK(lemirefastpforlib_8_8) {
  uint64_t numBatches = kNumValues / 32;
  for (auto i = 0; i < numBatches; i++) {
    // Read 32 bytes and unpack 32 values
    duckdb_fastpforlib::fastunpack(
        inputBuffer + i * 32, outputBuffer + i * 32, 8);
  }
}

BENCHMARK(lemirefastpforlib_1_16) {
  uint64_t numBatches = kNumValues / 32;
  for (auto i = 0; i < numBatches; i++) {
    // Read 4 bytes and unpack 32 values
    duckdb_fastpforlib::fastunpack(
        reinterpret_cast<const uint16_t*>(inputBuffer + i * 4),
        reinterpret_cast<uint16_t*>(outputBuffer) + i * 32,
        1);
  }
}

BENCHMARK(lemirefastpforlib_2_16) {
  uint64_t numBatches = kNumValues / 32;
  for (auto i = 0; i < numBatches; i++) {
    // Read 8 bytes and unpack 32 values
    duckdb_fastpforlib::fastunpack(
        reinterpret_cast<const uint16_t*>(inputBuffer + i * 8),
        reinterpret_cast<uint16_t*>(outputBuffer) + i * 32,
        2);
  }
}

BENCHMARK(lemirefastpforlib_3_16) {
  uint64_t numBatches = kNumValues / 32;
  for (auto i = 0; i < numBatches; i++) {
    // Read 12 bytes and unpack 32 values
    duckdb_fastpforlib::fastunpack(
        reinterpret_cast<const uint16_t*>(inputBuffer + i * 12),
        reinterpret_cast<uint16_t*>(outputBuffer) + i * 32,
        3);
  }
}

BENCHMARK(lemirefastpforlib_4_16) {
  uint64_t numBatches = kNumValues / 32;
  for (auto i = 0; i < numBatches; i++) {
    // Read 16 bytes and unpack 32 values
    duckdb_fastpforlib::fastunpack(
        reinterpret_cast<const uint16_t*>(inputBuffer + i * 16),
        reinterpret_cast<uint16_t*>(outputBuffer) + i * 32,
        4);
  }
}

BENCHMARK(lemirefastpforlib_5_16) {
  uint64_t numBatches = kNumValues / 32;
  for (auto i = 0; i < numBatches; i++) {
    // Read 20 bytes and unpack 32 values
    duckdb_fastpforlib::fastunpack(
        reinterpret_cast<const uint16_t*>(inputBuffer + i * 20),
        reinterpret_cast<uint16_t*>(outputBuffer) + i * 32,
        5);
  }
}

BENCHMARK(lemirefastpforlib_6_16) {
  uint64_t numBatches = kNumValues / 16;
  for (auto i = 0; i < numBatches; i++) {
    // Read 24 bytes and unpack 32 values
    duckdb_fastpforlib::fastunpack(
        reinterpret_cast<const uint16_t*>(inputBuffer + i * 24),
        reinterpret_cast<uint16_t*>(outputBuffer) + i * 32,
        6);
  }
}

BENCHMARK(lemirefastpforlib_7_16) {
  uint64_t numBatches = kNumValues / 32;
  for (auto i = 0; i < numBatches; i++) {
    // Read 28 bytes and unpack 32 values
    duckdb_fastpforlib::fastunpack(
        reinterpret_cast<const uint16_t*>(inputBuffer + i * 28),
        reinterpret_cast<uint16_t*>(outputBuffer) + i * 32,
        7);
  }
}

BENCHMARK(lemirefastpforlib_8_16) {
  uint64_t numBatches = kNumValues / 32;
  for (auto i = 0; i < numBatches; i++) {
    // Read 32 bytes and unpack 32 values
    duckdb_fastpforlib::fastunpack(
        reinterpret_cast<const uint16_t*>(inputBuffer + i * 32),
        reinterpret_cast<uint16_t*>(outputBuffer) + i * 32,
        8);
  }
}

BENCHMARK(lemirefastpforlib_9_16) {
  uint64_t numBatches = kNumValues / 32;
  for (auto i = 0; i < numBatches; i++) {
    // Read 36 bytes and unpack 32 values
    duckdb_fastpforlib::fastunpack(
        reinterpret_cast<const uint16_t*>(inputBuffer + i * 36),
        reinterpret_cast<uint16_t*>(outputBuffer) + i * 32,
        9);
  }
}

BENCHMARK(lemirefastpforlib_10_16) {
  uint64_t numBatches = kNumValues / 32;
  for (auto i = 0; i < numBatches; i++) {
    // Read 40 bytes and unpack 32 values
    duckdb_fastpforlib::fastunpack(
        reinterpret_cast<const uint16_t*>(inputBuffer + i * 40),
        reinterpret_cast<uint16_t*>(outputBuffer) + i * 32,
        10);
  }
}

BENCHMARK(lemirefastpforlib_11_16) {
  uint64_t numBatches = kNumValues / 32;
  for (auto i = 0; i < numBatches; i++) {
    // Read 44 bytes and unpack 32 values
    duckdb_fastpforlib::fastunpack(
        reinterpret_cast<const uint16_t*>(inputBuffer + i * 44),
        reinterpret_cast<uint16_t*>(outputBuffer) + i * 32,
        11);
  }
}

BENCHMARK(lemirefastpforlib_12_16) {
  uint64_t numBatches = kNumValues / 32;
  for (auto i = 0; i < numBatches; i++) {
    // Read 48 bytes and unpack 32 values
    duckdb_fastpforlib::fastunpack(
        reinterpret_cast<const uint16_t*>(inputBuffer + i * 48),
        reinterpret_cast<uint16_t*>(outputBuffer) + i * 32,
        12);
  }
}

BENCHMARK(lemirefastpforlib_13_16) {
  uint64_t numBatches = kNumValues / 32;
  for (auto i = 0; i < numBatches; i++) {
    // Read 52 bytes and unpack 32 values
    duckdb_fastpforlib::fastunpack(
        reinterpret_cast<const uint16_t*>(inputBuffer + i * 52),
        reinterpret_cast<uint16_t*>(outputBuffer) + i * 32,
        13);
  }
}

BENCHMARK(lemirefastpforlib_14_16) {
  uint64_t numBatches = kNumValues / 32;
  for (auto i = 0; i < numBatches; i++) {
    // Read 56 bytes and unpack 32 values
    duckdb_fastpforlib::fastunpack(
        reinterpret_cast<const uint16_t*>(inputBuffer + i * 56),
        reinterpret_cast<uint16_t*>(outputBuffer) + i * 32,
        14);
  }
}

BENCHMARK(lemirefastpforlib_15_16) {
  uint64_t numBatches = kNumValues / 32;
  for (auto i = 0; i < numBatches; i++) {
    // Read 60 bytes and unpack 32 values
    duckdb_fastpforlib::fastunpack(
        reinterpret_cast<const uint16_t*>(inputBuffer + i * 60),
        reinterpret_cast<uint16_t*>(outputBuffer) + i * 32,
        15);
  }
}

BENCHMARK(lemirefastpforlib_16_16) {
  uint64_t numBatches = kNumValues / 32;
  for (auto i = 0; i < numBatches; i++) {
    // Read 64 bytes and unpack 32 values
    duckdb_fastpforlib::fastunpack(
        reinterpret_cast<const uint16_t*>(inputBuffer + i * 64),
        reinterpret_cast<uint16_t*>(outputBuffer) + i * 32,
        16);
  }
}

BENCHMARK(lemirefastpforlib_1_32) {
  uint64_t numBatches = kNumValues / 32;
  for (auto i = 0; i < numBatches; i++) {
    // Read 4 bytes and unpack 32 values
    duckdb_fastpforlib::fastunpack(
        reinterpret_cast<const uint32_t*>(inputBuffer + i * 4),
        reinterpret_cast<uint32_t*>(outputBuffer) + i * 32,
        1);
  }
}

BENCHMARK(lemirefastpforlib_2_32) {
  uint64_t numBatches = kNumValues / 32;
  for (auto i = 0; i < numBatches; i++) {
    // Read 8 bytes and unpack 32 values
    duckdb_fastpforlib::fastunpack(
        reinterpret_cast<const uint32_t*>(inputBuffer + i * 8),
        reinterpret_cast<uint32_t*>(outputBuffer) + i * 32,
        2);
  }
}

BENCHMARK(lemirefastpforlib_3_32) {
  uint64_t numBatches = kNumValues / 32;
  for (auto i = 0; i < numBatches; i++) {
    // Read 12 bytes and unpack 32 values
    duckdb_fastpforlib::fastunpack(
        reinterpret_cast<const uint32_t*>(inputBuffer + i * 12),
        reinterpret_cast<uint32_t*>(outputBuffer) + i * 32,
        3);
  }
}

BENCHMARK(lemirefastpforlib_4_32) {
  uint64_t numBatches = kNumValues / 32;
  for (auto i = 0; i < numBatches; i++) {
    // Read 16 bytes and unpack 32 values
    duckdb_fastpforlib::fastunpack(
        reinterpret_cast<const uint32_t*>(inputBuffer + i * 16),
        reinterpret_cast<uint32_t*>(outputBuffer) + i * 32,
        4);
  }
}

BENCHMARK(lemirefastpforlib_5_32) {
  uint64_t numBatches = kNumValues / 32;
  for (auto i = 0; i < numBatches; i++) {
    // Read 20 bytes and unpack 32 values
    duckdb_fastpforlib::fastunpack(
        reinterpret_cast<const uint32_t*>(inputBuffer + i * 20),
        reinterpret_cast<uint32_t*>(outputBuffer) + i * 32,
        5);
  }
}

BENCHMARK(lemirefastpforlib_6_32) {
  uint64_t numBatches = kNumValues / 32;
  for (auto i = 0; i < numBatches; i++) {
    // Read 24 bytes and unpack 32 values
    duckdb_fastpforlib::fastunpack(
        reinterpret_cast<const uint32_t*>(inputBuffer + i * 24),
        reinterpret_cast<uint32_t*>(outputBuffer) + i * 32,
        6);
  }
}

BENCHMARK(lemirefastpforlib_7_32) {
  uint64_t numBatches = kNumValues / 32;
  for (auto i = 0; i < numBatches; i++) {
    // Read 28 bytes and unpack 32 values
    duckdb_fastpforlib::fastunpack(
        reinterpret_cast<const uint32_t*>(inputBuffer + i * 28),
        reinterpret_cast<uint32_t*>(outputBuffer) + i * 32,
        7);
  }
}

BENCHMARK(lemirefastpforlib_8_32) {
  uint64_t numBatches = kNumValues / 32;
  for (auto i = 0; i < numBatches; i++) {
    // Read 32 bytes and unpack 32 values
    duckdb_fastpforlib::fastunpack(
        reinterpret_cast<const uint32_t*>(inputBuffer + i * 32),
        reinterpret_cast<uint32_t*>(outputBuffer) + i * 32,
        8);
  }
}

BENCHMARK(lemirefastpforlib_9_32) {
  uint64_t numBatches = kNumValues / 32;
  for (auto i = 0; i < numBatches; i++) {
    // Read 36 bytes and unpack 32 values
    duckdb_fastpforlib::fastunpack(
        reinterpret_cast<const uint32_t*>(inputBuffer + i * 36),
        reinterpret_cast<uint32_t*>(outputBuffer) + i * 32,
        9);
  }
}

BENCHMARK(lemirefastpforlib_10_32) {
  uint64_t numBatches = kNumValues / 32;
  for (auto i = 0; i < numBatches; i++) {
    // Read 40 bytes and unpack 32 values
    duckdb_fastpforlib::fastunpack(
        reinterpret_cast<const uint32_t*>(inputBuffer + i * 40),
        reinterpret_cast<uint32_t*>(outputBuffer) + i * 32,
        10);
  }
}

BENCHMARK(lemirefastpforlib_11_32) {
  uint64_t numBatches = kNumValues / 32;
  for (auto i = 0; i < numBatches; i++) {
    // Read 44 bytes and unpack 32 values
    duckdb_fastpforlib::fastunpack(
        reinterpret_cast<const uint32_t*>(inputBuffer + i * 44),
        reinterpret_cast<uint32_t*>(outputBuffer) + i * 32,
        11);
  }
}

BENCHMARK(lemirefastpforlib_12_32) {
  uint64_t numBatches = kNumValues / 32;
  for (auto i = 0; i < numBatches; i++) {
    // Read 48 bytes and unpack 32 values
    duckdb_fastpforlib::fastunpack(
        reinterpret_cast<const uint32_t*>(inputBuffer + i * 48),
        reinterpret_cast<uint32_t*>(outputBuffer) + i * 32,
        12);
  }
}

BENCHMARK(lemirefastpforlib_13_32) {
  uint64_t numBatches = kNumValues / 32;
  for (auto i = 0; i < numBatches; i++) {
    // Read 52 bytes and unpack 32 values
    duckdb_fastpforlib::fastunpack(
        reinterpret_cast<const uint32_t*>(inputBuffer + i * 52),
        reinterpret_cast<uint32_t*>(outputBuffer) + i * 32,
        13);
  }
}

BENCHMARK(lemirefastpforlib_14_32) {
  uint64_t numBatches = kNumValues / 32;
  for (auto i = 0; i < numBatches; i++) {
    // Read 56 bytes and unpack 32 values
    duckdb_fastpforlib::fastunpack(
        reinterpret_cast<const uint32_t*>(inputBuffer + i * 56),
        reinterpret_cast<uint32_t*>(outputBuffer) + i * 32,
        14);
  }
}

BENCHMARK(lemirefastpforlib_15_32) {
  uint64_t numBatches = kNumValues / 32;
  for (auto i = 0; i < numBatches; i++) {
    // Read 60 bytes and unpack 32 values
    duckdb_fastpforlib::fastunpack(
        reinterpret_cast<const uint32_t*>(inputBuffer + i * 60),
        reinterpret_cast<uint32_t*>(outputBuffer) + i * 32,
        15);
  }
}

BENCHMARK(lemirefastpforlib_16_32) {
  uint64_t numBatches = kNumValues / 32;
  for (auto i = 0; i < numBatches; i++) {
    // Read 64 bytes and unpack 32 values
    duckdb_fastpforlib::fastunpack(
        reinterpret_cast<const uint32_t*>(inputBuffer + i * 64),
        reinterpret_cast<uint32_t*>(outputBuffer) + i * 32,
        16);
  }
}

BENCHMARK(lemirefastpforlib_17_32) {
  uint64_t numBatches = kNumValues / 32;
  for (auto i = 0; i < numBatches; i++) {
    // Read 4 bytes and unpack 32 values
    duckdb_fastpforlib::fastunpack(
        reinterpret_cast<const uint32_t*>(inputBuffer + i * 4),
        reinterpret_cast<uint32_t*>(outputBuffer) + i * 32,
        17);
  }
}

BENCHMARK(lemirefastpforlib_18_32) {
  uint64_t numBatches = kNumValues / 32;
  for (auto i = 0; i < numBatches; i++) {
    // Read 8 bytes and unpack 32 values
    duckdb_fastpforlib::fastunpack(
        reinterpret_cast<const uint32_t*>(inputBuffer + i * 8),
        reinterpret_cast<uint32_t*>(outputBuffer) + i * 32,
        18);
  }
}

BENCHMARK(lemirefastpforlib_19_32) {
  uint64_t numBatches = kNumValues / 32;
  for (auto i = 0; i < numBatches; i++) {
    // Read 12 bytes and unpack 32 values
    duckdb_fastpforlib::fastunpack(
        reinterpret_cast<const uint32_t*>(inputBuffer + i * 12),
        reinterpret_cast<uint32_t*>(outputBuffer) + i * 32,
        19);
  }
}

BENCHMARK(lemirefastpforlib_20_32) {
  uint64_t numBatches = kNumValues / 32;
  for (auto i = 0; i < numBatches; i++) {
    // Read 16 bytes and unpack 32 values
    duckdb_fastpforlib::fastunpack(
        reinterpret_cast<const uint32_t*>(inputBuffer + i * 16),
        reinterpret_cast<uint32_t*>(outputBuffer) + i * 32,
        20);
  }
}

BENCHMARK(lemirefastpforlib_21_32) {
  uint64_t numBatches = kNumValues / 32;
  for (auto i = 0; i < numBatches; i++) {
    // Read 20 bytes and unpack 32 values
    duckdb_fastpforlib::fastunpack(
        reinterpret_cast<const uint32_t*>(inputBuffer + i * 20),
        reinterpret_cast<uint32_t*>(outputBuffer) + i * 32,
        21);
  }
}

BENCHMARK(lemirefastpforlib_22_32) {
  uint64_t numBatches = kNumValues / 32;
  for (auto i = 0; i < numBatches; i++) {
    // Read 24 bytes and unpack 32 values
    duckdb_fastpforlib::fastunpack(
        reinterpret_cast<const uint32_t*>(inputBuffer + i * 24),
        reinterpret_cast<uint32_t*>(outputBuffer) + i * 32,
        22);
  }
}

BENCHMARK(lemirefastpforlib_23_32) {
  uint64_t numBatches = kNumValues / 32;
  for (auto i = 0; i < numBatches; i++) {
    // Read 28 bytes and unpack 32 values
    duckdb_fastpforlib::fastunpack(
        reinterpret_cast<const uint32_t*>(inputBuffer + i * 28),
        reinterpret_cast<uint32_t*>(outputBuffer) + i * 32,
        23);
  }
}

BENCHMARK(lemirefastpforlib_24_32) {
  uint64_t numBatches = kNumValues / 32;
  for (auto i = 0; i < numBatches; i++) {
    // Read 32 bytes and unpack 32 values
    duckdb_fastpforlib::fastunpack(
        reinterpret_cast<const uint32_t*>(inputBuffer + i * 32),
        reinterpret_cast<uint32_t*>(outputBuffer) + i * 32,
        24);
  }
}

BENCHMARK(lemirefastpforlib_25_32) {
  uint64_t numBatches = kNumValues / 32;
  for (auto i = 0; i < numBatches; i++) {
    // Read 36 bytes and unpack 32 values
    duckdb_fastpforlib::fastunpack(
        reinterpret_cast<const uint32_t*>(inputBuffer + i * 36),
        reinterpret_cast<uint32_t*>(outputBuffer) + i * 32,
        25);
  }
}

BENCHMARK(lemirefastpforlib_26_32) {
  uint64_t numBatches = kNumValues / 32;
  for (auto i = 0; i < numBatches; i++) {
    // Read 40 bytes and unpack 32 values
    duckdb_fastpforlib::fastunpack(
        reinterpret_cast<const uint32_t*>(inputBuffer + i * 40),
        reinterpret_cast<uint32_t*>(outputBuffer) + i * 32,
        26);
  }
}

BENCHMARK(lemirefastpforlib_27_32) {
  uint64_t numBatches = kNumValues / 32;
  for (auto i = 0; i < numBatches; i++) {
    // Read 44 bytes and unpack 32 values
    duckdb_fastpforlib::fastunpack(
        reinterpret_cast<const uint32_t*>(inputBuffer + i * 44),
        reinterpret_cast<uint32_t*>(outputBuffer) + i * 32,
        27);
  }
}

BENCHMARK(lemirefastpforlib_28_32) {
  uint64_t numBatches = kNumValues / 32;
  for (auto i = 0; i < numBatches; i++) {
    // Read 48 bytes and unpack 32 values
    duckdb_fastpforlib::fastunpack(
        reinterpret_cast<const uint32_t*>(inputBuffer + i * 48),
        reinterpret_cast<uint32_t*>(outputBuffer) + i * 32,
        28);
  }
}

BENCHMARK(lemirefastpforlib_29_32) {
  uint64_t numBatches = kNumValues / 32;
  for (auto i = 0; i < numBatches; i++) {
    // Read 52 bytes and unpack 32 values
    duckdb_fastpforlib::fastunpack(
        reinterpret_cast<const uint32_t*>(inputBuffer + i * 52),
        reinterpret_cast<uint32_t*>(outputBuffer) + i * 32,
        29);
  }
}

BENCHMARK(lemirefastpforlib_30_32) {
  uint64_t numBatches = kNumValues / 32;
  for (auto i = 0; i < numBatches; i++) {
    // Read 56 bytes and unpack 32 values
    duckdb_fastpforlib::fastunpack(
        reinterpret_cast<const uint32_t*>(inputBuffer + i * 56),
        reinterpret_cast<uint32_t*>(outputBuffer) + i * 32,
        30);
  }
}

BENCHMARK(lemirefastpforlib_31_32) {
  uint64_t numBatches = kNumValues / 32;
  for (auto i = 0; i < numBatches; i++) {
    // Read 60 bytes and unpack 32 values
    duckdb_fastpforlib::fastunpack(
        reinterpret_cast<const uint32_t*>(inputBuffer + i * 60),
        reinterpret_cast<uint32_t*>(outputBuffer) + i * 32,
        31);
  }
}

BENCHMARK(lemirefastpforlib_32_32) {
  uint64_t numBatches = kNumValues / 32;
  for (auto i = 0; i < numBatches; i++) {
    // Read 64 bytes and unpack 32 values
    duckdb_fastpforlib::fastunpack(
        reinterpret_cast<const uint32_t*>(inputBuffer + i * 64),
        reinterpret_cast<uint32_t*>(outputBuffer) + i * 32,
        32);
  }
}

BENCHMARK(lemirebmi2_1_32) {
  const uint8_t* inputIter = inputBuffer;
  uint32_t* outputIter = reinterpret_cast<uint32_t*>(outputBuffer);

  bmiunpack32(inputIter, kNumValues, 1, outputIter);
}

BENCHMARK(lemirebmi2_2_32) {
  const uint8_t* inputIter = inputBuffer;
  uint32_t* outputIter = reinterpret_cast<uint32_t*>(outputBuffer);

  bmiunpack32(inputIter, kNumValues, 2, outputIter);
}

BENCHMARK(lemirebmi2_3_32) {
  const uint8_t* inputIter = inputBuffer;
  uint32_t* outputIter = reinterpret_cast<uint32_t*>(outputBuffer);

  bmiunpack32(inputIter, kNumValues, 3, outputIter);
}

BENCHMARK(lemirebmi2_4_32) {
  const uint8_t* inputIter = inputBuffer;
  uint32_t* outputIter = reinterpret_cast<uint32_t*>(outputBuffer);

  bmiunpack32(inputIter, kNumValues, 4, outputIter);
}

BENCHMARK(lemirebmi2_5_32) {
  const uint8_t* inputIter = inputBuffer;
  uint32_t* outputIter = reinterpret_cast<uint32_t*>(outputBuffer);

  bmiunpack32(inputIter, kNumValues, 5, outputIter);
}

BENCHMARK(lemirebmi2_6_32) {
  const uint8_t* inputIter = inputBuffer;
  uint32_t* outputIter = reinterpret_cast<uint32_t*>(outputBuffer);

  bmiunpack32(inputIter, kNumValues, 6, outputIter);
}

BENCHMARK(lemirebmi2_7_32) {
  const uint8_t* inputIter = inputBuffer;
  uint32_t* outputIter = reinterpret_cast<uint32_t*>(outputBuffer);

  bmiunpack32(inputIter, kNumValues, 7, outputIter);
}

BENCHMARK(lemirebmi2_8_32) {
  const uint8_t* inputIter = inputBuffer;
  uint32_t* outputIter = reinterpret_cast<uint32_t*>(outputBuffer);

  bmiunpack32(inputIter, kNumValues, 8, outputIter);
}

BENCHMARK(lemirebmi2_9_32) {
  const uint8_t* inputIter = inputBuffer;
  uint32_t* outputIter = reinterpret_cast<uint32_t*>(outputBuffer);

  bmiunpack32(inputIter, kNumValues, 9, outputIter);
}

BENCHMARK(lemirebmi2_10_32) {
  const uint8_t* inputIter = inputBuffer;
  uint32_t* outputIter = reinterpret_cast<uint32_t*>(outputBuffer);

  bmiunpack32(inputIter, kNumValues, 10, outputIter);
}

BENCHMARK(lemirebmi2_11_32) {
  const uint8_t* inputIter = inputBuffer;
  uint32_t* outputIter = reinterpret_cast<uint32_t*>(outputBuffer);

  bmiunpack32(inputIter, kNumValues, 12, outputIter);
}

BENCHMARK(lemirebmi2_12_32) {
  const uint8_t* inputIter = inputBuffer;
  uint32_t* outputIter = reinterpret_cast<uint32_t*>(outputBuffer);

  bmiunpack32(inputIter, kNumValues, 12, outputIter);
}

BENCHMARK(lemirebmi2_13_32) {
  const uint8_t* inputIter = inputBuffer;
  uint32_t* outputIter = reinterpret_cast<uint32_t*>(outputBuffer);

  bmiunpack32(inputIter, kNumValues, 13, outputIter);
}

BENCHMARK(lemirebmi2_14_32) {
  const uint8_t* inputIter = inputBuffer;
  uint32_t* outputIter = reinterpret_cast<uint32_t*>(outputBuffer);

  bmiunpack32(inputIter, kNumValues, 14, outputIter);
}

BENCHMARK(lemirebmi2_15_32) {
  const uint8_t* inputIter = inputBuffer;
  uint32_t* outputIter = reinterpret_cast<uint32_t*>(outputBuffer);

  bmiunpack32(inputIter, kNumValues, 15, outputIter);
}

BENCHMARK(lemirebmi2_16_32) {
  const uint8_t* inputIter = inputBuffer;
  uint32_t* outputIter = reinterpret_cast<uint32_t*>(outputBuffer);

  bmiunpack32(inputIter, kNumValues, 16, outputIter);
}

BENCHMARK(lemirebmi2_17_32) {
  const uint8_t* inputIter = inputBuffer;
  uint32_t* outputIter = reinterpret_cast<uint32_t*>(outputBuffer);

  bmiunpack32(inputIter, kNumValues, 17, outputIter);
}

BENCHMARK(lemirebmi2_18_32) {
  const uint8_t* inputIter = inputBuffer;
  uint32_t* outputIter = reinterpret_cast<uint32_t*>(outputBuffer);

  bmiunpack32(inputIter, kNumValues, 18, outputIter);
}

BENCHMARK(lemirebmi2_19_32) {
  const uint8_t* inputIter = inputBuffer;
  uint32_t* outputIter = reinterpret_cast<uint32_t*>(outputBuffer);

  bmiunpack32(inputIter, kNumValues, 19, outputIter);
}

BENCHMARK(lemirebmi2_20_32) {
  const uint8_t* inputIter = inputBuffer;
  uint32_t* outputIter = reinterpret_cast<uint32_t*>(outputBuffer);

  bmiunpack32(inputIter, kNumValues, 20, outputIter);
}

BENCHMARK(lemirebmi2_21_32) {
  const uint8_t* inputIter = inputBuffer;
  uint32_t* outputIter = reinterpret_cast<uint32_t*>(outputBuffer);

  bmiunpack32(inputIter, kNumValues, 21, outputIter);
}

BENCHMARK(lemirebmi2_22_32) {
  const uint8_t* inputIter = inputBuffer;
  uint32_t* outputIter = reinterpret_cast<uint32_t*>(outputBuffer);

  bmiunpack32(inputIter, kNumValues, 22, outputIter);
}

BENCHMARK(lemirebmi2_23_32) {
  const uint8_t* inputIter = inputBuffer;
  uint32_t* outputIter = reinterpret_cast<uint32_t*>(outputBuffer);

  bmiunpack32(inputIter, kNumValues, 23, outputIter);
}

BENCHMARK(lemirebmi2_24_32) {
  const uint8_t* inputIter = inputBuffer;
  uint32_t* outputIter = reinterpret_cast<uint32_t*>(outputBuffer);

  bmiunpack32(inputIter, kNumValues, 24, outputIter);
}

BENCHMARK(lemirebmi2_25_32) {
  const uint8_t* inputIter = inputBuffer;
  uint32_t* outputIter = reinterpret_cast<uint32_t*>(outputBuffer);

  bmiunpack32(inputIter, kNumValues, 25, outputIter);
}

BENCHMARK(lemirebmi2_26_32) {
  const uint8_t* inputIter = inputBuffer;
  uint32_t* outputIter = reinterpret_cast<uint32_t*>(outputBuffer);

  bmiunpack32(inputIter, kNumValues, 26, outputIter);
}

BENCHMARK(lemirebmi2_27_32) {
  const uint8_t* inputIter = inputBuffer;
  uint32_t* outputIter = reinterpret_cast<uint32_t*>(outputBuffer);

  bmiunpack32(inputIter, kNumValues, 27, outputIter);
}

BENCHMARK(lemirebmi2_28_32) {
  const uint8_t* inputIter = inputBuffer;
  uint32_t* outputIter = reinterpret_cast<uint32_t*>(outputBuffer);

  bmiunpack32(inputIter, kNumValues, 28, outputIter);
}

BENCHMARK(lemirebmi2_29_32) {
  const uint8_t* inputIter = inputBuffer;
  uint32_t* outputIter = reinterpret_cast<uint32_t*>(outputBuffer);

  bmiunpack32(inputIter, kNumValues, 13, outputIter);
}

BENCHMARK(lemirebmi2_30_32) {
  const uint8_t* inputIter = inputBuffer;
  uint32_t* outputIter = reinterpret_cast<uint32_t*>(outputBuffer);

  bmiunpack32(inputIter, kNumValues, 30, outputIter);
}

BENCHMARK(lemirebmi2_31_32) {
  const uint8_t* inputIter = inputBuffer;
  uint32_t* outputIter = reinterpret_cast<uint32_t*>(outputBuffer);

  bmiunpack32(inputIter, kNumValues, 31, outputIter);
}

BENCHMARK(lemirebmi2_32_32) {
  const uint8_t* inputIter = inputBuffer;
  uint32_t* outputIter = reinterpret_cast<uint32_t*>(outputBuffer);

  bmiunpack32(inputIter, kNumValues, 32, outputIter);
}

BENCHMARK(velox_1_8) {
  const uint8_t* inputIter = inputBuffer;
  uint8_t* outputIter = outputBuffer;
  facebook::velox::dwio::common::unpack(
      1, inputIter, BYTES(1), kNumValues, outputIter);
}

BENCHMARK(velox_2_8) {
  const uint8_t* inputIter = inputBuffer;
  uint8_t* outputIter = outputBuffer;
  facebook::velox::dwio::common::unpack(
      2, inputIter, BYTES(2), kNumValues, outputIter);
}

BENCHMARK(velox_3_8) {
  const uint8_t* inputIter = inputBuffer;
  uint8_t* outputIter = outputBuffer;

  facebook::velox::dwio::common::unpack(
      3, inputIter, BYTES(3), kNumValues, outputIter);
}

BENCHMARK(velox_4_8) {
  const uint8_t* inputIter = inputBuffer;
  uint8_t* outputIter = outputBuffer;
  facebook::velox::dwio::common::unpack(
      4, inputIter, BYTES(4), kNumValues, outputIter);
}

BENCHMARK(velox_5_8) {
  const uint8_t* inputIter = inputBuffer;
  uint8_t* outputIter = outputBuffer;
  facebook::velox::dwio::common::unpack(
      5, inputIter, BYTES(5), kNumValues, outputIter);
}

BENCHMARK(velox_6_8) {
  const uint8_t* inputIter = inputBuffer;
  uint8_t* outputIter = outputBuffer;
  facebook::velox::dwio::common::unpack(
      6, inputIter, BYTES(6), kNumValues, outputIter);
}

BENCHMARK(velox_7_8) {
  const uint8_t* inputIter = inputBuffer;
  uint8_t* outputIter = outputBuffer;
  facebook::velox::dwio::common::unpack(
      7, inputIter, BYTES(7), kNumValues, outputIter);
}

BENCHMARK(velox_8_8) {
  const uint8_t* inputIter = inputBuffer;
  uint8_t* outputIter = outputBuffer;
  facebook::velox::dwio::common::unpack(
      8, inputIter, BYTES(8), kNumValues, outputIter);
}

BENCHMARK(velox_1_16) {
  const uint8_t* inputIter = inputBuffer;
  uint16_t* outputIter = reinterpret_cast<uint16_t*>(outputBuffer);

  facebook::velox::dwio::common::unpack(
      1, inputIter, BYTES(1), kNumValues, outputIter);
}

BENCHMARK(velox_2_16) {
  const uint8_t* inputIter = inputBuffer;
  uint16_t* outputIter = reinterpret_cast<uint16_t*>(outputBuffer);

  facebook::velox::dwio::common::unpack(
      2, inputIter, BYTES(2), kNumValues, outputIter);
}

BENCHMARK(velox_3_16) {
  const uint8_t* inputIter = inputBuffer;
  uint16_t* outputIter = reinterpret_cast<uint16_t*>(outputBuffer);

  facebook::velox::dwio::common::unpack(
      3, inputIter, BYTES(3), kNumValues, outputIter);
}

BENCHMARK(velox_4_16) {
  const uint8_t* inputIter = inputBuffer;
  uint16_t* outputIter = reinterpret_cast<uint16_t*>(outputBuffer);

  facebook::velox::dwio::common::unpack(
      4, inputIter, BYTES(4), kNumValues, outputIter);
}

BENCHMARK(velox_5_16) {
  const uint8_t* inputIter = inputBuffer;
  uint16_t* outputIter = reinterpret_cast<uint16_t*>(outputBuffer);

  facebook::velox::dwio::common::unpack(
      5, inputIter, BYTES(5), kNumValues, outputIter);
}

BENCHMARK(velox_6_16) {
  const uint8_t* inputIter = inputBuffer;
  uint16_t* outputIter = reinterpret_cast<uint16_t*>(outputBuffer);

  facebook::velox::dwio::common::unpack(
      6, inputIter, BYTES(6), kNumValues, outputIter);
}

BENCHMARK(velox_7_16) {
  const uint8_t* inputIter = inputBuffer;
  uint16_t* outputIter = reinterpret_cast<uint16_t*>(outputBuffer);

  facebook::velox::dwio::common::unpack(
      7, inputIter, BYTES(7), kNumValues, outputIter);
}

BENCHMARK(velox_8_16) {
  const uint8_t* inputIter = inputBuffer;
  uint16_t* outputIter = reinterpret_cast<uint16_t*>(outputBuffer);

  facebook::velox::dwio::common::unpack(
      8, inputIter, BYTES(8), kNumValues, outputIter);
}

BENCHMARK(velox_9_16) {
  const uint8_t* inputIter = inputBuffer;
  uint16_t* outputIter = reinterpret_cast<uint16_t*>(outputBuffer);

  facebook::velox::dwio::common::unpack(
      9, inputIter, BYTES(9), kNumValues, outputIter);
}

BENCHMARK(velox_10_16) {
  const uint8_t* inputIter = inputBuffer;
  uint32_t* outputIter = reinterpret_cast<uint32_t*>(outputBuffer);

  facebook::velox::dwio::common::unpack(
      10, inputIter, BYTES(10), kNumValues, outputIter);
}

BENCHMARK(velox_11_16) {
  const uint8_t* inputIter = inputBuffer;
  uint32_t* outputIter = reinterpret_cast<uint32_t*>(outputBuffer);

  facebook::velox::dwio::common::unpack(
      11, inputIter, BYTES(11), kNumValues, outputIter);
}

BENCHMARK(velox_12_16) {
  const uint8_t* inputIter = inputBuffer;
  uint32_t* outputIter = reinterpret_cast<uint32_t*>(outputBuffer);

  facebook::velox::dwio::common::unpack(
      12, inputIter, BYTES(12), kNumValues, outputIter);
}

BENCHMARK(velox_13_16) {
  const uint8_t* inputIter = inputBuffer;
  uint32_t* outputIter = reinterpret_cast<uint32_t*>(outputBuffer);

  facebook::velox::dwio::common::unpack(
      13, inputIter, BYTES(13), kNumValues, outputIter);
}

BENCHMARK(velox_14_16) {
  const uint8_t* inputIter = inputBuffer;
  uint32_t* outputIter = reinterpret_cast<uint32_t*>(outputBuffer);
  facebook::velox::dwio::common::unpack(
      14, inputIter, BYTES(14), kNumValues, outputIter);
}

BENCHMARK(velox_15_16) {
  const uint8_t* inputIter = inputBuffer;
  uint32_t* outputIter = reinterpret_cast<uint32_t*>(outputBuffer);
  facebook::velox::dwio::common::unpack(
      15, inputIter, BYTES(15), kNumValues, outputIter);
}

BENCHMARK(velox_16_16) {
  const uint8_t* inputIter = inputBuffer;
  uint32_t* outputIter = reinterpret_cast<uint32_t*>(outputBuffer);
  facebook::velox::dwio::common::unpack(
      16, inputIter, BYTES(16), kNumValues, outputIter);
}

BENCHMARK(velox_1_32) {
  const uint8_t* inputIter = inputBuffer;
  uint32_t* outputIter = reinterpret_cast<uint32_t*>(outputBuffer);
  facebook::velox::dwio::common::unpack(
      1, inputIter, BYTES(1), kNumValues, outputIter);
}

BENCHMARK(velox_2_32) {
  const uint8_t* inputIter = inputBuffer;
  uint32_t* outputIter = reinterpret_cast<uint32_t*>(outputBuffer);
  facebook::velox::dwio::common::unpack(
      2, inputIter, BYTES(2), kNumValues, outputIter);
}

BENCHMARK(velox_3_32) {
  const uint8_t* inputIter = inputBuffer;
  uint32_t* outputIter = reinterpret_cast<uint32_t*>(outputBuffer);
  facebook::velox::dwio::common::unpack(
      3, inputIter, BYTES(3), kNumValues, outputIter);
}

BENCHMARK(velox_4_32) {
  const uint8_t* inputIter = inputBuffer;
  uint32_t* outputIter = reinterpret_cast<uint32_t*>(outputBuffer);
  facebook::velox::dwio::common::unpack(
      4, inputIter, BYTES(4), kNumValues, outputIter);
}

BENCHMARK(velox_5_32) {
  const uint8_t* inputIter = inputBuffer;
  uint32_t* outputIter = reinterpret_cast<uint32_t*>(outputBuffer);
  facebook::velox::dwio::common::unpack(
      5, inputIter, BYTES(5), kNumValues, outputIter);
}

BENCHMARK(velox_6_32) {
  const uint8_t* inputIter = inputBuffer;
  uint32_t* outputIter = reinterpret_cast<uint32_t*>(outputBuffer);
  facebook::velox::dwio::common::unpack(
      6, inputIter, BYTES(6), kNumValues, outputIter);
}

BENCHMARK(velox_7_32) {
  const uint8_t* inputIter = inputBuffer;
  uint32_t* outputIter = reinterpret_cast<uint32_t*>(outputBuffer);
  facebook::velox::dwio::common::unpack(
      7, inputIter, BYTES(7), kNumValues, outputIter);
}

BENCHMARK(velox_8_32) {
  const uint8_t* inputIter = inputBuffer;
  uint32_t* outputIter = reinterpret_cast<uint32_t*>(outputBuffer);
  facebook::velox::dwio::common::unpack(
      8, inputIter, BYTES(8), kNumValues, outputIter);
}

BENCHMARK(velox_9_32) {
  const uint8_t* inputIter = inputBuffer;
  uint32_t* outputIter = reinterpret_cast<uint32_t*>(outputBuffer);
  facebook::velox::dwio::common::unpack(
      9, inputIter, BYTES(9), kNumValues, outputIter);
}

BENCHMARK(velox_10_32) {
  const uint8_t* inputIter = inputBuffer;
  uint32_t* outputIter = reinterpret_cast<uint32_t*>(outputBuffer);
  facebook::velox::dwio::common::unpack(
      10, inputIter, BYTES(10), kNumValues, outputIter);
}

BENCHMARK(velox_11_32) {
  const uint8_t* inputIter = inputBuffer;
  uint32_t* outputIter = reinterpret_cast<uint32_t*>(outputBuffer);
  facebook::velox::dwio::common::unpack(
      11, inputIter, BYTES(11), kNumValues, outputIter);
}

BENCHMARK(velox_12_32) {
  const uint8_t* inputIter = inputBuffer;
  uint32_t* outputIter = reinterpret_cast<uint32_t*>(outputBuffer);
  facebook::velox::dwio::common::unpack(
      12, inputIter, BYTES(12), kNumValues, outputIter);
}

BENCHMARK(velox_13_32) {
  const uint8_t* inputIter = inputBuffer;
  uint32_t* outputIter = reinterpret_cast<uint32_t*>(outputBuffer);
  facebook::velox::dwio::common::unpack(
      13, inputIter, BYTES(13), kNumValues, outputIter);
}

BENCHMARK(velox_14_32) {
  const uint8_t* inputIter = inputBuffer;
  uint32_t* outputIter = reinterpret_cast<uint32_t*>(outputBuffer);
  facebook::velox::dwio::common::unpack(
      14, inputIter, BYTES(14), kNumValues, outputIter);
}

BENCHMARK(velox_15_32) {
  const uint8_t* inputIter = inputBuffer;
  uint32_t* outputIter = reinterpret_cast<uint32_t*>(outputBuffer);
  facebook::velox::dwio::common::unpack(
      15, inputIter, BYTES(15), kNumValues, outputIter);
}

BENCHMARK(velox_16_32) {
  const uint8_t* inputIter = inputBuffer;
  uint32_t* outputIter = reinterpret_cast<uint32_t*>(outputBuffer);
  facebook::velox::dwio::common::unpack(
      16, inputIter, BYTES(16), kNumValues, outputIter);
}

BENCHMARK(velox_17_32) {
  const uint8_t* inputIter = inputBuffer;
  uint32_t* outputIter = reinterpret_cast<uint32_t*>(outputBuffer);
  facebook::velox::dwio::common::unpack(
      17, inputIter, BYTES(17), kNumValues, outputIter);
}

BENCHMARK(velox_18_32) {
  const uint8_t* inputIter = inputBuffer;
  uint32_t* outputIter = reinterpret_cast<uint32_t*>(outputBuffer);
  facebook::velox::dwio::common::unpack(
      18, inputIter, BYTES(18), kNumValues, outputIter);
}

BENCHMARK(velox_19_32) {
  const uint8_t* inputIter = inputBuffer;
  uint32_t* outputIter = reinterpret_cast<uint32_t*>(outputBuffer);
  facebook::velox::dwio::common::unpack(
      19, inputIter, BYTES(19), kNumValues, outputIter);
}

BENCHMARK(velox_20_32) {
  const uint8_t* inputIter = inputBuffer;
  uint32_t* outputIter = reinterpret_cast<uint32_t*>(outputBuffer);
  facebook::velox::dwio::common::unpack(
      20, inputIter, BYTES(20), kNumValues, outputIter);
}

BENCHMARK(velox_21_32) {
  const uint8_t* inputIter = inputBuffer;
  uint32_t* outputIter = reinterpret_cast<uint32_t*>(outputBuffer);
  facebook::velox::dwio::common::unpack(
      21, inputIter, BYTES(21), kNumValues, outputIter);
}

BENCHMARK(velox_22_32) {
  const uint8_t* inputIter = inputBuffer;
  uint32_t* outputIter = reinterpret_cast<uint32_t*>(outputBuffer);
  facebook::velox::dwio::common::unpack(
      22, inputIter, BYTES(22), kNumValues, outputIter);
}

BENCHMARK(velox_23_32) {
  const uint8_t* inputIter = inputBuffer;
  uint32_t* outputIter = reinterpret_cast<uint32_t*>(outputBuffer);
  facebook::velox::dwio::common::unpack(
      23, inputIter, BYTES(23), kNumValues, outputIter);
}

BENCHMARK(velox_24_32) {
  const uint8_t* inputIter = inputBuffer;
  uint32_t* outputIter = reinterpret_cast<uint32_t*>(outputBuffer);
  facebook::velox::dwio::common::unpack(
      24, inputIter, BYTES(24), kNumValues, outputIter);
}

BENCHMARK(velox_25_32) {
  const uint8_t* inputIter = inputBuffer;
  uint32_t* outputIter = reinterpret_cast<uint32_t*>(outputBuffer);
  facebook::velox::dwio::common::unpack(
      9, inputIter, BYTES(9), kNumValues, outputIter);
}

BENCHMARK(velox_26_32) {
  const uint8_t* inputIter = inputBuffer;
  uint32_t* outputIter = reinterpret_cast<uint32_t*>(outputBuffer);
  facebook::velox::dwio::common::unpack(
      10, inputIter, BYTES(10), kNumValues, outputIter);
}

BENCHMARK(velox_27_32) {
  const uint8_t* inputIter = inputBuffer;
  uint32_t* outputIter = reinterpret_cast<uint32_t*>(outputBuffer);
  facebook::velox::dwio::common::unpack(
      11, inputIter, BYTES(11), kNumValues, outputIter);
}

BENCHMARK(velox_28_32) {
  const uint8_t* inputIter = inputBuffer;
  uint32_t* outputIter = reinterpret_cast<uint32_t*>(outputBuffer);
  facebook::velox::dwio::common::unpack(
      12, inputIter, BYTES(12), kNumValues, outputIter);
}

BENCHMARK(velox_29_32) {
  const uint8_t* inputIter = inputBuffer;
  uint32_t* outputIter = reinterpret_cast<uint32_t*>(outputBuffer);
  facebook::velox::dwio::common::unpack(
      13, inputIter, BYTES(13), kNumValues, outputIter);
}

BENCHMARK(velox_30_32) {
  const uint8_t* inputIter = inputBuffer;
  uint32_t* outputIter = reinterpret_cast<uint32_t*>(outputBuffer);
  facebook::velox::dwio::common::unpack(
      14, inputIter, BYTES(14), kNumValues, outputIter);
}

BENCHMARK(velox_31_32) {
  const uint8_t* inputIter = inputBuffer;
  uint32_t* outputIter = reinterpret_cast<uint32_t*>(outputBuffer);
  facebook::velox::dwio::common::unpack(
      15, inputIter, BYTES(15), kNumValues, outputIter);
}

BENCHMARK(velox_32_32) {
  const uint8_t* inputIter = inputBuffer;
  uint32_t* outputIter = reinterpret_cast<uint32_t*>(outputBuffer);
  facebook::velox::dwio::common::unpack(
      16, inputIter, BYTES(16), kNumValues, outputIter);
}

// The following code was copied from IntDecoder.cpp, which was a modified
// version of BitUnpacking.h.
typedef int32_t __m256si __attribute__((__vector_size__(32), __may_alias__));

typedef int32_t __m256si_u
    __attribute__((__vector_size__(32), __may_alias__, __aligned__(1)));

template <int8_t i>
auto as4x64(__m256i x) {
  return _mm256_cvtepu32_epi64(_mm256_extracti128_si256(x, i));
}

template <typename T>
void store8Ints(__m256i eightInts, int32_t i, T* FOLLY_NONNULL result) {
  if (sizeof(T) == 4) {
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(result + i), eightInts);
  } else {
    _mm256_storeu_si256(
        reinterpret_cast<__m256i*>(result + i), as4x64<0>(eightInts));
    _mm256_storeu_si256(
        reinterpret_cast<__m256i*>(result + i + 4), as4x64<1>(eightInts));
  }
}

template <typename T>
inline T* addBytes(T* pointer, int32_t bytes) {
  return reinterpret_cast<T*>(reinterpret_cast<uint64_t>(pointer) + bytes);
}

template <typename T>
inline __m256i as256i(T x) {
  return reinterpret_cast<__m256i>(x);
}

template <typename T>
inline __m256si as8x32(T x) {
  return reinterpret_cast<__m256si>(x);
}

template <uint8_t width, typename T>
FOLLY_ALWAYS_INLINE __m256i gather8Sparse(
    const uint64_t* bits,
    int32_t bitOffset,
    const int32_t* rows,
    int32_t i,
    __m256si masks,
    T* result) {
  constexpr __m256si kMultipliers = {256, 128, 64, 32, 16, 8, 4, 2};

  auto indices =
      *reinterpret_cast<const __m256si_u*>(rows + i) * width + bitOffset;
  __m256si multipliers;
  if (width % 8 != 0) {
    multipliers = (__m256si)_mm256_permutevar8x32_epi32(
        as256i(kMultipliers), as256i(indices & 7));
  }
  auto byteIndices = indices >> 3;
  auto data = as8x32(_mm256_i32gather_epi32(
      reinterpret_cast<const int*>(bits), as256i(byteIndices), 1));
  if (width % 8 != 0) {
    data = (data * multipliers) >> 8;
  }
  return as256i(data & masks);
}

template <uint8_t width, typename T>
int32_t decode1To24(
    const uint64_t* bits,
    int32_t bitOffset,
    const int* rows,
    int32_t numRows,
    T* result) {
  constexpr uint64_t kMask = bits::lowMask(width);
  constexpr uint64_t kMask2 = kMask | (kMask << 8);
  constexpr uint64_t kMask4 = kMask2 | (kMask2 << 16);
  constexpr uint64_t kDepMask8 = kMask4 | (kMask4 << 32);
  constexpr uint64_t kMask16 = kMask | (kMask << 16);
  constexpr uint64_t kDepMask16 = kMask16 | (kMask16 << 32);
  int32_t i = 0;
  const auto masks = as8x32(_mm256_set1_epi32(kMask));
  for (; i + 8 <= numRows; i += 8) {
    auto row = rows[i];
    auto endRow = rows[i + 7];
    __m256i eightInts;
    if (width <= 16 && endRow - row == 7) {
      // Special cases for 8 contiguous values with <= 16 bits.
      if (width <= 8) {
        uint64_t eightBytes;
        if (width == 8) {
          if (!bitOffset) {
            eightBytes = *addBytes(bits, row);
          } else {
            eightBytes =
                bits::detail::loadBits<uint64_t>(bits, bitOffset + 8 * row, 64);
          }
        } else {
          auto bit = row * width + bitOffset;
          auto byte = bit >> 3;
          auto shift = bit & 7;
          uint64_t word = *addBytes(bits, byte) >> shift;
          eightBytes = _pdep_u64(word, kDepMask8);
        }
        eightInts = _mm256_cvtepu8_epi32(
            _mm_loadl_epi64(reinterpret_cast<const __m128i*>(&eightBytes)));
      } else {
        // Use pdep to shift 2 words of bit packed data with width
        // 9-16. For widts <= 14 four bit packed fields can always be
        // loaded with a single uint64_t load. For 15 and 16 bits this
        // depends on the start bit position. For either case we fill
        // an array of 2x64 bits and widen that to a 8x32 word.
        uint64_t words[2];
        if (width <= 14) {
          auto bit = row * width + bitOffset;
          auto byte = bit >> 3;
          auto shift = bit & 7;
          uint64_t word = *addBytes(bits, byte) >> shift;
          words[0] = _pdep_u64(word, kDepMask16);
          bit += 4 * width;
          byte = bit >> 3;
          shift = bit & 7;
          word = *addBytes(bits, byte) >> shift;
          words[1] = _pdep_u64(word, kDepMask16);
        } else {
          words[0] = bits::detail::loadBits<uint64_t>(
              bits, bitOffset + width * row, 64);
          words[1] = bits::detail::loadBits<uint64_t>(
              bits, bitOffset + width * (row + 4), 64);
          if (width == 15) {
            words[0] = _pdep_u64(words[0], kDepMask16);
            words[1] = _pdep_u64(words[1], kDepMask16);
          }
        }
        eightInts = _mm256_cvtepu16_epi32(
            _mm_load_si128(reinterpret_cast<const __m128i*>(&words)));
      }
    } else {
      eightInts = gather8Sparse<width>(bits, bitOffset, rows, i, masks, result);
    }
    store8Ints(eightInts, i, result);
  }
  return i;
}

BENCHMARK(intdecoder_1_32) {
  const uint64_t* inputIter = reinterpret_cast<const uint64_t*>(inputBuffer);
  uint32_t* outputIter = reinterpret_cast<uint32_t*>(outputBuffer);
  decode1To24<1, uint32_t>(
      inputIter,
      0,
      reinterpret_cast<const int*>(rows.data()),
      kNumValues,
      outputIter);
}

BENCHMARK(intdecoder_2_32) {
  const uint64_t* inputIter = reinterpret_cast<const uint64_t*>(inputBuffer);
  uint32_t* outputIter = reinterpret_cast<uint32_t*>(outputBuffer);
  decode1To24<2, uint32_t>(
      inputIter,
      0,
      reinterpret_cast<const int*>(rows.data()),
      kNumValues,
      outputIter);
}

BENCHMARK(intdecoder_3_32) {
  const uint64_t* inputIter = reinterpret_cast<const uint64_t*>(inputBuffer);
  uint32_t* outputIter = reinterpret_cast<uint32_t*>(outputBuffer);
  decode1To24<3, uint32_t>(
      inputIter,
      0,
      reinterpret_cast<const int*>(rows.data()),
      kNumValues,
      outputIter);
}

BENCHMARK(intdecoder_4_32) {
  const uint64_t* inputIter = reinterpret_cast<const uint64_t*>(inputBuffer);
  uint32_t* outputIter = reinterpret_cast<uint32_t*>(outputBuffer);
  decode1To24<4, uint32_t>(
      inputIter,
      0,
      reinterpret_cast<const int*>(rows.data()),
      kNumValues,
      outputIter);
}

BENCHMARK(intdecoder_5_32) {
  const uint64_t* inputIter = reinterpret_cast<const uint64_t*>(inputBuffer);
  uint32_t* outputIter = reinterpret_cast<uint32_t*>(outputBuffer);
  decode1To24<5, uint32_t>(
      inputIter,
      0,
      reinterpret_cast<const int*>(rows.data()),
      kNumValues,
      outputIter);
}

BENCHMARK(intdecoder_6_32) {
  const uint64_t* inputIter = reinterpret_cast<const uint64_t*>(inputBuffer);
  uint32_t* outputIter = reinterpret_cast<uint32_t*>(outputBuffer);
  decode1To24<6, uint32_t>(
      inputIter,
      0,
      reinterpret_cast<const int*>(rows.data()),
      kNumValues,
      outputIter);
}

BENCHMARK(intdecoder_7_32) {
  const uint64_t* inputIter = reinterpret_cast<const uint64_t*>(inputBuffer);
  uint32_t* outputIter = reinterpret_cast<uint32_t*>(outputBuffer);
  decode1To24<7, uint32_t>(
      inputIter,
      0,
      reinterpret_cast<const int*>(rows.data()),
      kNumValues,
      outputIter);
}

BENCHMARK(intdecoder_8_32) {
  const uint64_t* inputIter = reinterpret_cast<const uint64_t*>(inputBuffer);
  uint32_t* outputIter = reinterpret_cast<uint32_t*>(outputBuffer);
  decode1To24<8, uint32_t>(
      inputIter,
      0,
      reinterpret_cast<const int*>(rows.data()),
      kNumValues,
      outputIter);
}

BENCHMARK(intdecoder_9_32) {
  const uint64_t* inputIter = reinterpret_cast<const uint64_t*>(inputBuffer);
  uint32_t* outputIter = reinterpret_cast<uint32_t*>(outputBuffer);
  decode1To24<9, uint32_t>(
      inputIter,
      0,
      reinterpret_cast<const int*>(rows.data()),
      kNumValues,
      outputIter);
}

BENCHMARK(intdecoder_10_32) {
  const uint64_t* inputIter = reinterpret_cast<const uint64_t*>(inputBuffer);
  uint32_t* outputIter = reinterpret_cast<uint32_t*>(outputBuffer);
  decode1To24<10, uint32_t>(
      inputIter,
      0,
      reinterpret_cast<const int*>(rows.data()),
      kNumValues,
      outputIter);
}

BENCHMARK(intdecoder_11_32) {
  const uint64_t* inputIter = reinterpret_cast<const uint64_t*>(inputBuffer);
  uint32_t* outputIter = reinterpret_cast<uint32_t*>(outputBuffer);
  decode1To24<11, uint32_t>(
      inputIter,
      0,
      reinterpret_cast<const int*>(rows.data()),
      kNumValues,
      outputIter);
}

BENCHMARK(intdecoder_12_32) {
  const uint64_t* inputIter = reinterpret_cast<const uint64_t*>(inputBuffer);
  uint32_t* outputIter = reinterpret_cast<uint32_t*>(outputBuffer);
  decode1To24<12, uint32_t>(
      inputIter,
      0,
      reinterpret_cast<const int*>(rows.data()),
      kNumValues,
      outputIter);
}

BENCHMARK(intdecoder_13_32) {
  const uint64_t* inputIter = reinterpret_cast<const uint64_t*>(inputBuffer);
  uint32_t* outputIter = reinterpret_cast<uint32_t*>(outputBuffer);
  decode1To24<13, uint32_t>(
      inputIter,
      0,
      reinterpret_cast<const int*>(rows.data()),
      kNumValues,
      outputIter);
}

BENCHMARK(intdecoder_14_32) {
  const uint64_t* inputIter = reinterpret_cast<const uint64_t*>(inputBuffer);
  uint32_t* outputIter = reinterpret_cast<uint32_t*>(outputBuffer);
  decode1To24<14, uint32_t>(
      inputIter,
      0,
      reinterpret_cast<const int*>(rows.data()),
      kNumValues,
      outputIter);
}

BENCHMARK(intdecoder_15_32) {
  const uint64_t* inputIter = reinterpret_cast<const uint64_t*>(inputBuffer);
  uint32_t* outputIter = reinterpret_cast<uint32_t*>(outputBuffer);
  decode1To24<15, uint32_t>(
      inputIter,
      0,
      reinterpret_cast<const int*>(rows.data()),
      kNumValues,
      outputIter);
}

BENCHMARK(intdecoder_16_32) {
  const uint64_t* inputIter = reinterpret_cast<const uint64_t*>(inputBuffer);
  uint32_t* outputIter = reinterpret_cast<uint32_t*>(outputBuffer);
  decode1To24<16, uint32_t>(
      inputIter,
      0,
      reinterpret_cast<const int*>(rows.data()),
      kNumValues,
      outputIter);
}

BENCHMARK(intdecoder_17_32) {
  const uint64_t* inputIter = reinterpret_cast<const uint64_t*>(inputBuffer);
  uint32_t* outputIter = reinterpret_cast<uint32_t*>(outputBuffer);
  decode1To24<17, uint32_t>(
      inputIter,
      0,
      reinterpret_cast<const int*>(rows.data()),
      kNumValues,
      outputIter);
}

BENCHMARK(intdecoder_18_32) {
  const uint64_t* inputIter = reinterpret_cast<const uint64_t*>(inputBuffer);
  uint32_t* outputIter = reinterpret_cast<uint32_t*>(outputBuffer);
  decode1To24<18, uint32_t>(
      inputIter,
      0,
      reinterpret_cast<const int*>(rows.data()),
      kNumValues,
      outputIter);
}

BENCHMARK(intdecoder_19_32) {
  const uint64_t* inputIter = reinterpret_cast<const uint64_t*>(inputBuffer);
  uint32_t* outputIter = reinterpret_cast<uint32_t*>(outputBuffer);
  decode1To24<19, uint32_t>(
      inputIter,
      0,
      reinterpret_cast<const int*>(rows.data()),
      kNumValues,
      outputIter);
}
BENCHMARK(intdecoder_20_32) {
  const uint64_t* inputIter = reinterpret_cast<const uint64_t*>(inputBuffer);
  uint32_t* outputIter = reinterpret_cast<uint32_t*>(outputBuffer);
  decode1To24<20, uint32_t>(
      inputIter,
      0,
      reinterpret_cast<const int*>(rows.data()),
      kNumValues,
      outputIter);
}

BENCHMARK(intdecoder_21_32) {
  const uint64_t* inputIter = reinterpret_cast<const uint64_t*>(inputBuffer);
  uint32_t* outputIter = reinterpret_cast<uint32_t*>(outputBuffer);
  decode1To24<21, uint32_t>(
      inputIter,
      0,
      reinterpret_cast<const int*>(rows.data()),
      kNumValues,
      outputIter);
}

BENCHMARK(intdecoder_22_32) {
  const uint64_t* inputIter = reinterpret_cast<const uint64_t*>(inputBuffer);
  uint32_t* outputIter = reinterpret_cast<uint32_t*>(outputBuffer);
  decode1To24<22, uint32_t>(
      inputIter,
      0,
      reinterpret_cast<const int*>(rows.data()),
      kNumValues,
      outputIter);
}

BENCHMARK(intdecoder_23_32) {
  const uint64_t* inputIter = reinterpret_cast<const uint64_t*>(inputBuffer);
  uint32_t* outputIter = reinterpret_cast<uint32_t*>(outputBuffer);
  decode1To24<23, uint32_t>(
      inputIter,
      0,
      reinterpret_cast<const int*>(rows.data()),
      kNumValues,
      outputIter);
}

BENCHMARK(intdecoder_24_32) {
  const uint64_t* inputIter = reinterpret_cast<const uint64_t*>(inputBuffer);
  uint32_t* outputIter = reinterpret_cast<uint32_t*>(outputBuffer);
  decode1To24<24, uint32_t>(
      inputIter,
      0,
      reinterpret_cast<const int*>(rows.data()),
      kNumValues,
      outputIter);
}

void populateInputBuffer(
    uint8_t bitWidth,
    const uint8_t* inputBuf,
    uint32_t inputBytes) {
  auto gen = std::bind(
      std::uniform_int_distribution<>(0, (1L << bitWidth) - 1),
      std::default_random_engine());
  arrow::bit_util::BitWriter bitWriter(
      const_cast<uint8_t*>(inputBuf), inputBytes);
  for (auto j = 0; j < kNumValues; j++) {
    bitWriter.PutValue(gen(), bitWidth);
  }
}

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  populateInputBuffer(16, inputBuffer, BYTES(16));
  std::iota(rows.begin(), rows.end(), 0);

  runBenchmarks();

  return 0;
}

#if 0

The following results were from Apple clang version 12.0.5 (clang-1205.0.22.11),
on Macbook Pro with CPU Intel(R) Core(TM) i9-9980HK CPU @ 2.40GHz (CoffeeLake)
============================================================================
[...]ommon/tests/BitUnpackingBenchmark.cpp     relative  time/iter   iters/s
============================================================================
duckdb_1_8                                                 12.30ms     81.29
duckdb_2_8                                                 13.74ms     72.80
duckdb_3_8                                                 14.94ms     66.91
duckdb_4_8                                                 16.52ms     60.52
duckdb_5_8                                                 18.08ms     55.31
duckdb_6_8                                                 19.63ms     50.95
duckdb_7_8                                                 21.18ms     47.22
duckdb_8_8                                                 22.79ms     43.88
duckdb_1_16                                                12.01ms     83.27
duckdb_2_16                                                13.20ms     75.78
duckdb_3_16                                                14.42ms     69.35
duckdb_4_16                                                15.73ms     63.58
duckdb_5_16                                                17.22ms     58.06
duckdb_6_16                                                18.73ms     53.39
duckdb_7_16                                                20.20ms     49.51
duckdb_8_16                                                21.80ms     45.87
duckdb_9_16                                                23.31ms     42.90
duckdb_10_16                                               25.01ms     39.99
duckdb_11_16                                               26.70ms     37.45
duckdb_12_16                                               28.35ms     35.27
duckdb_13_16                                               37.07ms     26.97
duckdb_14_16                                               31.60ms     31.64
duckdb_15_16                                               39.37ms     25.40
duckdb_16_16                                               35.28ms     28.35
duckdb_1_32                                                11.99ms     83.41
duckdb_2_32                                                13.31ms     75.11
duckdb_3_32                                                14.44ms     69.24
duckdb_4_32                                                15.50ms     64.52
duckdb_5_32                                                16.81ms     59.50
duckdb_6_32                                                18.14ms     55.13
duckdb_7_32                                                19.35ms     51.67
duckdb_8_32                                                20.61ms     48.51
duckdb_9_32                                                30.14ms     33.17
duckdb_10_32                                               37.12ms     26.94
duckdb_11_32                                               44.60ms     22.42
duckdb_12_32                                               52.79ms     18.94
duckdb_13_32                                               59.12ms     16.92
duckdb_14_32                                               67.31ms     14.86
duckdb_15_32                                               76.58ms     13.06
duckdb_16_32                                               85.32ms     11.72
duckdb_17_32                                               82.09ms     12.18
duckdb_18_32                                               82.70ms     12.09
duckdb_19_32                                               82.48ms     12.12
duckdb_20_32                                               84.62ms     11.82
duckdb_21_32                                               85.79ms     11.66
duckdb_22_32                                               88.56ms     11.29
duckdb_23_32                                               88.57ms     11.29
duckdb_24_32                                               87.55ms     11.42
duckdb_25_32                                               91.97ms     10.87
duckdb_26_32                                               93.33ms     10.71
duckdb_27_32                                               94.92ms     10.53
duckdb_28_32                                               99.32ms     10.07
duckdb_29_32                                               99.75ms     10.03
duckdb_30_32                                               98.47ms     10.16
duckdb_31_32                                               99.14ms     10.09
duckdb_32_32                                              104.87ms      9.54
arrow_1_8                                                   3.09ms    323.94
arrow_2_8                                                   3.06ms    326.69
arrow_3_8                                                   3.17ms    314.97
arrow_4_8                                                   2.97ms    336.83
arrow_5_8                                                   3.53ms    283.05
arrow_6_8                                                   3.51ms    284.67
arrow_7_8                                                   3.63ms    275.14
arrow_8_8                                                   3.22ms    310.69
arrow_1_16                                                  3.09ms    323.87
arrow_2_16                                                  3.06ms    326.73
arrow_3_16                                                  3.18ms    314.93
arrow_4_16                                                  2.97ms    337.02
arrow_5_16                                                  3.53ms    283.36
arrow_6_16                                                  3.51ms    284.79
arrow_7_16                                                  3.63ms    275.21
arrow_8_16                                                  3.22ms    310.82
arrow_9_16                                                  3.90ms    256.25
arrow_10_16                                                 3.94ms    253.89
arrow_11_16                                                 4.10ms    244.07
arrow_12_16                                                 4.02ms    248.86
arrow_13_16                                                 4.28ms    233.58
arrow_14_16                                                 4.33ms    231.17
arrow_15_16                                                 4.60ms    217.18
arrow_16_16                                                 3.66ms    273.14
arrow_1_32                                                  3.09ms    323.91
arrow_2_32                                                  3.06ms    326.78
arrow_3_32                                                  3.18ms    314.88
arrow_4_32                                                  2.97ms    337.15
arrow_5_32                                                  3.50ms    285.89
arrow_6_32                                                  3.51ms    284.81
arrow_7_32                                                  3.58ms    279.26
arrow_8_32                                                  3.22ms    310.87
arrow_9_32                                                  3.90ms    256.11
arrow_10_32                                                 3.95ms    253.43
arrow_11_32                                                 4.10ms    244.10
arrow_12_32                                                 4.00ms    249.84
arrow_13_32                                                 4.29ms    233.25
arrow_14_32                                                 4.33ms    231.12
arrow_15_32                                                 4.59ms    217.76
arrow_16_32                                                 3.55ms    281.32
arrow_17_32                                                 4.76ms    210.30
arrow_18_32                                                 4.77ms    209.44
arrow_19_32                                                 4.92ms    203.11
arrow_20_32                                                 4.79ms    208.74
arrow_21_32                                                 5.10ms    196.17
arrow_22_32                                                 5.12ms    195.28
arrow_23_32                                                 5.30ms    188.54
arrow_24_32                                                 5.00ms    199.82
arrow_25_32                                                 5.57ms    179.45
arrow_26_32                                                 5.64ms    177.35
arrow_27_32                                                 5.75ms    173.78
arrow_28_32                                                 5.70ms    175.32
arrow_29_32                                                 5.99ms    166.99
arrow_30_32                                                 6.20ms    161.33
arrow_31_32                                                 6.34ms    157.73
arrow_32_32                                                 4.17ms    240.06
lemirefastpforlib_1_8                                      4.98ms    200.83
lemirefastpforlib_2_8                                      4.56ms    219.24
lemirefastpforlib_3_8                                      5.33ms    187.66
lemirefastpforlib_4_8                                      4.03ms    247.87
lemirefastpforlib_5_8                                      5.52ms    181.26
lemirefastpforlib_6_8                                      5.13ms    194.87
lemirefastpforlib_7_8                                      6.41ms    155.99
lemirefastpforlib_8_8                                      3.06ms    327.05
lemirefastpforlib_1_16                                     2.59ms    386.46
lemirefastpforlib_2_16                                     2.61ms    383.68
lemirefastpforlib_3_16                                     3.08ms    324.72
lemirefastpforlib_4_16                                     2.63ms    380.12
lemirefastpforlib_5_16                                     3.37ms    297.16
lemirefastpforlib_6_16                                     6.87ms    145.58
lemirefastpforlib_7_16                                     3.67ms    272.80
lemirefastpforlib_8_16                                     2.75ms    363.85
lemirefastpforlib_9_16                                     3.97ms    251.76
lemirefastpforlib_10_16                                    4.11ms    243.16
lemirefastpforlib_11_16                                    4.39ms    227.67
lemirefastpforlib_12_16                                    4.17ms    239.78
lemirefastpforlib_13_16                                    4.73ms    211.62
lemirefastpforlib_14_16                                    4.76ms    210.17
lemirefastpforlib_15_16                                    5.05ms    198.06
lemirefastpforlib_16_16                                    3.10ms    322.69
lemirefastpforlib_1_32                                     2.88ms    346.90
lemirefastpforlib_2_32                                     2.90ms    344.37
lemirefastpforlib_3_32                                     3.11ms    321.07
lemirefastpforlib_4_32                                     2.96ms    338.00
lemirefastpforlib_5_32                                     3.47ms    288.03
lemirefastpforlib_6_32                                     3.49ms    286.62
lemirefastpforlib_7_32                                     4.06ms    246.33
lemirefastpforlib_8_32                                     3.05ms    327.89
lemirefastpforlib_9_32                                     4.36ms    229.18
lemirefastpforlib_10_32                                    4.37ms    228.89
lemirefastpforlib_11_32                                    4.68ms    213.73
lemirefastpforlib_12_32                                    4.38ms    228.41
lemirefastpforlib_13_32                                    4.84ms    206.64
lemirefastpforlib_14_32                                    4.35ms    229.91
lemirefastpforlib_15_32                                    4.49ms    222.47
lemirefastpforlib_16_32                                    3.39ms    294.87
lemirefastpforlib_17_32                                    4.68ms    213.63
lemirefastpforlib_18_32                                    4.95ms    201.90
lemirefastpforlib_19_32                                    4.43ms    225.56
lemirefastpforlib_20_32                                    4.31ms    232.12
lemirefastpforlib_21_32                                    4.91ms    203.52
lemirefastpforlib_22_32                                    4.62ms    216.63
lemirefastpforlib_23_32                                    5.05ms    198.22
lemirefastpforlib_24_32                                    4.45ms    224.80
lemirefastpforlib_25_32                                    5.89ms    169.69
lemirefastpforlib_26_32                                    5.34ms    187.11
lemirefastpforlib_27_32                                    6.09ms    164.08
lemirefastpforlib_28_32                                    5.47ms    182.89
lemirefastpforlib_29_32                                    6.18ms    161.83
lemirefastpforlib_30_32                                    5.79ms    172.65
lemirefastpforlib_31_32                                    6.00ms    166.73
lemirefastpforlib_32_32                                    3.50ms    285.39
lemirebmi2_1_32                                            1.59ms    628.45
lemirebmi2_2_32                                            1.60ms    625.96
lemirebmi2_3_32                                            1.67ms    600.46
lemirebmi2_4_32                                            1.71ms    583.61
lemirebmi2_5_32                                            1.74ms    573.20
lemirebmi2_6_32                                            2.52ms    396.05
lemirebmi2_7_32                                            2.58ms    387.43
lemirebmi2_8_32                                            2.72ms    367.92
lemirebmi2_9_32                                            2.80ms    356.63
lemirebmi2_10_32                                           2.70ms    370.26
lemirebmi2_11_32                                           2.73ms    366.18
lemirebmi2_12_32                                           2.77ms    360.66
lemirebmi2_13_32                                           3.10ms    322.79
lemirebmi2_14_32                                           3.34ms    299.07
lemirebmi2_15_32                                           3.83ms    260.93
lemirebmi2_16_32                                           3.99ms    250.58
lemirebmi2_17_32                                           4.14ms    241.47
lemirebmi2_18_32                                           4.18ms    238.96
lemirebmi2_19_32                                           4.29ms    232.83
lemirebmi2_20_32                                           4.28ms    233.47
lemirebmi2_21_32                                           4.41ms    226.52
lemirebmi2_22_32                                           4.42ms    226.02
lemirebmi2_23_32                                           4.55ms    219.62
lemirebmi2_24_32                                           4.54ms    220.22
lemirebmi2_25_32                                           4.72ms    211.98
lemirebmi2_26_32                                           4.71ms    212.36
lemirebmi2_27_32                                           4.83ms    207.20
lemirebmi2_28_32                                           4.86ms    205.79
lemirebmi2_29_32                                           3.09ms    323.51
lemirebmi2_30_32                                           4.99ms    200.21
lemirebmi2_31_32                                           5.09ms    196.54
lemirebmi2_32_32                                           4.32ms    231.40
velox_1_8                                                 382.82us     2.61K
velox_2_8                                                 394.91us     2.53K
velox_3_8                                                 417.56us     2.39K
velox_4_8                                                 421.39us     2.37K
velox_5_8                                                 444.86us     2.25K
velox_6_8                                                 472.78us     2.12K
velox_7_8                                                 498.44us     2.01K
velox_8_8                                                 530.15us     1.89K
velox_1_16                                                642.65us     1.56K
velox_2_16                                                676.22us     1.48K
velox_3_16                                                746.03us     1.34K
velox_4_16                                                769.44us     1.30K
velox_5_16                                                839.33us     1.19K
velox_6_16                                                895.07us     1.12K
velox_7_16                                                  1.07ms    931.30
velox_8_16                                                909.95us     1.10K
velox_9_16                                                  1.38ms    724.52
velox_10_16                                                 2.02ms    495.74
velox_11_16                                                 2.12ms    472.43
velox_12_16                                                 2.15ms    464.57
velox_13_16                                                 2.26ms    442.60
velox_14_16                                                 2.56ms    390.19
velox_15_16                                                 2.45ms    408.13
velox_16_16                                                 2.25ms    444.85
velox_1_32                                                  1.63ms    612.95
velox_2_32                                                  1.69ms    590.77
velox_3_32                                                  1.71ms    584.78
velox_4_32                                                  1.77ms    565.15
velox_5_32                                                  1.80ms    555.44
velox_6_32                                                  1.81ms    553.06
velox_7_32                                                  1.89ms    529.10
velox_8_32                                                  1.94ms    515.75
velox_9_32                                                  2.01ms    496.36
velox_10_32                                                 2.06ms    486.47
velox_11_32                                                 2.17ms    461.82
velox_12_32                                                 2.19ms    456.21
velox_13_32                                                 2.29ms    436.91
velox_14_32                                                 2.62ms    381.49
velox_15_32                                                 2.50ms    400.35
velox_16_32                                                 2.23ms    448.82
velox_17_32                                                 2.66ms    375.26
velox_18_32                                                 2.70ms    370.81
velox_19_32                                                 2.80ms    357.74
velox_20_32                                                 2.76ms    362.27
velox_21_32                                                 2.86ms    350.09
velox_22_32                                                 2.94ms    340.09
velox_23_32                                                 3.05ms    328.25
velox_24_32                                                 2.78ms    360.28
velox_25_32                                                 2.03ms    493.09
velox_26_32                                                 2.07ms    483.37
velox_27_32                                                 2.22ms    451.07
velox_28_32                                                 2.24ms    445.69
velox_29_32                                                 2.31ms    433.55
velox_30_32                                                 2.63ms    379.84
velox_31_32                                                 2.54ms    392.98
velox_32_32                                                 2.33ms    428.55
intdecoder_1_32                                             3.37ms    296.55
intdecoder_2_32                                             3.44ms    290.67
intdecoder_3_32                                             3.48ms    287.58
intdecoder_4_32                                             3.47ms    288.28
intdecoder_5_32                                             3.56ms    280.54
intdecoder_6_32                                             3.63ms    275.58
intdecoder_7_32                                             3.66ms    273.46
intdecoder_8_32                                             3.34ms    299.58
intdecoder_9_32                                             4.10ms    244.18
intdecoder_10_32                                            4.17ms    239.81
intdecoder_11_32                                            4.25ms    235.46
intdecoder_12_32                                            4.25ms    235.54
intdecoder_13_32                                            4.31ms    231.90
intdecoder_14_32                                            4.40ms    227.33
intdecoder_15_32                                            4.62ms    216.47
intdecoder_16_32                                            3.91ms    255.71
intdecoder_17_32                                            4.63ms    215.90
intdecoder_18_32                                            4.64ms    215.41
intdecoder_19_32                                            4.71ms    212.36
intdecoder_20_32                                            4.73ms    211.33
intdecoder_21_32                                            4.77ms    209.62
intdecoder_22_32                                            4.79ms    208.93
intdecoder_23_32                                            4.83ms    207.03
intdecoder_24_32                                            4.58ms    218.57

#endif
