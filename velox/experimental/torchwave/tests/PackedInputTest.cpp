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
#include <thread>

#include "velox/experimental/torchwave/PackRewrite.h"
#include "velox/experimental/torchwave/PackedInput.h"

#include <cuda_runtime.h> // @manual
#include <folly/ScopeGuard.h>
#include <folly/init/Init.h>
#include <gmock/gmock.h>

#include <torch/nativert/graph/Graph.h>

#include "velox/experimental/torchwave/Utils.h"
#include "velox/experimental/torchwave/WaveConfig.h"
#include "velox/experimental/torchwave/WaveGraph.h"
#include "velox/experimental/torchwave/tests/ExecutorTestBase.h"

namespace torch::wave {
namespace {

class PackedInputTest : public ExecutorTestBase {};

// Slot values whose ranges call for each stored width: the expected width is
// known from how the values are drawn, not from the encoder.
struct SlotData {
  at::Tensor values;
  int64_t bits;
};

std::vector<SlotData> makeSlots() {
  auto uniform = [](int64_t count, int64_t low, int64_t high) {
    return at::randint(low, high, {count}, at::kLong);
  };
  constexpr int64_t kInt32Min = std::numeric_limits<int32_t>::min();
  constexpr int64_t kInt32Max = std::numeric_limits<int32_t>::max();
  auto wide = uniform(3'000, kInt32Min, kInt32Max) * 1'000'003;
  wide[0] = std::numeric_limits<int64_t>::min();
  wide[1] = std::numeric_limits<int64_t>::max();
  return {
      {at::empty({0}, at::kLong), 0},
      {at::zeros({4'000}, at::kLong), 0},
      {at::full({500}, -5, at::kLong), 0},
      {uniform(9'000, 1'000'000, 1'000'200), 8},
      {uniform(20'000, -30'000, 30'000), 16},
      {uniform(5'000, kInt32Min, kInt32Max), 32},
      {wide, 64},
  };
}

PackRecipe recipeFor(const std::vector<SlotData>& slots) {
  PackRecipe recipe;
  recipe.inputName = "values";
  recipe.inputIndex = 0;
  for (const auto& slot : slots) {
    recipe.slots.push_back({.constLength = slot.values.numel()});
  }
  return recipe;
}

std::vector<int64_t> lengthsOf(const std::vector<SlotData>& slots) {
  std::vector<int64_t> lengths;
  for (const auto& slot : slots) {
    lengths.push_back(slot.values.numel());
  }
  return lengths;
}

at::Tensor concatenated(const std::vector<SlotData>& slots) {
  std::vector<at::Tensor> parts;
  for (const auto& slot : slots) {
    parts.push_back(slot.values);
  }
  return at::cat(parts);
}

// Every slot comes back exactly, at the width its range needs, on one thread
// and on several, with slots split across threads.
TEST_F(PackedInputTest, encodeRoundTrip) {
  at::manual_seed(1);
  const auto slots = makeSlots();
  const auto recipe = recipeFor(slots);
  const auto values = concatenated(slots);
  int32_t expectedByWidth[5]{};
  for (const auto& slot : slots) {
    const int32_t index = slot.bits == 0 ? 0
        : slot.bits == 8                 ? 1
        : slot.bits == 16                ? 2
        : slot.bits == 32                ? 3
                                         : 4;
    ++expectedByWidth[index];
  }
  for (const auto numThreads : {1, 4}) {
    SCOPED_TRACE(fmt::format("{} threads", numThreads));
    PackStats stats;
    const auto packed = encodePacked(
        recipe,
        values,
        lengthsOf(slots),
        {.numThreads = numThreads, .chunkValues = 1'000},
        &stats);
    EXPECT_TRUE(at::equal(decodePacked(packed), values));
    EXPECT_THAT(stats.slotsByWidth, testing::ElementsAreArray(expectedByWidth));
    EXPECT_LT(stats.packedBytes, stats.inputBytes / 2);
  }
}

// A slot whose every use is a remainder is stored reduced, so it needs only
// the width of the divisor.
TEST_F(PackedInputTest, reducesSlotWithModulus) {
  at::manual_seed(2);
  const auto values =
      at::randint(-(int64_t{1} << 60), int64_t{1} << 60, {6'000}, at::kLong);
  PackRecipe recipe;
  recipe.inputName = "values";
  recipe.slots.push_back({.constLength = values.numel(), .modulus = 1'000});
  PackStats stats;
  const auto packed =
      encodePacked(recipe, values, {values.numel()}, {}, &stats);
  EXPECT_TRUE(at::equal(decodePacked(packed), at::remainder(values, 1'000)));
  EXPECT_EQ(stats.slotsByWidth[2], 1);
}

// Four features of 50 rows with windows, laid out as a KeyedJaggedTensor
// flattens them: values back to back, an int32 [feature, row] lengths input,
// and one length input per feature. Features 0-2 form a guard group: each row
// is first cut to the group's shortest, then feature 0 keeps the first 3 of
// what is left and feature 1 the last 2. Feature 3 has no window. The trimmed
// values, lengths and totals come back from the encoding as the reference
// arithmetic below computes them.
TEST_F(PackedInputTest, trimsWindowedSlots) {
  constexpr int64_t kNumFeatures = 4;
  constexpr int64_t kNumRows = 50;
  at::manual_seed(4);
  const auto lengths =
      at::randint(0, 9, {kNumFeatures, kNumRows}, at::kInt).contiguous();
  std::vector<at::Tensor> features;
  for (int64_t f = 0; f < kNumFeatures; ++f) {
    features.push_back(
        at::randint(
            f * 1'000,
            f * 1'000 + 100,
            {lengths[f].sum().item<int64_t>()},
            at::kLong));
  }

  PackRecipe recipe;
  recipe.inputName = "values";
  recipe.inputIndex = 0;
  recipe.lengthsInput = 1;
  recipe.lengthsRows = kNumFeatures;
  for (int32_t f = 0; f < kNumFeatures; ++f) {
    recipe.slots.push_back({.lengthInput = 2 + f, .lengthsRow = f});
  }
  for (int32_t f = 0; f < 3; ++f) {
    recipe.slots[f].guardGroup = 0;
  }
  recipe.slots[0].maxLength = 3;
  recipe.slots[1].maxLength = 2;
  recipe.slots[1].keepTail = true;

  // Each row of feature f keeps [begin, end) of itself.
  std::vector<std::vector<int64_t>> expectedValues(kNumFeatures);
  auto expectedLengths = lengths.clone();
  int64_t numDropped{0};
  std::vector<int64_t> rowStarts(kNumFeatures, 0);
  for (int64_t r = 0; r < kNumRows; ++r) {
    int64_t groupMin = std::numeric_limits<int64_t>::max();
    for (int64_t f = 0; f < 3; ++f) {
      groupMin = std::min<int64_t>(groupMin, lengths[f][r].item<int32_t>());
    }
    for (int64_t f = 0; f < kNumFeatures; ++f) {
      const int64_t length = lengths[f][r].item<int32_t>();
      int64_t begin = 0;
      int64_t end = f < 3 ? groupMin : length;
      if (f == 0) {
        end = std::min<int64_t>(end, 3);
      } else if (f == 1) {
        begin = std::max<int64_t>(0, end - 2);
      }
      const auto* row = features[f].const_data_ptr<int64_t>() + rowStarts[f];
      expectedValues[f].insert(expectedValues[f].end(), row + begin, row + end);
      expectedLengths[f][r] = static_cast<int32_t>(end - begin);
      numDropped += length - (end - begin);
      rowStarts[f] += length;
    }
  }

  std::vector<c10::IValue> inputs{at::cat(features), lengths.flatten()};
  for (const auto& feature : features) {
    inputs.emplace_back(feature.numel());
  }
  for (const auto numThreads : {1, 3}) {
    SCOPED_TRACE(fmt::format("{} threads", numThreads));
    auto encoded = inputs;
    std::vector<PackStats> stats;
    encodePackedInputs({recipe}, encoded, {.numThreads = numThreads}, &stats);

    std::vector<at::Tensor> expectedParts;
    for (int64_t f = 0; f < kNumFeatures; ++f) {
      EXPECT_EQ(
          encoded[2 + f].toInt(),
          static_cast<int64_t>(expectedValues[f].size()));
      expectedParts.push_back(at::tensor(expectedValues[f], at::kLong));
    }
    EXPECT_TRUE(
        at::equal(decodePacked(encoded[0].toTensor()), at::cat(expectedParts)));
    EXPECT_TRUE(at::equal(encoded[1].toTensor(), expectedLengths.flatten()));
    EXPECT_EQ(stats[0].droppedValues, numDropped);
  }
}

// 'values' cut into segments at 'cuts' (positions within it, ascending).
std::vector<SlotSegment> segmentsOf(
    const at::Tensor& values,
    const std::vector<int64_t>& cuts) {
  std::vector<SlotSegment> segments;
  int64_t begin{0};
  for (auto cut : cuts) {
    segments.push_back({values.const_data_ptr<int64_t>() + begin, cut - begin});
    begin = cut;
  }
  segments.push_back(
      {values.const_data_ptr<int64_t>() + begin, values.numel() - begin});
  return segments;
}

// Each slot cut into uneven segments, some of them empty, encodes to the same
// words as the slots back to back.
TEST_F(PackedInputTest, encodesSegments) {
  at::manual_seed(5);
  const auto slots = makeSlots();
  const auto recipe = recipeFor(slots);
  std::vector<std::vector<SlotSegment>> segments;
  for (const auto& slot : slots) {
    const auto n = slot.values.numel();
    segments.push_back(segmentsOf(slot.values, {n / 3, n / 3, n / 2, n}));
  }
  for (const auto numThreads : {1, 4}) {
    SCOPED_TRACE(fmt::format("{} threads", numThreads));
    const PackOptions options{.numThreads = numThreads, .chunkValues = 700};
    const auto packed = PackEncoder(recipe, options).encode(segments).packed;
    EXPECT_TRUE(
        at::equal(
            packed,
            encodePacked(
                recipe, concatenated(slots), lengthsOf(slots), options)));
  }
}

// A slot's values for one batch, drawn from [low, high).
at::Tensor draw(int64_t count, int64_t low, int64_t high) {
  return at::randint(low, high, {count}, at::kLong);
}

// Three slots encoded batch after batch with kept frames: batches inside the
// first one's ranges reuse its frames; a batch that leaves a slot's frame
// re-encodes only that slot, in a widened frame, and moves the slots after it.
TEST_F(PackedInputTest, keepsFramesAndReencodes) {
  at::manual_seed(6);
  PackRecipe recipe;
  recipe.inputName = "values";
  for (int i = 0; i < 3; ++i) {
    recipe.slots.push_back({});
  }
  PackEncoder encoder(
      recipe,
      {.numThreads = 4,
       .writeThreads = 1,
       .chunkValues = 1'000,
       .keepFrames = true});
  auto encodeBatch = [&](std::vector<at::Tensor> batch, PackStats& stats) {
    std::vector<std::vector<SlotSegment>> segments;
    for (const auto& slot : batch) {
      const auto n = slot.numel();
      segments.push_back(segmentsOf(slot, {n / 4, n / 2}));
    }
    const auto packed = encoder.encode(segments, {}, &stats).packed;
    EXPECT_TRUE(at::equal(decodePacked(packed), at::cat(batch)));
  };

  PackStats stats;
  encodeBatch(
      {draw(5'000, 1'000, 1'200),
       draw(8'000, 0, 60'000),
       at::full({3'000}, 5, at::kLong)},
      stats);
  EXPECT_TRUE(stats.measured);
  const auto first = encoder.frames();
  EXPECT_EQ(first[0].bits, 8);
  EXPECT_EQ(first[1].bits, 16);
  EXPECT_EQ(first[2].bits, 0);

  // Inside the frames, and narrower than slot 1's: no re-encode, and the
  // effective range shows the room.
  encodeBatch(
      {draw(6'000, 1'050, 1'150),
       draw(7'000, 100, 300),
       at::full({2'000}, 5, at::kLong)},
      stats);
  EXPECT_FALSE(stats.measured);
  EXPECT_EQ(stats.reencodedSlots, 0);
  EXPECT_EQ(encoder.frames()[1].bits, 16);
  EXPECT_GE(encoder.ranges()[1].min, 100);
  EXPECT_LT(encoder.ranges()[1].max, 300);
  EXPECT_EQ(stats.widerThanNeeded, 1);
  EXPECT_LT(stats.neededDataBytes, stats.storedDataBytes);
  EXPECT_LE(stats.bitPackedDataBytes, stats.neededDataBytes);

  // Slot 0 leaves its 8-bit frame: re-encoded at 16 bits, its frame widened
  // below the new minimum; slots 1 and 2 move.
  encodeBatch(
      {draw(5'000, 1'000, 1'400),
       draw(8'000, 100, 60'000),
       at::full({3'000}, 5, at::kLong)},
      stats);
  EXPECT_EQ(stats.reencodedSlots, 1);
  EXPECT_GT(stats.rewrittenBytes, 0);
  const auto widened = encoder.frames();
  EXPECT_EQ(widened[0].bits, 16);
  EXPECT_LT(widened[0].base, encoder.ranges()[0].min);
  EXPECT_EQ(widened[1].bits, 16);

  // The constant slot changes: re-encoded too. Later batches in the widened
  // frames do not re-encode.
  encodeBatch(
      {draw(5'000, 1'000, 1'400), draw(8'000, 100, 60'000), draw(3'000, 5, 8)},
      stats);
  EXPECT_EQ(stats.reencodedSlots, 1);
  EXPECT_EQ(encoder.frames()[2].bits, 8);
  encodeBatch(
      {draw(5'000, 1'000, 1'400), draw(8'000, 100, 60'000), draw(3'000, 5, 8)},
      stats);
  EXPECT_EQ(stats.reencodedSlots, 0);
}

// Without headroom, a re-encode that widens a slot from 8 to 64 bits does not
// fit the buffer: it is reallocated, keeping what was written before the slot.
TEST_F(PackedInputTest, regrowsBufferOnReencode) {
  at::manual_seed(7);
  PackRecipe recipe;
  recipe.inputName = "values";
  recipe.slots.resize(2);
  PackEncoder encoder(recipe, {.keepFrames = true, .bufferHeadroom = 0.0});
  auto encodeBatch = [&](std::vector<at::Tensor> batch) {
    PackStats stats;
    std::vector<std::vector<SlotSegment>> segments;
    for (const auto& slot : batch) {
      segments.push_back(segmentsOf(slot, {}));
    }
    EXPECT_TRUE(
        at::equal(
            decodePacked(encoder.encode(segments, {}, &stats).packed),
            at::cat(batch)));
    return stats;
  };
  encodeBatch({draw(4'000, 0, 100), draw(4'000, 0, 100)});
  auto stats = encodeBatch(
      {draw(4'000, 0, 100),
       draw(4'000, -(int64_t{1} << 62), int64_t{1} << 62)});
  EXPECT_EQ(stats.reencodedSlots, 1);
  EXPECT_EQ(stats.regrownBuffers, 1);
  EXPECT_EQ(encoder.frames()[1].bits, 64);
}

// Windows applied to segments cut at arbitrary points, mid-row included, give
// the same words and lengths as on the slots back to back.
TEST_F(PackedInputTest, trimsWindowsOnSegments) {
  constexpr int64_t kNumRows = 40;
  at::manual_seed(8);
  const auto lengths = at::randint(0, 9, {2, kNumRows}, at::kInt).contiguous();
  std::vector<at::Tensor> features;
  for (int64_t f = 0; f < 2; ++f) {
    features.push_back(
        draw(lengths[f].sum().item<int64_t>(), f * 500, f * 500 + 200));
  }
  PackRecipe recipe;
  recipe.inputName = "values";
  recipe.inputIndex = 0;
  recipe.lengthsInput = 1;
  recipe.lengthsRows = 2;
  for (int32_t f = 0; f < 2; ++f) {
    recipe.slots.push_back(
        {.lengthInput = 2 + f, .lengthsRow = f, .guardGroup = 0});
  }
  recipe.slots[0].maxLength = 2;
  recipe.slots[0].keepTail = true;

  std::vector<c10::IValue> inputs{at::cat(features), lengths.flatten()};
  for (const auto& feature : features) {
    inputs.emplace_back(feature.numel());
  }
  auto expected = inputs;
  encodePackedInputs({recipe}, expected, {}, nullptr);

  std::vector<std::vector<SlotSegment>> segments;
  for (const auto& feature : features) {
    const auto n = feature.numel();
    segments.push_back(segmentsOf(feature, {n / 5, n / 2 + 1, n - 3}));
  }
  const auto result = PackEncoder(recipe, {.numThreads = 2})
                          .encode(segments, lengths.flatten());
  EXPECT_TRUE(at::equal(result.packed, expected[0].toTensor()));
  EXPECT_TRUE(
      at::equal(result.rowLengths.to(at::kInt), expected[1].toTensor()));
  EXPECT_EQ(result.slotLengths[0], expected[2].toInt());
  EXPECT_EQ(result.slotLengths[1], expected[3].toInt());
}

// Several threads encode with one encoder, some of their batches leaving the
// kept frames: every batch decodes exactly.
TEST_F(PackedInputTest, encodesConcurrently) {
  at::manual_seed(9);
  PackRecipe recipe;
  recipe.inputName = "values";
  recipe.slots.resize(3);
  PackEncoder encoder(
      recipe,
      {.numThreads = 2,
       .writeThreads = 1,
       .chunkValues = 500,
       .keepFrames = true});
  constexpr int kNumThreads = 4;
  constexpr int kBatches = 6;
  std::vector<std::vector<std::vector<at::Tensor>>> batches(kNumThreads);
  for (int t = 0; t < kNumThreads; ++t) {
    for (int b = 0; b < kBatches; ++b) {
      const int64_t spread = int64_t{100} << (2 * ((t + b) % 5));
      batches[t].push_back(
          {draw(3'000, 0, spread), draw(2'000, -spread, 0), draw(1'000, 7, 9)});
    }
  }
  std::vector<int> mismatches(kNumThreads, 0);
  std::vector<std::thread> threads;
  for (int t = 0; t < kNumThreads; ++t) {
    threads.emplace_back([&, t]() {
      for (const auto& batch : batches[t]) {
        std::vector<std::vector<SlotSegment>> segments;
        for (const auto& slot : batch) {
          segments.push_back(segmentsOf(slot, {slot.numel() / 2}));
        }
        const auto packed = encoder.encode(segments).packed;
        mismatches[t] += !at::equal(decodePacked(packed), at::cat(batch));
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  EXPECT_THAT(mismatches, testing::Each(0));
}

// With a hash size, the values are stored as they are and decode to
// torch.fmod(values, hashSize), the hashing the trainer applies to unpacked
// ids, negative ones included. A hash set after the encoding decodes the same.
TEST_F(PackedInputTest, hashesValues) {
  at::manual_seed(10);
  PackRecipe recipe;
  recipe.inputName = "values";
  recipe.slots.resize(2);
  const auto first = draw(3'000, -(int64_t{1} << 40), int64_t{1} << 40);
  const auto second = draw(2'000, 0, 50'000);
  const std::vector<std::vector<SlotSegment>> segments{
      segmentsOf(first, {1'000}), segmentsOf(second, {})};
  constexpr int64_t kHash = 100'000;
  const auto expected = at::fmod(at::cat({first, second}), kHash);

  PackStats stats;
  const auto packed = PackEncoder(recipe, {.hashSize = kHash})
                          .encode(segments, at::Tensor(), &stats)
                          .packed;
  EXPECT_TRUE(at::equal(decodePacked(packed), expected));
  // Unhashed, the first slot needs 64 bits.
  EXPECT_EQ(stats.slotsByWidth[4], 1);
  EXPECT_EQ(stats.slotsByWidth[2], 1);

  auto late = PackEncoder(recipe, {.reduce = false}).encode(segments).packed;
  setPackedHash(late, kHash);
  EXPECT_TRUE(at::equal(decodePacked(late), expected));
}

// A slot stored reduced by its modulus cannot take a hash later: its
// remainder would come before the hash.
TEST_F(PackedInputTest, refusesLateHashOfReducedSlot) {
  const auto values = draw(1'000, -(int64_t{1} << 40), int64_t{1} << 40);
  PackRecipe recipe;
  recipe.inputName = "values";
  recipe.slots.push_back({.constLength = values.numel(), .modulus = 1'000});
  auto packed = encodePacked(recipe, values, {values.numel()}, {}, nullptr);
  EXPECT_THROW(setPackedHash(packed, 100'000), c10::Error);
  setPackedHash(packed, 0);
  EXPECT_TRUE(at::equal(decodePacked(packed), at::remainder(values, 1'000)));
}

// The divide-free remainder agrees with torch.remainder over the whole int64
// range and odd divisors, in blocks that need it and blocks that do not. With
// a hash, a slot with a modulus is stored unreduced and decodes to the hash.
TEST_F(PackedInputTest, hashesAndReducesFullRange) {
  at::manual_seed(11);
  constexpr auto kMin = std::numeric_limits<int64_t>::min();
  constexpr auto kMax = std::numeric_limits<int64_t>::max();
  auto edges = at::tensor(
      std::vector<int64_t>{kMin, kMin + 1, -1, 0, 1, kMax - 1, kMax},
      at::kLong);
  // 2'000 values already in range, then the full range.
  auto values = at::cat(
      {draw(2'000, 0, 900),
       edges,
       draw(5'000, -(int64_t{1} << 62), int64_t{1} << 62) * 2 +
           draw(5'000, 0, 2),
       draw(3'000, -(int64_t{1} << 40), int64_t{1} << 40)});
  for (const int64_t hash :
       {int64_t{1},
        int64_t{3},
        int64_t{100'000},
        (int64_t{1} << 62) + 1,
        kMax}) {
    PackRecipe recipe;
    recipe.inputName = "values";
    recipe.slots.push_back({.constLength = values.numel(), .modulus = 1'000});
    const auto packed = encodePacked(
        recipe, values, {values.numel()}, {.hashSize = hash}, nullptr);
    EXPECT_TRUE(at::equal(decodePacked(packed), at::fmod(values, hash)))
        << "hash " << hash;
  }
  for (const int64_t modulus :
       {int64_t{1},
        int64_t{3},
        int64_t{7},
        int64_t{1'000},
        (int64_t{1} << 61) - 1,
        kMax}) {
    PackRecipe recipe;
    recipe.inputName = "values";
    recipe.slots.push_back({.constLength = values.numel(), .modulus = modulus});
    const auto packed =
        encodePacked(recipe, values, {values.numel()}, {}, nullptr);
    EXPECT_TRUE(at::equal(decodePacked(packed), at::remainder(values, modulus)))
        << "modulus " << modulus;
  }
}

// Graph meta: every tensor value is int64; 'parts' is the split's list.
std::unordered_map<std::string, torch::_export::TensorMeta> int64Meta(
    const std::vector<std::string>& names) {
  std::unordered_map<std::string, torch::_export::TensorMeta> meta;
  for (const auto& name : names) {
    torch::_export::TensorMeta tensorMeta;
    tensorMeta.set_dtype(torch::_export::ScalarType::LONG);
    tensorMeta.set_layout(torch::_export::Layout::Strided);
    tensorMeta.set_requires_grad(false);
    torch::_export::Device device;
    device.set_type("cuda");
    device.set_index(0);
    tensorMeta.set_device(std::move(device));
    torch::_export::SymInt zero;
    zero.set_as_int(0);
    tensorMeta.set_storage_offset(std::move(zero));
    std::vector<torch::_export::SymInt> sizes(1);
    sizes[0].set_as_int(1);
    tensorMeta.set_sizes(std::move(sizes));
    meta[name] = std::move(tensorMeta);
  }
  return meta;
}

// Runs 'graph' on the GPU with 'inputs' in user-input order, after encoding
// the inputs of 'recipes'.
// 'lateHash' is set on the encoded inputs after the encoding, as a trainer
// does with a batch's max_hash_size.
std::vector<at::Tensor> runPackedGraph(
    std::unique_ptr<nativert::Graph> graph,
    const std::vector<PackRecipe>& recipes,
    std::vector<c10::IValue> inputs,
    PackOptions options = {.numThreads = 2},
    int64_t lateHash = 0) {
  setGraphDevice(graph.get(), /*isCuda=*/true);
  auto context = std::make_unique<ModelContext>();
  context->weights = std::make_shared<nativert::Weights>(graph.get());
  context->graph = std::move(graph);
  WaveGraphExecutor executor(std::move(context));

  const auto& runGraph = executor.graph();
  const auto& inputNames = runGraph.signature().userInputs();
  encodePackedInputs(recipes, inputs, options, nullptr);
  if (lateHash > 0) {
    for (const auto& recipe : recipes) {
      for (size_t i = 0; i < inputs.size(); ++i) {
        if (inputNames[i] == recipe.inputName) {
          auto packed = inputs[i].toTensor();
          setPackedHash(packed, lateHash);
        }
      }
    }
  }
  auto [deviceInputs, transferUs] = inputsToDevice(inputs);
  auto frame = executor.getFrame();
  for (size_t i = 0; i < deviceInputs.size(); ++i) {
    frame->setIValue(runGraph.getValue(inputNames[i])->id(), deviceInputs[i]);
  }
  auto outputs = executor.executeWithPrefilledFrame(*frame);
  executor.returnFrame(std::move(frame));
  std::vector<at::Tensor> hostOutputs;
  for (auto& output : outputsToHost(outputs, "packed")) {
    hostOutputs.push_back(output.toTensor());
  }
  return hostOutputs;
}

int32_t countTarget(const nativert::Graph& graph, std::string_view target) {
  int32_t count{0};
  for (const auto& node : graph.nodes()) {
    count += node.target() == target;
  }
  return count;
}

// The id lists of seven features arrive concatenated in 'v' and split apart.
// Each feature takes its own arithmetic: two concatenated, one never read, one
// reduced. The split becomes per-feature decodes of the encoded 'v', and the
// results match the same arithmetic on the plain values.
TEST_F(PackedInputTest, decodesSplitOfPackedInput) {
  const char* kGraph = R"(graph(%v, %w):
%parts[] = torch.ops.aten.split_with_sizes.default(self=%v, split_sizes=[9000, 20000, 6000, 5000, 500, 4000, 3000], dim=0)
%p0, %p1, %p2, %p3, %p4, %p5, %p6 = prim.ListUnpack(input=%parts)
%a = torch.ops.aten.add.Scalar(self=%p0, other=3, alpha=1)
%b = torch.ops.aten.mul.Scalar(self=%p1, other=2)
%c = torch.ops.aten.remainder.Scalar(self=%p2, other=1000)
%d = torch.ops.aten.sub.Tensor(self=%p3, other=%w, alpha=1)
%list[] = prim.ListPack(l0=%p4, l1=%p5)
%e = torch.ops.aten.cat.default(tensors=%list, dim=0)
return(%a, %b, %c, %d, %e)
)";
  const std::vector<std::string> tensorNames{
      "v",
      "w",
      "p0",
      "p1",
      "p2",
      "p3",
      "p4",
      "p5",
      "p6",
      "a",
      "b",
      "c",
      "d",
      "e"};
  at::manual_seed(3);
  auto uniform = [](int64_t count, int64_t low, int64_t high) {
    return at::randint(low, high, {count}, at::kLong);
  };
  const std::vector<at::Tensor> features{
      uniform(9'000, 1'000'000, 1'000'200),
      uniform(20'000, -30'000, 30'000),
      uniform(6'000, -(int64_t{1} << 60), int64_t{1} << 60),
      uniform(5'000, -(int64_t{1} << 31), int64_t{1} << 31),
      at::full({500}, 7, at::kLong),
      at::zeros({4'000}, at::kLong),
      uniform(3'000, 0, int64_t{1} << 62),
  };
  const auto v = at::cat(features);
  const auto w = uniform(5'000, 0, 100);
  const std::vector<at::Tensor> expected{
      features[0] + 3,
      features[1] * 2,
      at::remainder(features[2], 1'000),
      features[3] - w,
      at::cat({features[4], features[5]}),
  };

  auto& config = WaveConfig::get();
  const auto savedCg = config.isCg;
  auto resetConfig = folly::makeGuard([&, savedCg] { config.isCg = savedCg; });
  for (const auto cg :
       {std::optional<bool>(),
        std::optional<bool>(false),
        std::optional<bool>(true)}) {
    SCOPED_TRACE(cg.has_value() ? (*cg ? "cg 1" : "cg 0") : "cg -1");
    config.isCg = cg;
    auto graph = nativert::stringToGraph(kGraph);
    graph->setTensorValuesMeta(int64Meta(tensorNames));
    const auto recipes = rewritePackedInputs(*graph);

    ASSERT_EQ(recipes.size(), 1);
    EXPECT_EQ(recipes[0].inputName, "v");
    std::vector<bool> used;
    std::vector<int64_t> moduli;
    for (const auto& slot : recipes[0].slots) {
      used.push_back(slot.used);
      moduli.push_back(slot.modulus);
    }
    EXPECT_THAT(used, testing::ElementsAre(1, 1, 1, 1, 1, 1, 0));
    EXPECT_THAT(moduli, testing::ElementsAre(0, 0, 1'000, 0, 0, 0, 0));
    EXPECT_EQ(
        countTarget(*graph, "torch.ops.aten.split_with_sizes.default"), 0);
    EXPECT_EQ(countTarget(*graph, "tw.unpack_slot"), 6);

    const auto outputs = runPackedGraph(std::move(graph), recipes, {v, w});

    ASSERT_EQ(outputs.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
      EXPECT_TRUE(tensorsMatch(outputs[i], expected[i]))
          << "output " << i << ": " << firstDifference(outputs[i], expected[i]);
    }
  }
}

// With a hash, the device decodes each slot to torch.fmod(value, hash) before
// the graph's arithmetic, whether the hash came with the encoding or was set
// on the encoded tensor afterwards. The slot with a modulus is stored
// unreduced and the graph takes its remainder after the hash. One feature
// holds the int64 extremes.
TEST_F(PackedInputTest, decodesHashedSlots) {
  const char* kGraph = R"(graph(%v):
%parts[] = torch.ops.aten.split_with_sizes.default(self=%v, split_sizes=[4000, 3000, 2007], dim=0)
%p0, %p1, %p2 = prim.ListUnpack(input=%parts)
%a = torch.ops.aten.add.Scalar(self=%p0, other=3, alpha=1)
%b = torch.ops.aten.remainder.Scalar(self=%p1, other=1000)
%c = torch.ops.aten.mul.Scalar(self=%p2, other=2)
return(%a, %b, %c)
)";
  const std::vector<std::string> tensorNames{
      "v", "p0", "p1", "p2", "a", "b", "c"};
  at::manual_seed(12);
  constexpr auto kMin = std::numeric_limits<int64_t>::min();
  constexpr auto kMax = std::numeric_limits<int64_t>::max();
  const std::vector<at::Tensor> features{
      draw(4'000, -(int64_t{1} << 60), int64_t{1} << 60),
      draw(3'000, -(int64_t{1} << 40), int64_t{1} << 40),
      at::cat(
          {at::tensor(
               std::vector<int64_t>{kMin, kMin + 1, -1, 0, 1, kMax - 1, kMax},
               at::kLong),
           draw(2'000, -(int64_t{1} << 62), int64_t{1} << 62) * 2}),
  };
  const auto v = at::cat(features);
  for (const int64_t hash : {int64_t{100'000}, (int64_t{1} << 62) + 1}) {
    std::vector<at::Tensor> hashed;
    for (const auto& feature : features) {
      hashed.push_back(at::fmod(feature, hash));
    }
    const std::vector<at::Tensor> expected{
        hashed[0] + 3, at::remainder(hashed[1], 1'000), hashed[2] * 2};
    for (const bool late : {false, true}) {
      SCOPED_TRACE(fmt::format("hash {} late {}", hash, late));
      auto graph = nativert::stringToGraph(kGraph);
      graph->setTensorValuesMeta(int64Meta(tensorNames));
      const auto recipes = rewritePackedInputs(*graph);
      ASSERT_EQ(recipes.size(), 1);
      EXPECT_EQ(recipes[0].slots[1].modulus, 1'000);

      const auto outputs = late ? runPackedGraph(
                                      std::move(graph),
                                      recipes,
                                      {v},
                                      {.numThreads = 2, .reduce = false},
                                      hash)
                                : runPackedGraph(
                                      std::move(graph),
                                      recipes,
                                      {v},
                                      {.numThreads = 2, .hashSize = hash});

      ASSERT_EQ(outputs.size(), expected.size());
      for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_TRUE(tensorsMatch(outputs[i], expected[i]))
            << "output " << i << ": "
            << firstDifference(outputs[i], expected[i]);
      }
    }
  }
}

// An input that something besides its split reads is not packed: the graph
// keeps its split.
TEST_F(PackedInputTest, keepsInputReadElsewhere) {
  const char* kGraph = R"(graph(%v):
%parts[] = torch.ops.aten.split_with_sizes.default(self=%v, split_sizes=[3, 5], dim=0)
%p0, %p1 = prim.ListUnpack(input=%parts)
%a = torch.ops.aten.add.Tensor(self=%p0, other=%p0, alpha=1)
%b = torch.ops.aten.mul.Scalar(self=%v, other=2)
return(%a, %p1, %b)
)";
  auto graph = nativert::stringToGraph(kGraph);
  graph->setTensorValuesMeta(int64Meta({"v", "p0", "p1", "a", "b"}));

  EXPECT_TRUE(rewritePackedInputs(*graph).empty());
  EXPECT_EQ(countTarget(*graph, "torch.ops.aten.split_with_sizes.default"), 1);
  EXPECT_EQ(countTarget(*graph, "tw.unpack_slot"), 0);
}

} // namespace
} // namespace torch::wave

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  folly::Init init{&argc, &argv};
  if (int device; cudaGetDevice(&device) != cudaSuccess) {
    LOG(WARNING) << "No CUDA detected, skipping all tests";
    return 0;
  }
  return RUN_ALL_TESTS();
}
