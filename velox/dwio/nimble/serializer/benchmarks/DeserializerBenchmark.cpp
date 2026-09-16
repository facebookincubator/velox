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

#include <folly/Benchmark.h>
#include <folly/init/Init.h>

#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/serializer/Deserializer.h"
#include "velox/dwio/nimble/serializer/Serializer.h"
#include "velox/dwio/nimble/velox/OrderedRanges.h"
#include "velox/dwio/nimble/velox/SchemaReader.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/FlatVector.h"

namespace facebook::nimble {
namespace {

constexpr velox::vector_size_t kNumRows = 10'000;
constexpr size_t kNumColumns = 100;
constexpr size_t kProjectionStride = 20;

struct BenchmarkState {
  std::string serialized;
  std::shared_ptr<const Type> schema;
  std::vector<Deserializer::Subfield> selectedSubfields;
};

BenchmarkState prepareBenchmark() {
  auto pool = velox::memory::memoryManager()->addLeafPool("prepare_projection");
  std::vector<std::string> names;
  std::vector<velox::TypePtr> types;
  std::vector<velox::VectorPtr> children;
  names.reserve(kNumColumns);
  types.reserve(kNumColumns);
  children.reserve(kNumColumns);

  for (size_t column = 0; column < kNumColumns; ++column) {
    names.emplace_back("column_" + std::to_string(column));
    types.emplace_back(velox::BIGINT());
    auto child = velox::BaseVector::create<velox::FlatVector<int64_t>>(
        velox::BIGINT(), kNumRows, pool.get());
    for (velox::vector_size_t row = 0; row < kNumRows; ++row) {
      child->set(row, static_cast<int64_t>(column * kNumRows + row));
    }
    children.emplace_back(std::move(child));
  }

  auto input = std::make_shared<velox::RowVector>(
      pool.get(),
      velox::ROW(std::move(names), std::move(types)),
      /*nulls=*/nullptr,
      kNumRows,
      std::move(children));
  Serializer serializer{
      SerializerOptions{.version = SerializationVersion::kSerialization},
      input->type(),
      pool.get()};
  const auto serialized =
      serializer.serialize(input, OrderedRanges::of(0, input->size()));

  BenchmarkState state{
      .serialized = std::string{serialized},
      .schema =
          SchemaReader::getSchema(serializer.schemaBuilder().schemaNodes())};
  for (size_t column = 0; column < kNumColumns; column += kProjectionStride) {
    state.selectedSubfields.emplace_back("column_" + std::to_string(column));
  }
  return state;
}

void constructProjectedDeserializer(uint32_t iters) {
  static const auto state = prepareBenchmark();
  auto pool =
      velox::memory::memoryManager()->addLeafPool("construct_projection");
  while (iters-- > 0) {
    Deserializer deserializer{
        state.schema,
        state.selectedSubfields,
        pool.get(),
        DeserializerOptions{.hasHeader = true}};
    folly::doNotOptimizeAway(deserializer);
  }
}

void constructAndDeserialize(uint32_t iters) {
  static const auto state = prepareBenchmark();
  auto pool = velox::memory::memoryManager()->addLeafPool("decode_projection");
  while (iters-- > 0) {
    Deserializer deserializer{
        state.schema,
        state.selectedSubfields,
        pool.get(),
        DeserializerOptions{.hasHeader = true}};
    velox::VectorPtr output;
    deserializer.deserialize(state.serialized, output);
    folly::doNotOptimizeAway(output);
  }
}

BENCHMARK(ConstructDeserializerWithProjection, iters) {
  constructProjectedDeserializer(iters);
}

BENCHMARK_DRAW_LINE();

BENCHMARK(ConstructAndDeserializeWithProjection, iters) {
  constructAndDeserialize(iters);
}

} // namespace
} // namespace facebook::nimble

int main(int argc, char** argv) {
  folly::Init init{&argc, &argv};
  facebook::velox::memory::MemoryManager::initialize({});
  folly::runBenchmarks();
  return 0;
}
