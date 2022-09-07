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
#include "velox/exec/HashPartitionFunction.h"
#include "velox/exec/tests/PartitionAndSerialize.h"
#include "velox/exec/tests/ShuffleWrite.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/expression/VectorFunction.h"
#include "velox/serializers/UnsafeRowSerde.h"

using namespace facebook::velox;

namespace facebook::velox::exec::test {

namespace {
class PartitionFunction : public exec::VectorFunction {
 public:
  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& outputType,
      EvalCtx* context,
      VectorPtr* result) const override {
    // TODO Verify that partition count is constant.
    auto numPartitions =
        args[0]->as<SimpleVector<int32_t>>()->valueAt(rows.begin());

    auto rowType = makeRowType(args);

    auto argsCopy = args;
    auto input = std::make_shared<RowVector>(
        context->pool(), rowType, nullptr, rows.size(), std::move(argsCopy));

    std::vector<column_index_t> keyChannels(args.size() - 1);
    std::iota(keyChannels.begin(), keyChannels.end(), 1);

    auto partitionFunction = std::make_unique<HashPartitionFunction>(
        numPartitions, rowType, keyChannels);

    std::vector<uint32_t> partitions(rows.size());
    partitionFunction->partition(*input, partitions);

    context->ensureWritable(rows, INTEGER(), *result);
    auto flatVector = (*result)->asFlatVector<int32_t>();
    rows.applyToSelected(
        [&](auto row) { flatVector->set(row, partitions[row]); });
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {exec::FunctionSignatureBuilder()
                .argumentType("integer")
                .argumentType("any")
                .variableArity()
                .returnType("integer")
                .build()};
  }

 private:
  RowTypePtr makeRowType(const std::vector<VectorPtr>& args) const {
    std::vector<std::string> inputNames;
    std::vector<TypePtr> inputTypes;
    inputNames.reserve(args.size());
    inputTypes.reserve(args.size());

    for (auto i = 0; i < args.size(); ++i) {
      inputNames.push_back(fmt::format("c{}", i));
      inputTypes.push_back(args[i]->type());
    }

    return ROW(std::move(inputNames), std::move(inputTypes));
  }
};

class SerializeToUnsafeRowFunction : public exec::VectorFunction {
 public:
  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& outputType,
      EvalCtx* context,
      VectorPtr* result) const override {
    auto serde = std::make_unique<batch::UnsafeRowVectorSerde>(context->pool());

    context->ensureWritable(rows, VARBINARY(), *result);
    auto flatVector = (*result)->asFlatVector<StringView>();

    auto argsCopy = args;
    auto input = std::make_shared<RowVector>(
        context->pool(),
        makeRowType(args),
        nullptr,
        rows.size(),
        std::move(argsCopy));

    rows.applyToSelected([&](auto row) {
      // TODO Handle errors. Avoid extra copy.
      std::string_view unused;
      std::string_view serialized;
      serde->serializeRow(input, row, unused, serialized);
      flatVector->set(row, StringView(serialized));
    });
  }

  RowTypePtr makeRowType(const std::vector<VectorPtr>& args) const {
    std::vector<std::string> inputNames;
    std::vector<TypePtr> inputTypes;
    inputNames.reserve(args.size());
    inputTypes.reserve(args.size());

    for (auto i = 0; i < args.size(); ++i) {
      inputNames.push_back(fmt::format("c{}", i));
      inputTypes.push_back(args[i]->type());
    }

    return ROW(std::move(inputNames), std::move(inputTypes));
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {exec::FunctionSignatureBuilder()
                .argumentType("any")
                .variableArity()
                .returnType("varbinary")
                .build()};
  }
};

} // namespace

class SparkShuffleTest : public OperatorTestBase {
 protected:
  RowVectorPtr deserialize(
      const RowVectorPtr& serializedResult,
      const RowTypePtr& rowType) {
    auto serde = std::make_unique<batch::UnsafeRowVectorSerde>(pool());

    auto serializedData =
        serializedResult->childAt(1)->as<SimpleVector<StringView>>();

    std::vector<std::optional<std::string_view>> serializedStrings;
    serializedStrings.reserve(serializedData->size());
    for (auto i = 0; i < serializedData->size(); ++i) {
      auto value = serializedData->valueAt(i);
      serializedStrings.push_back(std::string_view(value.data(), value.size()));
    }

    RowVectorPtr result;
    serde->deserializeVector(serializedStrings, rowType, &result);
    return result;
  }
};

// Use custom functions to partition and serialize data.
TEST_F(SparkShuffleTest, functions) {
  exec::registerVectorFunction(
      "serialize_to_unsaferow",
      SerializeToUnsafeRowFunction::signatures(),
      std::make_unique<SerializeToUnsafeRowFunction>());

  exec::registerVectorFunction(
      "partition",
      PartitionFunction::signatures(),
      std::make_unique<PartitionFunction>());

  Operator::registerOperator(std::make_unique<ShuffleWriteTranslator>());

  auto data = makeRowVector({
      makeFlatVector<int32_t>({1, 2, 3, 4}),
      makeFlatVector<int64_t>({10, 20, 30, 40}),
  });

  auto plan =
      PlanBuilder()
          .values({data}, true)
          .project(
              {"partition(4::integer, c0)", "serialize_to_unsaferow(c0, c1)"})
          .localPartition({})
          .addNode(
              [](core::PlanNodeId nodeId,
                 core::PlanNodePtr source) -> core::PlanNodePtr {
                return std::make_shared<ShuffleWriteNode>(
                    nodeId, std::move(source));
              })
          .planNode();

  CursorParameters params;
  params.planNode = plan;
  params.maxDrivers = 2;

  auto [taskCursor, serializedResults] =
      readCursor(params, [](auto /*task*/) {});
  ASSERT_EQ(serializedResults.size(), 0);
}

TEST_F(SparkShuffleTest, functionAndSerializeFunctions) {
  exec::registerVectorFunction(
      "serialize_to_unsaferow",
      SerializeToUnsafeRowFunction::signatures(),
      std::make_unique<SerializeToUnsafeRowFunction>());

  exec::registerVectorFunction(
      "partition",
      PartitionFunction::signatures(),
      std::make_unique<PartitionFunction>());

  Operator::registerOperator(std::make_unique<ShuffleWriteTranslator>());

  auto data = makeRowVector({
      makeFlatVector<int32_t>({1, 2, 3, 4}),
      makeFlatVector<int64_t>({10, 20, 30, 40}),
  });

  auto plan =
      PlanBuilder()
          .values({data}, true)
          .project(
              {"partition(4::integer, c0)", "serialize_to_unsaferow(c0, c1)"})
          .planNode();

  CursorParameters params;
  params.planNode = plan;
  params.maxDrivers = 2;

  auto [taskCursor, serializedResults] =
      readCursor(params, [](auto /*task*/) {});
  ASSERT_EQ(serializedResults.size(), 2);

  for (auto& serializedResult : serializedResults) {
    // Print out partition numbers.
    std::cout << serializedResult->childAt(0)->toString(0, 100) << std::endl;

    // Verify that serialized data can be deserialized successfully into the
    // original data.
    auto deserialized = deserialize(serializedResult, asRowType(data->type()));
    velox::test::assertEqualVectors(data, deserialized);
  }
}

// Use custom operator to partition and serialize data.
TEST_F(SparkShuffleTest, operators) {
  Operator::registerOperator(
      std::make_unique<PartitionAndSerializeTranslator>());
  Operator::registerOperator(std::make_unique<ShuffleWriteTranslator>());

  auto data = makeRowVector({
      makeFlatVector<int32_t>({1, 2, 3, 4}),
      makeFlatVector<int64_t>({10, 20, 30, 40}),
  });

  auto plan =
      PlanBuilder()
          .values({data}, true)
          .addNode(
              [](core::PlanNodeId nodeId,
                 core::PlanNodePtr source) -> core::PlanNodePtr {
                auto outputType = ROW({"p", "d"}, {INTEGER(), VARBINARY()});

                std::vector<core::TypedExprPtr> keys;
                keys.push_back(std::make_shared<core::FieldAccessTypedExpr>(
                    INTEGER(), "c0"));

                return std::make_shared<PartitionAndSerializeNode>(
                    nodeId, keys, 4, outputType, std::move(source));
              })
          .localPartition({})
          .addNode(
              [](core::PlanNodeId nodeId,
                 core::PlanNodePtr source) -> core::PlanNodePtr {
                return std::make_shared<ShuffleWriteNode>(
                    nodeId, std::move(source));
              })
          .planNode();

  CursorParameters params;
  params.planNode = plan;
  params.maxDrivers = 2;

  auto [taskCursor, serializedResults] =
      readCursor(params, [](auto /*task*/) {});
  ASSERT_EQ(serializedResults.size(), 0);
}

TEST_F(SparkShuffleTest, partitionAndSerializeOperator) {
  Operator::registerOperator(
      std::make_unique<PartitionAndSerializeTranslator>());
  Operator::registerOperator(std::make_unique<ShuffleWriteTranslator>());

  auto data = makeRowVector({
      makeFlatVector<int32_t>(1'000, [](auto row) { return row; }),
      makeFlatVector<int64_t>(1'000, [](auto row) { return row * 10; }),
  });

  auto plan =
      PlanBuilder()
          .values({data}, true)
          .addNode(
              [](core::PlanNodeId nodeId,
                 core::PlanNodePtr source) -> core::PlanNodePtr {
                auto outputType = ROW({"p", "d"}, {INTEGER(), VARBINARY()});

                std::vector<core::TypedExprPtr> keys;
                keys.push_back(std::make_shared<core::FieldAccessTypedExpr>(
                    INTEGER(), "c0"));

                return std::make_shared<PartitionAndSerializeNode>(
                    nodeId, keys, 4, outputType, std::move(source));
              })
          .planNode();

  CursorParameters params;
  params.planNode = plan;
  params.maxDrivers = 2;

  auto [taskCursor, serializedResults] =
      readCursor(params, [](auto /*task*/) {});
  ASSERT_EQ(serializedResults.size(), 2);

  for (auto& serializedResult : serializedResults) {
    // Print out partition numbers.
    std::cout << serializedResult->childAt(0)->toString(0, 100) << std::endl;

    // Verify that serialized data can be deserialized successfully into the
    // original data.
    auto deserialized = deserialize(serializedResult, asRowType(data->type()));
    velox::test::assertEqualVectors(data, deserialized);
  }
}
} // namespace facebook::velox::exec::test
