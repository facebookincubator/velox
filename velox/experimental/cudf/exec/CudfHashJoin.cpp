/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

// For custom hash join operator
#include "velox/exec/JoinBridge.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

#include <cudf/join.hpp>
#include <cudf/copying.hpp>

#include "VeloxCudfInterop.hpp"
#include "CudfHashJoin.hpp"
#include <nvtx3/nvtx3.hpp>

namespace facebook::velox::cudf_velox {

// Custom hash join operator which uses libcudf
// need to define a new PlanNode, JoinBridge, Operators (Build, Probe), and a PlanNodeTranslator
// and Register the PlanNodeTranslator
class CudfHashJoinNode : public core::PlanNode {
public:
    CudfHashJoinNode(const core::PlanNodeId& id,
    core::PlanNodePtr left,
    core::PlanNodePtr right)
    : PlanNode(id), sources_{std::move(left), std::move(right)} {}
    // 4 abstract functions to implement
    const RowTypePtr& outputType() const override {
        // TODO similar to PlanBuilder::hashJoin()
        return sources_.front()->outputType();
    }
    const std::vector<core::PlanNodePtr>& sources() const override {
        return sources_;
    }
    std::string_view name() const override {
        return "cudf hash join";
    }
private:
    void addDetails(std::stringstream& /* stream */) const override {}
    std::vector<core::PlanNodePtr> sources_;
};

class CudfHashJoinBridge : public exec::JoinBridge {
    public:
    using HashType = std::pair<std::unique_ptr<cudf::table>, std::shared_ptr<cudf::hash_join>>;
    // using HashType = int;
    void setHashTable(std::optional<HashType> hashObject) {
      std::vector<ContinuePromise> promises;
      {
        std::lock_guard<std::mutex> l(mutex_);
        VELOX_CHECK(!hashObject_.has_value(), "HashJoinBridge already has a hash table");
        hashObject_ = std::move(hashObject);
        promises = std::move(promises_);
      }
      notify(std::move(promises));
    }

    std::optional<HashType> HashOrFuture(ContinueFuture* future) {
        std::lock_guard<std::mutex> l(mutex_);
        if (hashObject_.has_value()) {
            return std::move(hashObject_);
        }
        promises_.emplace_back("CudfHashJoinBridge::HashOrFuture");
        *future = promises_.back().getSemiFuture();
        return std::nullopt;
    }

private:
  std::optional<HashType> hashObject_;
};

class CudfHashJoinBuild : public exec::Operator {
public:
  CudfHashJoinBuild(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    std::shared_ptr<const CudfHashJoinNode> joinNode)
    // TODO check outputType should be set or not?
    : exec::Operator(driverCtx,  nullptr, // joinNode->sources(),
    operatorId, joinNode->id(), "CudfHashJoinBuild") {}

    void addInput(RowVectorPtr input) override {
        // Queue inputs, process all at once.
        // TODO distribute work equally.
        auto inputSize = input->size();
        if (inputSize > 0) {
            inputs_.push_back(std::move(input));
        }
    }
    bool needsInput() const override {
    return !noMoreInput_;
    }
    RowVectorPtr getOutput() override {
        return nullptr;
    }
    void noMoreInput() override {
        NVTX3_FUNC_RANGE();
        Operator::noMoreInput();
        // TODO
        std::vector<ContinuePromise> promises;
        std::vector<std::shared_ptr<exec::Driver>> peers;
        // Only last driver collects all answers
        if (!operatorCtx_->task()->allPeersFinished(
            planNodeId(), operatorCtx_->driver(), &future_, promises, peers)) {
            return;
        }
        // Collect results from peers
        for (auto& peer : peers) {
            auto op = peer->findOperator(planNodeId());
            auto* build = dynamic_cast<CudfHashJoinBuild*>(op);
            VELOX_CHECK(build);
            // numRows_ += build->numRows_;
            inputs_.insert(inputs_.end(), build->inputs_.begin(), build->inputs_.end());
        }
        // TODO build hash table
        auto tbl = to_cudf_table(inputs_[0]); // TODO how to process multiple inputs?
        // copy host to device table,
        // CudfHashJoinBridge::HashType hashObject = 1;
        // TODO create hash table in device.
        // CudfHashJoinBridge::HashType
        auto hashObject =
         std::make_shared<cudf::hash_join>(tbl->view(), cudf::null_equality::EQUAL);

        // Copied
        peers.clear();
        for (auto& promise : promises) {
            promise.setValue();
        }

        // set hash table to CudfHashJoinBridge
        auto joinBridge = operatorCtx_->task()->getCustomJoinBridge(
            operatorCtx_->driverCtx()->splitGroupId, planNodeId());
        auto cudf_HashJoinBridge =
            std::dynamic_pointer_cast<CudfHashJoinBridge>(joinBridge);
        cudf_HashJoinBridge->setHashTable(std::make_optional(std::make_pair(std::move(tbl), std::move(hashObject))));
    }

    exec::BlockingReason isBlocked(ContinueFuture* future) override {
        if (!future_.valid()) {
        return exec::BlockingReason::kNotBlocked;
        }
        *future = std::move(future_);
        return exec::BlockingReason::kWaitForJoinBuild;
    }

    bool isFinished() override {
        return !future_.valid() && noMoreInput_;
    }

private:
    std::vector<RowVectorPtr> inputs_;
    ContinueFuture future_{ContinueFuture::makeEmpty()};
};

class CudfHashJoinProbe : public exec::Operator {
    public:
    using HashType = CudfHashJoinBridge::HashType;
    CudfHashJoinProbe(
        int32_t operatorId,
        exec::DriverCtx* driverCtx,
        std::shared_ptr<const CudfHashJoinNode> joinNode)
        : exec::Operator(driverCtx, nullptr, // joinNode->sources(),
        operatorId, joinNode->id(), "CudfHashJoinProbe") {}

    bool needsInput() const override {
        return !finished_ && input_ == nullptr;
    }
    void addInput(RowVectorPtr input) override {
        input_ = std::move(input);
    }

    RowVectorPtr getOutput() override {
        NVTX3_FUNC_RANGE();
        if (!input_) {
            return nullptr;
        }
        const auto inputSize = input_->size();
        if(!hashObject_.has_value()) {
            return nullptr;
        }
        // std::cout<<"here\n\n";
        // TODO convert input to cudf table
        auto tbl = to_cudf_table(input_);
        // TODO pass the input pool !!!
        RowVectorPtr output;
        // RowVectorPtr output;
        auto const [left_join_indices, right_join_indices] = hashObject_.value().second->inner_join(tbl->view());
        auto left_indices_span  = cudf::device_span<cudf::size_type const>{*left_join_indices};
        auto right_indices_span = cudf::device_span<cudf::size_type const>{*right_join_indices};
        auto left_input  = tbl->view();
        auto right_input = hashObject_.value().first->view();

        auto left_indices_col  = cudf::column_view{left_indices_span};
        auto right_indices_col = cudf::column_view{right_indices_span};
        auto constexpr oob_policy = cudf::out_of_bounds_policy::DONT_CHECK;
        auto left_result  = cudf::gather(left_input, left_indices_col, oob_policy);
        auto right_result = cudf::gather(right_input, right_indices_col, oob_policy);
        auto joined_cols = left_result->release();
        auto right_cols  = right_result->release();
        joined_cols.insert(joined_cols.end(),
                            std::make_move_iterator(right_cols.begin()),
                            std::make_move_iterator(right_cols.end()));
        auto cudf_output = std::make_unique<cudf::table>(std::move(joined_cols));
        // TODO convert output to RowVector
        if (cudf_output->num_columns() == 0 or cudf_output->num_rows() == 0) {
            output = nullptr;
        } else {
            output = to_velox_column(cudf_output->view(), input_->pool());
        }
        // auto output = input_;
        // auto output = std::make_shared<RowVector>(
        //     input_->pool(),
        //     input_->type(),
        //     input_->nulls(),
        //     std::min(20, inputSize-2),
        //     input_->children());
        // std::cout<<"there\n\n";
        input_.reset();
        finished_ = true;
        // printResults(output, std::cout);
        return output;
    }

    exec::BlockingReason isBlocked(ContinueFuture* future) override {
        if (hashObject_.has_value()) {
        return exec::BlockingReason::kNotBlocked;
        }

        auto joinBridge = operatorCtx_->task()->getCustomJoinBridge(
            operatorCtx_->driverCtx()->splitGroupId, planNodeId());
        auto hashObject = std::dynamic_pointer_cast<CudfHashJoinBridge>(joinBridge)
                        ->HashOrFuture(future);

        if (!hashObject.has_value()) {
        return exec::BlockingReason::kWaitForJoinBuild;
        }
        hashObject_ = std::move(hashObject);
        // remainingLimit_ = hashObject.value();

        return exec::BlockingReason::kNotBlocked;
    }

    bool isFinished() override {
        return finished_ || (noMoreInput_ && input_ == nullptr);
    }

    private:
    std::optional<HashType> hashObject_;
    bool finished_{false};
};

class CudfHashJoinBridgeTranslator : public exec::Operator::PlanNodeTranslator {
  std::unique_ptr<exec::Operator>
  toOperator(exec::DriverCtx* ctx, int32_t id, const core::PlanNodePtr& node) {
    if (auto joinNode = std::dynamic_pointer_cast<const CudfHashJoinNode>(node)) {
      return std::make_unique<CudfHashJoinProbe>(id, ctx, joinNode);
    }
    return nullptr;
  }

  std::unique_ptr<exec::JoinBridge> toJoinBridge(const core::PlanNodePtr& node) {
    if (auto joinNode = std::dynamic_pointer_cast<const CudfHashJoinNode>(node)) {
      auto joinBridge = std::make_unique<CudfHashJoinBridge>();
      return joinBridge;
    }
    return nullptr;
  }

  exec::OperatorSupplier toOperatorSupplier(const core::PlanNodePtr& node) {
    if (auto joinNode = std::dynamic_pointer_cast<const CudfHashJoinNode>(node)) {
      return [joinNode](int32_t operatorId, exec::DriverCtx* ctx) {
        return std::make_unique<CudfHashJoinBuild>(operatorId, ctx, joinNode);
      };
    }
    return nullptr;
  }
};

/*

// few utility functions
std::vector<std::string> concat(
    const std::vector<std::string>& a,
    const std::vector<std::string>& b) {
  std::vector<std::string> result;
  result.insert(result.end(), a.begin(), a.end());
  result.insert(result.end(), b.begin(), b.end());
  return result;
}

// CudfHashJoinDemo class methods implementation
CudfHashJoinDemo::CudfHashJoinDemo() {
    // // Register Presto scalar functions.
    // functions::prestosql::registerAllScalarFunctions();

    // // Register Presto aggregate functions.
    // aggregate::prestosql::registerAllAggregateFunctions();

    // // Register type resolver with DuckDB SQL parser.
    // parse::registerTypeResolver();

    // Register custom Operator
    // Operator::registerOperator(std::make_unique<CustomJoinBridgeTranslator>());
    Operator::registerOperator(std::make_unique<CudfHashJoinBridgeTranslator>());
    }

RowVectorPtr CudfHashJoinDemo::makeSimpleRowVector(vector_size_t size, vector_size_t init, std::string name_prefix) {
    std::vector<VectorPtr> col_vec = {
        makeFlatVector<int64_t>(size, [init](auto row) { return init+row; })
        // ,makeFlatVector<int64_t>(size, [](auto row) { return row; })
        };
    std::vector<std::string> names;
    for (int32_t i = 0; i < col_vec.size(); ++i) {
        names.push_back(fmt::format("{}{}", name_prefix, i));
    }
    return makeRowVector(std::move(names), std::move(col_vec));
    }

CudfHashJoinDemo::result_type CudfHashJoinDemo::testVeloxHashJoin(
      int32_t numThreads,
      const std::vector<RowVectorPtr>& leftBatch, // probe input
      const std::vector<RowVectorPtr>& rightBatch, // build input
      const std::string& referenceQuery) {
    NVTX3_FUNC_RANGE();
    // createDuckDbTable("t", {leftBatch});
    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    CursorParameters params;
    params.maxDrivers = numThreads;
    params.planNode = PlanBuilder(planNodeIdGenerator)
                            .values(leftBatch, true)
                            .hashJoin(
                                {"c0"},
                                {"d0"},
                                PlanBuilder(planNodeIdGenerator)
                                    .values(rightBatch, true)
                                    .planNode(),
                                "",
                                // {"c0"})
                                concat({"c0"}, {"d0"}))
                            // .project({"c0"}) // project only first column
                            .planNode();
        auto result = readCursor(params, [](Task*) {});
        // std::cout<<"Velox Hash Join Result: \n";
        // printResults(result.second.front(), std::cout);
        return result;
      }

CudfHashJoinDemo::result_type CudfHashJoinDemo::testCudfHashJoin(
      int32_t numThreads,
      const std::vector<RowVectorPtr>& leftBatch, // probe input
      const std::vector<RowVectorPtr>& rightBatch, // build input
      const std::string& referenceQuery) {
    NVTX3_FUNC_RANGE();
    // createDuckDbTable("t", {leftBatch});

    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    auto leftNode =
        PlanBuilder(planNodeIdGenerator).values({leftBatch}, true).planNode();
    auto rightNode =
        PlanBuilder(planNodeIdGenerator).values({rightBatch}, true).planNode();

    CursorParameters params;
    params.maxDrivers = numThreads;
    params.planNode =
        PlanBuilder(planNodeIdGenerator)
            .values({leftBatch}, true)
            .addNode([&leftNode, &rightNode](
                         std::string id, core::PlanNodePtr // input
                        ) {
              return std::make_shared<CudfHashJoinNode>(
                  id, std::move(leftNode), std::move(rightNode));
            })
            // .project({"c0"}) // project only first column
            .planNode();

    // OperatorTestBase::assertQuery(params, referenceQuery);

    // assertQuery(params, leftBatch);

    auto result = readCursor(params, [](Task*) {});
    // std::cout<<"cudf Hash Join Result: \n";
    // printResults(result.second.front(), std::cout);
    return result;

    // Shared pointer of pool must be in scope. Otherwise pure virtual method called error.
    // auto pool = memory::getDefaultMemoryPool();
  }

bool CudfHashJoinDemo::CompareResults(
      int32_t numThreads,
      const std::vector<RowVectorPtr>& leftBatch, // probe input
      const std::vector<RowVectorPtr>& rightBatch, // build input
      const std::string& referenceQuery) {
    auto reference = testVeloxHashJoin(numThreads, leftBatch, rightBatch, referenceQuery);
    auto result    = testCudfHashJoin (numThreads, leftBatch, rightBatch, referenceQuery);
    return assertEqualResults(result.second, reference.second);
  }
*/

} // namespace facebook::velox::cudf_velox