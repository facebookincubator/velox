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

#include <folly/executors/task_queue/UnboundedBlockingQueue.h>

#include "velox/common/memory/Memory.h"
#include "velox/core/PlanNode.h"
#include "velox/exec/Task.h"
#include "velox/functions/Udf.h"
#include "velox/type/Type.h"
#include "velox/vector/BaseVector.h"

using namespace facebook::velox;

template <typename T>
struct MyEqFunction {
  FOLLY_ALWAYS_INLINE bool call(bool& out, const int8_t& a, const int8_t& b) {
    out = a == b;
    return true; // True if result is not null.
  }
};

void InitPlan(
    memory::MemoryPool* memoryPool,
    std::shared_ptr<core::PlanNode>& planNode,
    VectorPtr flatVector) {
  auto inputRowType = ROW({{"my_col", TINYINT()}});
  const size_t vectorSize = 10;
  auto rowVector = std::make_shared<RowVector>(
      memoryPool, // pool where allocations will be made.
      inputRowType, // input row type (defined above).
      BufferPtr(nullptr), // no nulls for this example.
      vectorSize, // length of the vectors.
      std::vector<VectorPtr>{flatVector}); // the input vector data.

  std::shared_ptr<core::PlanNode> valueNode =
      std::make_shared<core::ValuesNode>(
          "0", std::vector<std::shared_ptr<RowVector>>{rowVector}, false);

  auto fieldAccessExprNode =
      std::make_shared<core::FieldAccessTypedExpr>(TINYINT(), "my_col");

  auto variant = facebook::velox::variant((int8_t)4);
  auto constant = std::make_shared<core::ConstantTypedExpr>(variant);

  auto exprTree = std::make_shared<core::CallTypedExpr>(
      BOOLEAN(),
      std::vector<core::TypedExprPtr>{fieldAccessExprNode, constant},
      "eq");
  std::shared_ptr<core::PlanNode> filterNode =
      std::make_shared<core::FilterNode>("1", exprTree, valueNode);

  auto fieldAccessExprNode2 =
      std::make_shared<core::FieldAccessTypedExpr>(TINYINT(), "my_col");

  planNode = std::make_shared<core::ProjectNode>(
      "2",
      std::vector<std::string>{"project"},
      std::vector<std::shared_ptr<const core::ITypedExpr>>{
          fieldAccessExprNode2},
      filterNode);
}

int main(int argc, char** argv) {
  auto pool = memory::getDefaultScopedMemoryPool();
  registerFunction<MyEqFunction, bool, int8_t, int8_t>({"eq"});
  auto queryCtx = core::QueryCtx::create();

  std::shared_ptr<core::PlanNode> planNode{nullptr};
  const size_t vectorSize = 10;
  auto flatVector = std::dynamic_pointer_cast<FlatVector<int8_t>>(
      BaseVector::create(TINYINT(), vectorSize, pool.get()));
  auto rawValues = flatVector->mutableRawValues();
  std::iota(rawValues, rawValues + vectorSize, 0); // 0, 1, 2, 3, ...
  // Project(`my_col` -> `project`)
  //   Filter(`my_col` eq 4)
  //     Values(col_name = 'my_col', values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
  InitPlan(pool.get(), planNode, flatVector);

  auto queue = std::make_shared<std::vector<RowVectorPtr>>();
  auto task = std::make_shared<exec::Task>(
      "simple_query",
      planNode,
      0,
      queryCtx,
      // consumer
      [queue, p = pool.get()](
          RowVectorPtr vector, exec::ContinueFuture* future) {
        if (!vector) {
          return exec::BlockingReason::kNotBlocked;
        }
        // Make sure to load lazy vector if not loaded already.
        for (auto& child : vector->children()) {
          child->loadedVector();
        }
        RowVectorPtr copy = std::dynamic_pointer_cast<RowVector>(
            BaseVector::create(vector->type(), vector->size(), p));
        copy->copy(vector.get(), 0, 0, vector->size());
        queue->push_back(std::move(copy));
        return exec::BlockingReason::kNotBlocked;
      });

  exec::Task::start(task, 1);
  // wait for task
  sleep(1);
  auto result = queue->back()->childAt(0);
  for (int i = 0; i < result->size(); i++) {
    LOG(INFO) << result->toString(i);
  }
  return 0;
}
