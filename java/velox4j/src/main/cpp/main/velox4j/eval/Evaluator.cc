#include "Evaluator.h"
#include "Evaluation.h"

namespace velox4j {
using namespace facebook::velox;

Evaluator::Evaluator(MemoryManager* memoryManager, const std::string& exprJson)
    : exprJson_(exprJson) {
  static std::atomic<uint32_t> executionId{0};
  const uint32_t eid = executionId++;
  auto evaluatorSerdePool = memoryManager->getVeloxPool(
      fmt::format("Evaluator Serde Memory Pool - EID {}", std::to_string(eid)),
      memory::MemoryPool::Kind::kLeaf);
  auto exprDynamic = folly::parseJson(exprJson_);
  auto expr =
      ISerializable::deserialize<Evaluation>(exprDynamic, evaluatorSerdePool);
  queryCtx_ = core::QueryCtx::create(
      nullptr,
      core::QueryConfig{expr->queryConfig()->toMap()},
      expr->connectorConfig()->toMap(),
      cache::AsyncDataCache::getInstance(),
      memoryManager
          ->getVeloxPool(
              fmt::format(
                  "Evaluator Memory Pool - EID {}", std::to_string(eid)),
              memory::MemoryPool::Kind::kAggregate)
          ->shared_from_this(),
      nullptr,
      fmt::format("Evaluator Context - EID {}", std::to_string(eid)));
  ee_ = std::make_unique<exec::SimpleExpressionEvaluator>(
      queryCtx_.get(),
      memoryManager->getVeloxPool(
          fmt::format(
              "Evaluator Leaf Memory Pool - EID {}", std::to_string(eid)),
          memory::MemoryPool::Kind::kLeaf));
  exprSet_ = ee_->compile(expr->expr());
}

VectorPtr Evaluator::eval(
    const SelectivityVector& rows,
    const RowVector& input) {
  VectorPtr vector{};
  ee_->evaluate(exprSet_.get(), rows, input, vector);
  VELOX_CHECK_NOT_NULL(vector, "Failed to evaluate expression");
  return vector;
}

} // namespace velox4j
