#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/exec/CudfBatchConcat.h"
#include "velox/experimental/cudf/exec/Utilities.h"

namespace facebook::velox::cudf_velox {

std::unique_ptr<exec::Operator> CudfBatchConcatTranslator::toOperator(
    exec::DriverCtx* ctx,
    int32_t id,
    const core::PlanNodePtr& node) {
  if (auto batchConcatNode =
          std::dynamic_pointer_cast<const core::CudfBatchConcatNode>(node)) {
    return std::make_unique<CudfBatchConcat>(id, ctx, batchConcatNode);
  }
  return nullptr;
}

CudfBatchConcat::CudfBatchConcat(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    std::shared_ptr<const core::PlanNode> planNode)
    : exec::Operator(
          driverCtx,
          planNode->outputType(),
          operatorId,
          planNode->id(),
          "CudfBatchConcat"),
      CudfOperator(operatorId, planNode->id()),
      driverCtx_(driverCtx),
      targetRows_(CudfConfig::getInstance().batchSizeMinThreshold) {}

void CudfBatchConcat::addInput(RowVectorPtr input) {
  auto cudfVector = std::dynamic_pointer_cast<CudfVector>(input);
  VELOX_CHECK_NOT_NULL(cudfVector, "CudfBatchConcat expects CudfVector input");

  // Push input cudf table to buffer
  currentNumRows_ += cudfVector->getTableView().num_rows();
  buffer_.push_back(std::move(cudfVector));
}

RowVectorPtr CudfBatchConcat::getOutput() {
  // Drain the queue if there is any output to be flushed
  if (!outputQueue_.empty()) {
    auto table = std::move(outputQueue_.front());
    auto rowCount = table->num_rows();
    outputQueue_.pop();
    auto stream = cudfGlobalStreamPool().get_stream();
    return std::make_shared<CudfVector>(
        pool(), outputType_, rowCount, std::move(table), stream);
  }

  // Merge tables if there are enough rows
  if (currentNumRows_ >= targetRows_ || (noMoreInput_ && !buffer_.empty())) {
    // Use stream from existing buffer vectors
    auto stream = buffer_[0]->stream();
    auto tables = getConcatenatedTableBatched(buffer_, outputType_, stream);

    buffer_.clear();
    currentNumRows_ = 0;

    for (size_t i = 0; i < tables.size(); ++i) {
      bool isLast = (i == tables.size() - 1);
      auto rowCount = tables[i]->num_rows();

      // Do not push the last batch into the queue if it is smaller than
      // targetRows_ But push it if it is the final batch
      if (isLast && !noMoreInput_ && rowCount < targetRows_) {
        currentNumRows_ = rowCount;
        buffer_.push_back(
            std::make_shared<CudfVector>(
                pool(), outputType_, rowCount, std::move(tables[i]), stream));
      } else {
        outputQueue_.push(std::move(tables[i]));
      }
    }

    // Return the first batch from the new queue
    if (!outputQueue_.empty()) {
      auto table = std::move(outputQueue_.front());
      stream = cudfGlobalStreamPool().get_stream();
      auto rowCount = table->num_rows();
      outputQueue_.pop();
      return std::make_shared<CudfVector>(
          pool(), outputType_, rowCount, std::move(table), stream);
    }
  }

  return nullptr;
}

bool CudfBatchConcat::isFinished() {
  return noMoreInput_ && buffer_.empty() && outputQueue_.empty();
}

} // namespace facebook::velox::cudf_velox
