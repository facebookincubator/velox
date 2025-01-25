#include "velox/dwio/common/Options.h"
#include "velox/exec/TableWriter.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

#include "velox/experimental/cudf/tests/utils/CudfPlanBuilder.h"

namespace facebook::velox::cudf_velox::exec::test {

std::function<PlanNodePtr(std::string, PlanNodePtr)> addTableWriter(
    const RowTypePtr& inputColumns,
    const std::vector<std::string>& tableColumnNames,
    const std::shared_ptr<core::AggregationNode>& aggregationNode,
    const std::shared_ptr<core::InsertTableHandle>& insertHandle,
    facebook::velox::connector::CommitStrategy commitStrategy) {
  return [=](core::PlanNodeId nodeId,
             core::PlanNodePtr source) -> core::PlanNodePtr {
    return std::make_shared<core::TableWriteNode>(
        nodeId,
        inputColumns,
        tableColumnNames,
        aggregationNode,
        insertHandle,
        false,
        TableWriteTraits::outputType(aggregationNode),
        commitStrategy,
        std::move(source));
  };
}

std::function<PlanNodePtr(std::string, PlanNodePtr)> cudfTableWrite(
    const std::string& outputDirectoryPath,
    const dwio::common::FileFormat fileFormat,
    const std::shared_ptr<core::AggregationNode>& aggregationNode,
    const std::shared_ptr<dwio::common::WriterOptions>& options,
    const std::string& outputFileName) {
  return cudfTableWrite(
      outputDirectoryPath,
      fileFormat,
      aggregationNode,
      kParquetConnectorId,
      {},
      options,
      outputFileName);
}

std::function<PlanNodePtr(std::string, PlanNodePtr)> cudfTableWrite(
    const std::string& outputDirectoryPath,
    const dwio::common::FileFormat fileFormat,
    const std::shared_ptr<core::AggregationNode>& aggregationNode,
    const std::string_view& connectorId,
    const std::unordered_map<std::string, std::string>& serdeParameters,
    const std::shared_ptr<dwio::common::WriterOptions>& options,
    const std::string& outputFileName,
    const common::CompressionKind compression,
    const RowTypePtr& schema) {
  return [=](core::PlanNodeId nodeId,
             core::PlanNodePtr source) -> core::PlanNodePtr {
    auto rowType = schema ? schema : source->outputType();

    auto locationHandle = ParquetConnectorTestBase::makeLocationHandle(
        outputDirectoryPath,
        cudf_velox::connector::parquet::LocationHandle::TableType::kNew,
        outputFileName);
    auto parquetHandle = ParquetConnectorTestBase::makeParquetInsertTableHandle(
        rowType->names(), rowType->children(), locationHandle, compression);
    auto insertHandle = std::make_shared<core::InsertTableHandle>(
        std::string(connectorId), parquetHandle);

    return std::make_shared<core::TableWriteNode>(
        nodeId,
        rowType,
        rowType->names(),
        aggregationNode,
        insertHandle,
        false,
        TableWriteTraits::outputType(aggregationNode),
        facebook::velox::connector::CommitStrategy::kNoCommit,
        std::move(source));
  };
}

} // namespace facebook::velox::cudf_velox::exec::test
