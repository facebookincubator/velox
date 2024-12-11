#include "velox/experimental/cudf/connectors/parquet/ParquetConnector.h"
#include "velox/experimental/cudf/connectors/parquet/ParquetDataSource.h"

namespace facebook::velox::cudf_velox::connector::parquet {

ParquetConnector::ParquetConnector(
    const std::string& id,
    std::shared_ptr<const config::ConfigBase> config,
    folly::Executor* /*executor*/)
    : Connector(id),
      parquetConfig_(std::make_shared<ParquetConfig>(config))
/*fileHandleFactory_(
    parquetConfig_->isFileHandleCacheEnabled()
        ? std::make_unique<SimpleLRUCache<std::string, FileHandle>>(
              parquetConfig_->numCacheFileHandles())
        : nullptr,
    std::make_unique<FileHandleGenerator>(config)),*/
/*, executor_(executor), */
{
  if (parquetConfig_->isFileHandleCacheEnabled()) {
    LOG(INFO) << "cudf::Parquet connector " << connectorId()
              << " created with maximum of "
              << parquetConfig_->numCacheFileHandles()
              << " cached file handles.";
  } else {
    LOG(INFO) << "cudf::Parquet connector " << connectorId()
              << " created with file handle cache disabled";
  }
}

std::unique_ptr<DataSource> createDataSource(
    const std::shared_ptr<const RowType>& outputType,
    const std::shared_ptr<ConnectorTableHandle>& tableHandle,
    const std::unordered_map<
        std::string,
        std::shared_ptr<connector::ColumnHandle>>& columnHandles,
    ConnectorQueryCtx* connectorQueryCtx) override final {
  return std::make_unique<ParquetDataSource>(
      outputType, tableHandle, columnHandles, connectorQueryCtx->memoryPool());
}

} // namespace facebook::velox::cudf_velox::connector::parquet
