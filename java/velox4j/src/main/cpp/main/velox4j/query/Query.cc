#include "Query.h"

#include <velox/exec/PlanNodeStats.h>

namespace velox4j {
using namespace facebook::velox;

Query::Query(
    const std::shared_ptr<const core::PlanNode>& plan,
    const std::shared_ptr<const ConfigArray>& queryConfig,
    const std::shared_ptr<const ConnectorConfigArray>& connectorConfig)
    : plan_(plan),
      queryConfig_(queryConfig),
      connectorConfig_(connectorConfig) {}

const std::shared_ptr<const core::PlanNode>& Query::plan() const {
  return plan_;
}

const std::shared_ptr<const ConfigArray>& Query::queryConfig() const {
  return queryConfig_;
}

const std::shared_ptr<const ConnectorConfigArray>& Query::connectorConfig()
    const {
  return connectorConfig_;
}

std::string Query::toString() const {
  return fmt::format("Query: plan {}", plan_->toString(true, true));
}

folly::dynamic Query::serialize() const {
  folly::dynamic obj = folly::dynamic::object;
  obj["name"] = "velox4j.Query";
  obj["plan"] = plan_->serialize();
  obj["queryConfig"] = queryConfig_->serialize();
  obj["connectorConfig"] = connectorConfig_->serialize();
  return obj;
}

std::shared_ptr<Query> Query::create(const folly::dynamic& obj, void* context) {
  auto plan = std::const_pointer_cast<const core::PlanNode>(
      ISerializable::deserialize<core::PlanNode>(obj["plan"], context));
  auto queryConfig = std::const_pointer_cast<const ConfigArray>(
      ISerializable::deserialize<ConfigArray>(obj["queryConfig"], context));
  auto connectorConfig = std::const_pointer_cast<const ConnectorConfigArray>(
      ISerializable::deserialize<ConnectorConfigArray>(
          obj["connectorConfig"], context));
  return std::make_shared<Query>(plan, queryConfig, connectorConfig);
}

void Query::registerSerDe() {
  auto& registry = DeserializationWithContextRegistryForSharedPtr();
  registry.Register("velox4j.Query", create);
}
} // namespace velox4j
