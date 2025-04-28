#include "Config.h"

namespace velox4j {
using namespace facebook::velox;

std::unordered_map<std::string, std::string> ConfigArray::toMap() const {
  std::unordered_map<std::string, std::string> map(values_.size());
  for (const auto& kv : values_) {
    if (map.find(kv.first) != map.end()) {
      VELOX_FAIL("Duplicate key {} in config array", kv.first);
    }
    map.emplace(kv.first, kv.second);
  }
  return std::move(map);
}

folly::dynamic ConfigArray::serialize() const {
  folly::dynamic obj = folly::dynamic::object;
  obj["name"] = "velox4j.Config";
  folly::dynamic values = folly::dynamic::array;
  for (const auto& kv : values_) {
    folly::dynamic kvObj = folly::dynamic::object;
    kvObj["key"] = kv.first;
    kvObj["value"] = kv.second;
    values.push_back(kvObj);
  }
  obj["values"] = values;
  return obj;
};

std::shared_ptr<ConfigArray> ConfigArray::create(const folly::dynamic& obj) {
  std::vector<std::pair<std::string, std::string>> values;
  for (const auto& kv : obj["values"]) {
    values.emplace_back(kv["key"].asString(), kv["value"].asString());
  }
  return std::make_shared<ConfigArray>(std::move(values));
}

void ConfigArray::registerSerDe() {
  auto& registry = DeserializationRegistryForSharedPtr();
  registry.Register("velox4j.Config", create);
}

std::shared_ptr<ConfigArray> ConfigArray::empty() {
  static auto empty = std::make_shared<ConfigArray>(
      std::vector<std::pair<std::string, std::string>>{});
  return empty;
}

std::unordered_map<std::string, std::shared_ptr<config::ConfigBase>>
ConnectorConfigArray::toMap() const {
  std::unordered_map<std::string, std::shared_ptr<config::ConfigBase>> map(
      values_.size());
  for (const auto& kv : values_) {
    if (map.find(kv.first) != map.end()) {
      VELOX_FAIL("Duplicate key {} in config array", kv.first);
    }
    map.emplace(
        kv.first, std::make_shared<config::ConfigBase>(kv.second->toMap()));
  }
  return std::move(map);
}

folly::dynamic ConnectorConfigArray::serialize() const {
  folly::dynamic obj = folly::dynamic::object;
  obj["name"] = "velox4j.ConnectorConfig";
  folly::dynamic values = folly::dynamic::array;
  for (const auto& kv : values_) {
    folly::dynamic kvObj = folly::dynamic::object;
    kvObj["connectorId"] = kv.first;
    kvObj["config"] = kv.second->serialize();
    values.push_back(kvObj);
  }
  obj["values"] = values;
  return obj;
};

std::shared_ptr<ConnectorConfigArray> ConnectorConfigArray::create(
    const folly::dynamic& obj) {
  std::vector<std::pair<std::string, std::shared_ptr<const ConfigArray>>>
      values;
  for (const auto& kv : obj["values"]) {
    auto conf = std::const_pointer_cast<const ConfigArray>(
        ISerializable::deserialize<ConfigArray>(kv["config"]));
    values.emplace_back(kv["connectorId"].asString(), conf);
  }
  return std::make_shared<ConnectorConfigArray>(std::move(values));
}

void ConnectorConfigArray::registerSerDe() {
  auto& registry = DeserializationRegistryForSharedPtr();
  registry.Register("velox4j.ConnectorConfig", create);
}
} // namespace velox4j
