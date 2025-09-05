#pragma once


#include <string>
#include <optional>
#include <any>
#include <unordered_map>

namespace facebook::velox::config {

class IConfig {
public:
  IConfig() = default;
  virtual std::optional<std::any> Get(const std::string& key) const = 0;
  virtual bool Has(const std::string& key) const = 0;
  virtual std::unordered_map<std::string, std::string> rawConfigsCopy() const {
    return {};
  }
  virtual ~IConfig() = default;
};

} // namespace facebook::velox::config
