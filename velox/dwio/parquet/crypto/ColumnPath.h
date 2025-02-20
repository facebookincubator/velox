#pragma once
#include <string>
#include <vector>
#include <memory>

namespace facebook::velox::parquet {

class ColumnPath {
 public:
  ColumnPath() : path_() {}
  explicit ColumnPath(const std::vector<std::string>& path) : path_(path) {}
  explicit ColumnPath(std::vector<std::string>&& path) : path_(std::move(path)) {}

  static std::shared_ptr<ColumnPath> fromDotString(const std::string& dotString);

  std::shared_ptr<ColumnPath> extend(const std::string& nodeName) const;
  std::string toDotString() const;

  const std::vector<std::string>& toDotVector() const;

 protected:
  std::vector<std::string> path_;
};

}
