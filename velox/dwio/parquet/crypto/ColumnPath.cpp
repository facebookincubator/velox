#include "velox/dwio/parquet/crypto/ColumnPath.h"
#include "folly/String.h"

namespace facebook::velox::parquet {

std::shared_ptr<ColumnPath> ColumnPath::fromDotString(const std::string& dotString) {
  std::vector<std::string> path;
  folly::split('-', dotString, path, true);
  return std::make_shared<ColumnPath>(std::move(path));
}

std::shared_ptr<ColumnPath> ColumnPath::extend(const std::string& nodeName) const {
  std::vector<std::string> path;
  path.reserve(path_.size() + 1);
  path.resize(path_.size() + 1);
  std::copy(path_.cbegin(), path_.cend(), path.begin());
  path[path_.size()] = nodeName;

  return std::make_shared<ColumnPath>(std::move(path));
}

std::string ColumnPath::toDotString() const {
  return folly::join(".", path_);
}

const std::vector<std::string>& ColumnPath::toDotVector() const { return path_; }

}
