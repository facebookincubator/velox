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
