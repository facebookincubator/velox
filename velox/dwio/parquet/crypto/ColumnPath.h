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
