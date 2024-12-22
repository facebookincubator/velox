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

#include "velox/runner/Schema.h"

namespace facebook::velox::runner {

const connector::Table* Schema::findTable(const std::string& name) {
  std::vector<int32_t> dots;
  std::string lookupName;
  connector::Connector* connector = nullptr;
  for (auto i = 0; i < name.size(); ++i) {
    if (name[i] == '.') {
      dots.push_back(i);
    }
  }
  if (dots.empty()) {
    lookupName = defaultSchema_.empty()
        ? name
        : fmt::format("{}.{}", defaultSchema_, name);
    connector = defaultConnector_.get();
  } else if (dots.back() == name.size() - 1) {
    VELOX_USER_FAIL("Table name ends in '.': {}", name);
  } else if (dots.size() == 1) {
    lookupName = name;
    connector = defaultConnector_.get();
  } else if (dots.size() > 2) {
    VELOX_USER_FAIL("Table name has more than 3 parts: {}", name);
  } else {
    connector = connector::getConnector(name.substr(0, dots[0])).get();
    lookupName = name.substr(dots[0], name.size());
  }

  return connector->metadata()->findTable(lookupName);
}

} // namespace facebook::velox::runner
