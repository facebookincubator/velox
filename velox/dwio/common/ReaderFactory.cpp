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

#include "velox/dwio/common/ReaderFactory.h"

namespace facebook::dwio::common {

namespace {
std::unordered_map<FileFormat, std::shared_ptr<ReaderFactory>>&
readerFactories() {
  static std::unordered_map<FileFormat, std::shared_ptr<ReaderFactory>>
      factories;
  return factories;
}
} // namespace

bool registerReaderFactory(std::shared_ptr<ReaderFactory> factory) {
  bool ok = readerFactories().insert({factory->getFormat(), factory}).second;
  VELOX_CHECK(
      ok,
      "ReaderFactory for format {} is already registered",
      toString(factory->getFormat()));
  return true;
}

std::shared_ptr<ReaderFactory> getReaderFactory(FileFormat format) {
  auto it = readerFactories().find(format);
  VELOX_CHECK(
      it != readerFactories().end(),
      "ReaderFactory for format {} is not registered",
      toString(format));
  return it->second;
}

} // namespace facebook::dwio::common
