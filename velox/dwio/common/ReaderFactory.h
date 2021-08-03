/*
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

#include <memory>

#include "velox/dwio/common/InputStream.h"
#include "velox/dwio/common/Options.h"
#include "velox/dwio/common/Reader.h"

namespace facebook::dwio::common {

class ReaderFactory {
 public:
  explicit ReaderFactory(FileFormat format) : format_(format) {}

  virtual ~ReaderFactory() = default;

  FileFormat getFormat() const {
    return format_;
  }

  virtual std::unique_ptr<Reader> createReader(
      std::unique_ptr<InputStream> stream,
      const ReaderOptions& options) = 0;

 private:
  const FileFormat format_;
};

bool registerReaderFactory(std::shared_ptr<ReaderFactory> factory);

std::shared_ptr<ReaderFactory> getReaderFactory(FileFormat format);

#define VELOX_REGISTER_READER_FACTORY(theFactory)                  \
  namespace {                                                      \
  static bool FB_ANONYMOUS_VARIABLE(g_ReaderFactory) =             \
      facebook::dwio::common::registerReaderFactory((theFactory)); \
  }

} // namespace facebook::dwio::common
