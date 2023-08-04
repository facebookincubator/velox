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

#include "velox/common/file/File.h"

namespace Aws::S3 {
class S3Client;
}

namespace facebook::velox::filesystems {

class S3WriteFile : public WriteFile {
 public:
  S3WriteFile(const std::string& path, Aws::S3::S3Client* client);

  // Appends data to the end of the file.
  void append(std::string_view data) override;

  // Flushes any local buffers, i.e. ensures the backing medium received
  // all data that has been appended.
  void flush() override;

  // Close the file. Any cleanup (disk flush, etc.) will be done here.
  void close() override;

  // Current file size, i.e. the sum of all previous Appends.
  uint64_t size() const override;

  int numPartsUploaded() const;

 protected:
  class Impl;
  std::shared_ptr<Impl> impl_;
};

} // namespace facebook::velox::filesystems
