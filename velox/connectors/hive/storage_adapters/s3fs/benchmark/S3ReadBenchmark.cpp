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

#include "velox/connectors/hive/storage_adapters/s3fs/benchmark/S3ReadBenchmark.h"

#include <folly/init/Init.h>
#include <gflags/gflags.h>

using namespace facebook::velox;

int main(int argc, char** argv) {
  std::string kUsage(
      "This program extends the ReadBenchmark for S3."
      " Run 'velox_s3read_benchmark -helpon=ReadBenchmark' for available options."
      " Please specify the S3 config via the 'config' option.");
  gflags::SetUsageMessage(kUsage);
  folly::init(&argc, &argv, false);
  S3ReadBenchmark bm;
  bm.initialize();
  bm.run();
}
