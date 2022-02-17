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

#include "velox/common/file/benchmark/ReadBenchmark.h"

using namespace facebook::velox;

// This benchmark measures the throughput of a Linux compatible FileSystem for
// various ReadFile APIs. The output helps us understand the maximum possible
// gains for queries. Example: If a single thread requires reading 1GB of data
// and the IO throughput is 100 MBps, then it takes 10 seconds to just read the
// data.
int main(int argc, char** argv) {
  folly::init(&argc, &argv, false);
  ReadBenchmark bm;
  bm.initialize();
  bm.run();
}
