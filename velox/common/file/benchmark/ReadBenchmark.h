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

#include <gflags/gflags.h>
#include <iostream>
#include <memory>

DECLARE_string(path);
DECLARE_int64(file_size_gb);
DECLARE_int32(num_threads);
DECLARE_int32(seed);
DECLARE_bool(odirect);
DECLARE_int32(request_bytes);
DECLARE_int32(gap);
DECLARE_int32(num_in_run);
DECLARE_string(config);
DECLARE_int32(measurement_size);

namespace facebook::velox {

// This benchmark aids in measuring the throughput of a FileSystem for various
// ReadFile APIs.
class ReadBenchmark {
 public:
  ReadBenchmark(std::ostream& out = std::cout);
  virtual ~ReadBenchmark() = default;

  // Initialize the benchmark.
  virtual void initialize();

  // Run the benchmark.
  // Default output results to standard out.
  virtual void run();

 protected:
  class Impl;
  std::shared_ptr<Impl> impl_;
};

} // namespace facebook::velox
