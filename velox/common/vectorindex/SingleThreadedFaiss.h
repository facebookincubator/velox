/*
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
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

#include <omp.h>

namespace facebook::velox::common::vectorindex {

/// Runs FAISS on the calling thread while the object lives. Velox already runs
/// many builds and searches in parallel on its drivers; FAISS's OpenMP regions
/// would start more threads on each driver, contend on libomp's global lock
/// and build HNSW graphs that differ from run to run. The setting applies to
/// the calling thread only.
class SingleThreadedFaiss {
 public:
  SingleThreadedFaiss() : previousNumThreads_{omp_get_max_threads()} {
    omp_set_num_threads(1);
  }

  ~SingleThreadedFaiss() {
    omp_set_num_threads(previousNumThreads_);
  }

  SingleThreadedFaiss(const SingleThreadedFaiss&) = delete;
  SingleThreadedFaiss& operator=(const SingleThreadedFaiss&) = delete;

 private:
  const int previousNumThreads_;
};

} // namespace facebook::velox::common::vectorindex
