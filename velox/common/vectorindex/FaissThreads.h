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

namespace facebook::velox::common::vectorindex {

/// Runs FAISS on 'numThreads' threads from the calling thread while the object
/// lives: its OpenMP regions, with OpenMP's dynamic adjustment of the team
/// size off, and its BLAS calls when FAISS uses MKL. The settings apply to the
/// calling thread only.
///
/// Velox runs many builds and searches in parallel on its drivers, so these
/// run on one thread. More threads per driver would contend on libomp's global
/// lock and build HNSW graphs that differ from run to run.
class FaissThreads {
 public:
  explicit FaissThreads(int numThreads);

  ~FaissThreads();

  FaissThreads(const FaissThreads&) = delete;
  FaissThreads& operator=(const FaissThreads&) = delete;

 private:
  const int previousNumThreads_;
  const int previousDynamic_;
  // MKL's previous setting for the calling thread, 0 if it followed MKL's
  // global one.
  const int previousMklThreads_;
};

} // namespace facebook::velox::common::vectorindex
