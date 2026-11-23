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

#include "velox/common/vectorindex/FaissThreads.h"

#include <omp.h>

// MKL takes its thread count from its own settings, not from OpenMP's. Its
// header is reachable when FAISS links MKL; with another BLAS there is no MKL
// to set.
#if __has_include(<mkl_service.h>)
#include <mkl_service.h> // @manual
#define VELOX_FAISS_MKL 1
#endif

namespace facebook::velox::common::vectorindex {
namespace {

// Sets the calling thread's MKL thread count and returns its previous setting,
// 0 if the thread followed MKL's global one. Does nothing without MKL.
int setMklThreads([[maybe_unused]] int numThreads) {
#ifdef VELOX_FAISS_MKL
  return mkl_set_num_threads_local(numThreads);
#else
  return 0;
#endif
}

} // namespace

FaissThreads::FaissThreads(int numThreads)
    : previousNumThreads_{omp_get_max_threads()},
      previousDynamic_{omp_get_dynamic()},
      previousMklThreads_{setMklThreads(numThreads)} {
  omp_set_dynamic(0);
  omp_set_num_threads(numThreads);
}

FaissThreads::~FaissThreads() {
  omp_set_num_threads(previousNumThreads_);
  omp_set_dynamic(previousDynamic_);
  setMklThreads(previousMklThreads_);
}

} // namespace facebook::velox::common::vectorindex
