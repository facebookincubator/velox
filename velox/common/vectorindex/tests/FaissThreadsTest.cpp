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

#include <folly/Portability.h>
#include <gtest/gtest.h>
#include <omp.h>

#if __has_include(<mkl_service.h>)
#include <mkl_service.h> // @manual
#define VELOX_FAISS_MKL 1
#endif

namespace facebook::velox::common::vectorindex {
namespace {

// The number of threads MKL runs on from the calling thread, or 'orElse'
// without MKL.
int mklThreads(int orElse) {
#ifdef VELOX_FAISS_MKL
  return mkl_get_max_threads();
#else
  return orElse;
#endif
}

// The number of threads OpenMP runs on after a request for 'requested'. TSan
// builds pin OpenMP to one thread and ignore omp_set_num_threads.
int ompThreads(int requested) {
  return folly::kIsSanitizeThread ? 1 : requested;
}

// OpenMP and MKL run on the requested number of threads while the object
// lives, with OpenMP's dynamic adjustment off, and return to their settings
// after, whatever MKL's global setting.
TEST(FaissThreadsTest, setsAndRestores) {
#ifdef VELOX_FAISS_MKL
  const int previousMklGlobal = mkl_get_max_threads();
  mkl_set_num_threads(1);
#endif
  const int previousOmpThreads = omp_get_max_threads();
  const int previousOmpDynamic = omp_get_dynamic();
  omp_set_num_threads(5);
  omp_set_dynamic(1);
  const int previousMkl = mklThreads(0);
  {
    FaissThreads threads{3};
    EXPECT_EQ(ompThreads(3), omp_get_max_threads());
    EXPECT_EQ(0, omp_get_dynamic());
    EXPECT_EQ(3, mklThreads(3));
    {
      FaissThreads singleThreaded{1};
      EXPECT_EQ(ompThreads(1), omp_get_max_threads());
      EXPECT_EQ(1, mklThreads(1));
    }
    EXPECT_EQ(ompThreads(3), omp_get_max_threads());
    EXPECT_EQ(3, mklThreads(3));
  }
  EXPECT_EQ(ompThreads(5), omp_get_max_threads());
  EXPECT_EQ(1, omp_get_dynamic());
  EXPECT_EQ(previousMkl, mklThreads(0));
  omp_set_num_threads(previousOmpThreads);
  omp_set_dynamic(previousOmpDynamic);
#ifdef VELOX_FAISS_MKL
  mkl_set_num_threads(previousMklGlobal);
#endif
}

} // namespace
} // namespace facebook::velox::common::vectorindex
