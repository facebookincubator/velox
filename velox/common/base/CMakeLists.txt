# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

velox_add_library(
  velox_exception
  Exceptions.cpp
  VeloxException.cpp
  Exceptions.h
  HEADERS
  FmtStdFormatters.h
  VeloxException.h
)
velox_link_libraries(
  velox_exception
  PUBLIC velox_flag_definitions velox_process Folly::folly fmt::fmt gflags::gflags glog::glog
)

velox_add_library(
  velox_common_base
  AdmissionController.cpp
  BitUtil.cpp
  Counters.cpp
  Fs.cpp
  PeriodicStatsReporter.cpp
  RandomUtil.cpp
  RuntimeMetrics.cpp
  SimdUtil.cpp
  SkewedPartitionBalancer.cpp
  SpillConfig.cpp
  SplitBlockBloomFilter.cpp
  StatsReporter.cpp
  SuccinctPrinter.cpp
  HEADERS
  AdmissionController.h
  BitSet.h
  BitUtil.h
  BloomFilter.h
  CheckedArithmetic.h
  ClassName.h
  CoalesceIo.h
  ConcurrentCounter.h
  CountBits.h
  Counters.h
  Crc.h
  Doubles.h
  Fs.h
  GTestMacros.h
  IOUtils.h
  IndexedPriorityQueue.h
  Nulls.h
  PeriodicStatsReporter.h
  Pointers.h
  Portability.h
  PrefixSortConfig.h
  RandomUtil.h
  Range.h
  RuntimeMetrics.h
  ScopedLock.h
  SelectivityInfo.h
  Semaphore.h
  SimdUtil-inl.h
  SimdUtil.h
  SkewedPartitionBalancer.h
  SortingNetwork.h
  SpillConfig.h
  SplitBlockBloomFilter.h
  StatsReporter.h
  SuccinctPrinter.h
  TreeOfLosers.h
)

velox_link_libraries(
  velox_common_base
  PUBLIC velox_exception Folly::folly fmt::fmt xsimd
  PRIVATE velox_caching velox_common_compression velox_process velox_test_util glog::glog
)

if(${VELOX_BUILD_TESTING})
  add_subdirectory(tests)
endif()

if(${VELOX_ENABLE_BENCHMARKS})
  add_subdirectory(benchmarks)
endif()

velox_add_library(velox_id_map BigintIdMap.cpp HEADERS BigintIdMap.h)
velox_link_libraries(
  velox_id_map
  velox_memory
  velox_flag_definitions
  velox_process
  glog::glog
  Folly::folly
  fmt::fmt
)

velox_add_library(velox_status Status.cpp HEADERS Status.h)
velox_link_libraries(velox_status PUBLIC fmt::fmt Folly::folly PRIVATE glog::glog)

velox_add_library(velox_compare_flags INTERFACE HEADERS CompareFlags.h)

velox_add_library(velox_macros INTERFACE HEADERS Macros.h)

velox_add_library(velox_async_source INTERFACE HEADERS AsyncSource.h)

velox_add_library(velox_lazy_cpu_thread_pool_executor INTERFACE HEADERS LazyCPUThreadPoolExecutor.h)

velox_add_library(velox_exception_helper INTERFACE HEADERS ExceptionHelper.h)
