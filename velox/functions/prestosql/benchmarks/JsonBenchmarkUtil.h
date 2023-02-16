#pragma once

#include <folly/Benchmark.h>

namespace facebook::velox::functions::prestosql {

#define JSON_BENCHMARK_NAMED_PARAM(                             \
    type, name, iter, jsonSize, frontPath, medianPath, endPath) \
  BENCHMARK_NAMED_PARAM(                                        \
      name,                                                     \
      type##_i_##iter##_jSize_##jsonSize##_front,               \
      iter,                                                     \
      #jsonSize,                                                \
      frontPath)                                                \
  BENCHMARK_NAMED_PARAM(                                        \
      name,                                                     \
      type##_i_##iter##_jSize_##jsonSize##_median,              \
      iter,                                                     \
      #jsonSize,                                                \
      medianPath)                                               \
  BENCHMARK_NAMED_PARAM(                                        \
      name,                                                     \
      type##_i_##iter##_jSize_##jsonSize##_end,                 \
      iter,                                                     \
      #jsonSize,                                                \
      endPath)

#define JSON_BENCHMARK_NAMED_PARAM_TWO_FUNCS(                           \
    type, func1, func2, iter, jsonSize, frontPath, medianPath, endPath) \
  JSON_BENCHMARK_NAMED_PARAM(                                           \
      type, func1, iter, jsonSize, frontPath, medianPath, endPath)      \
  JSON_BENCHMARK_NAMED_PARAM(                                           \
      type, func2, iter, jsonSize, frontPath, medianPath, endPath)

#define JSON_BENCHMARK_NAMED_PARAM_MULTI_FUNCS(                                \
    type, func1, func2, func3, iter, jsonSize, frontPath, medianPath, endPath) \
  JSON_BENCHMARK_NAMED_PARAM(                                                  \
      type, func1, iter, jsonSize, frontPath, medianPath, endPath)             \
  JSON_BENCHMARK_NAMED_PARAM(                                                  \
      type, func2, iter, jsonSize, frontPath, medianPath, endPath)             \
  JSON_BENCHMARK_NAMED_PARAM(                                                  \
      type, func3, iter, jsonSize, frontPath, medianPath, endPath)

} // namespace facebook::velox::functions::prestosql
