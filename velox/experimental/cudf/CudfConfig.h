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

#include <cudf/types.hpp>

#include <optional>
#include <string>
#include <unordered_map>

namespace facebook::velox::cudf_velox {

struct CudfConfig {
  /// Keys used by the initialize() method.
  static constexpr const char* kCudfEnabled{"cudf.enabled"};
  static constexpr const char* kCudfDebugEnabled{"cudf.debug_enabled"};
  static constexpr const char* kCudfMemoryResource{"cudf.memory_resource"};
  static constexpr const char* kCudfMemoryPercent{"cudf.memory_percent"};
  static constexpr const char* kCudfFunctionNamePrefix{
      "cudf.function_name_prefix"};
  static constexpr const char* kCudfAstExpressionEnabled{
      "cudf.ast_expression_enabled"};
  static constexpr const char* kCudfAstExpressionPriority{
      "cudf.ast_expression_priority"};
  static constexpr const char* kCudfJitExpressionEnabled{
      "cudf.jit_expression_enabled"};
  static constexpr const char* kCudfJitExpressionPriority{
      "cudf.jit_expression_priority"};
  static constexpr const char* kCudfOutputMr{"cudf.output_mr"};
  static constexpr const char* kCudfAllowCpuFallback{"cudf.allow_cpu_fallback"};
  static constexpr const char* kCudfLogFallback{"cudf.log_fallback"};
  static constexpr const char* kCudfBatchSizeMinThreshold{
      "cudf.batch_size_min_threshold"};
  static constexpr const char* kCudfBatchSizeMaxThreshold{
      "cudf.batch_size_max_threshold"};
  static constexpr const char* kCudfConcatOptimizationEnabled{
      "cudf.concat_optimization_enabled"};
  static constexpr const char* kCudfTimestampUnit{"cudf.timestamp_unit"};
  static constexpr const char* kUcxExchange{"cudf.exchange"};
  static constexpr const char* kUcxxErrorHandling{"ucxx.error_handling"};
  static constexpr const char* kUcxIntraNodeExchange{
      "cudf.intra_node_exchange"};
  static constexpr const char* kUcxxBlockingPolling{"ucxx.blocking_polling"};
  static constexpr const char* kUcxExchangeLogLevel{"cudf.exchange_log_level"};
  static constexpr const char* kUcxPartitionedOutputBatchRows{
      "cudf.partitioned_output_batch_rows"};
  static constexpr const char* kUcxExchangeCompression{
      "cudf.exchange_compression"};
  static constexpr const char* kUcxExchangeCompressionPipeline{
      "cudf.exchange_compression_pipeline"};
  static constexpr const char* kUcxExchangeCompressionPipelineThreads{
      "cudf.exchange_compression_pipeline_threads"};
  static constexpr const char* kUcxExchangeCompressionMinBytes{
      "cudf.exchange_compression_min_bytes"};
  static constexpr const char* kUcxExchangeCompressionSafetyMargin{
      "cudf.exchange_compression_safety_margin"};
  /// Query session configs for the cuDF Operators.
  static constexpr const char* kCudfTopNBatchSize{"cudf.topk_batch_size"};

  /// Singleton CudfConfig instance.
  /// Clients must set the configs below before invoking registerCudf().
  static CudfConfig& getInstance();

  /// Initialize from a map with the above keys.
  void initialize(std::unordered_map<std::string, std::string>&&);

  /// Enable cudf by default.
  /// Clients can disable here and enable it via the QueryConfig as well.
  bool enabled{true};

  /// Enable debug printing.
  bool debugEnabled{false};

  /// Allow fallback to CPU operators if GPU operator replacement fails.
  bool allowCpuFallback{true};

  /// Enable GPU exchange operators (UcxExchange / UcxPartitionedOutput).
  bool exchange{false};

  /// Whether to enable error handling in UCXX endpoints.
  bool ucxxErrorHandling{true};

  /// Whether intra-node exchange optimization is enabled.
  bool intraNodeExchange{false};

  /// Whether to use blocking polling in UCXX.
  bool ucxxBlockingPolling{true};

  /// VLOG level for ucx-exchange source files.
  int32_t exchangeLogLevel{0};

  /// Minimum rows accumulated per UCX partition before flushing.
  int64_t partitionedOutputBatchRows{10'000};

  /// GPU codec for the UCX exchange payload. "column-adaptive" runs the same
  /// per-column codec as "column", but uses fused real encode/send/decode
  /// samples to bypass work for query stages where it does not pay.
  /// "column-adaptive-freq-pfor-min128" additionally enables dictionary-PFOR,
  /// frequency-PFOR, and delta-frequency-PFOR candidates for numeric regions of
  /// at least 128 MiB. Non-adaptive and legacy policy names remain available
  /// for controlled comparisons.
  std::string exchangeCompression{"none"};

  /// Run exchange compression/decompression away from the single UCXX
  /// communicator thread so transport progress can overlap the GPU codec.
  /// Disabled by default to preserve the synchronous baseline for A/B testing.
  bool exchangeCompressionPipeline{false};

  /// Maximum number of codec tasks executing concurrently per worker process.
  /// Start with one because individual codec kernels already approach full-SM
  /// occupancy and excess concurrency can reduce overall efficiency.
  int32_t exchangeCompressionPipelineThreads{1};

  /// Send smaller exchange chunks without attempting the GPU codec. Zero
  /// preserves the existing behavior. This is an absolute-size complement to
  /// the codec's post-encode percentage-gain check.
  int64_t exchangeCompressionMinBytes{0};

  /// Require estimated transfer savings to exceed measured codec cost by this
  /// factor before adaptive compression is selected. Values above one reserve
  /// headroom for the codec's opportunity cost on concurrent query kernels.
  double exchangeCompressionSafetyMargin{1.10};

  /// Memory resource for cuDF.
  /// Possible values are (cuda, pool, async, arena, managed, managed_pool).
  std::string memoryResource{"async"};

  /// The initial percent of GPU memory to allocate for pool or arena memory
  /// resources.
  int32_t memoryPercent{50};

  /// Memory resource for output vectors. When set to a value different from
  /// memoryResource, a separate MR is created for output allocations.
  /// When empty, the main memoryResource is used.
  std::string outputMemoryResource;

  /// Register all the functions with the functionNamePrefix.
  std::string functionNamePrefix;

  /// Enable AST in expression evaluation.
  bool astExpressionEnabled{true};

  /// Enable JIT in expression evaluation.
  bool jitExpressionEnabled{true};

  /// Priority of AST expression. Expression with higher priority is chosen for
  /// a given root expression.
  /// Example:
  /// Priority of expression that uses individual cuDF functions is 50.
  /// If AST priority is 100 then for a velox expression node that is supported
  /// by both, AST will be chosen as replacement for cudf execution, if AST
  /// priority is 25 then standalone cudf function is chosen.
  int astExpressionPriority{100};

  /// Priority of JIT expression.
  int jitExpressionPriority{101};

  /// Whether to log a reason for falling back to Velox CPU execution.
  bool logFallback{true};

  /// Whether to insert CudfBatchConcat operators before supported Cudf
  /// operators.
  /// This can improve performance by reducing the number of cuda kernel
  /// launches on addInput of certain operators by collecting a minimum number
  /// of rows before concatenating and passing on to the next operator.
  /// This batch size is determined by batchSizeMinThreshold and
  /// batchSizeMaxThreshold
  bool concatOptimizationEnabled{false};

  /// Minimum rows to accumulate before GPU-side concatenation in
  /// `CudfBatchConcat` (default 100k).
  int32_t batchSizeMinThreshold{100000};

  /// Maximum rows allowed in a concatenated batch (user configurable).
  /// When not set, cuDF's own `size_type::max()` is used.
  std::optional<int32_t> batchSizeMaxThreshold;
  // Query config key for the TopN batch size in the cuDF TopN operator.
  int32_t topNBatchSize{5};

  /// Timestamp unit for cuDF timestamp types.
  /// Can be configured via kCudfTimestampUnit with string values:
  /// "s" (seconds), "ms" (milliseconds), "us" (microseconds), "ns"
  /// (nanoseconds).
  cudf::type_id timestampUnit = cudf::type_id::TIMESTAMP_NANOSECONDS;
};

} // namespace facebook::velox::cudf_velox
