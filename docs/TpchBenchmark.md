# Velox I/O Optimizations and TpchBenchmark
## Introduction
This document is derived from a cycle of benchmarking to determine best I/O performance against AWS S3 Parque data for TPCH. It is not intended to show raw data but instead describe process and conclusions.

Benchmarking in Velox is made easy with the optionally built TpchBenchmark (velox_tpch_benchmark) executable. To build the benchmark executable _(build/release/velox/benchamrks/tpch/velox_tpch_benchmark)_, patch the code base with a small patch ([Appendix A](#appendix-a-patch-to-build-tpchbenchmark-executable)) then use the following command line to do the build:
```bash
$ make release
```
Make sure that the patched code is not accidently checked-in via any Pull Requests submitted to this github repository.

---
## Use Models for Velox
Velox is a library so there are multiple ways to use it. This document will describe two models used by popular applications.

### ___In-Process Multi-Threaded Executor Model___
This model is used by Presto and, as it turns out, is the model used by the [Velox TpchBenchmark Tool](#appendix-b-tpchbenchmark-tool-help-output). This model uses a single multi-threaded process to perform executation of queries in parallel. Each query is broken up into threads called __drivers__ via a planning algorithm.  Each __driver__ may also have a thread pool to perform I/O in an parallel manor. In the bechmarking tool both __driver__ thread count and I/O thread count are exposed as command line configuration options. In this model, care must be taken not to create too many threads since maximum number of threads is a product of __driver__ threads and I/O threads. In, this model the application owns creating __drivers__ and I/O Thread pools for Velox. In the case of the benchmark tool, the tool is responsible for both __driver__ threads and I/O threads. Presto likewise performs the same functions.

### ___Multiple Process Executor Model___
This model is used by Spark+Gluten and is different from the Presto model by where parallelism is achieved. Spark uses multiple processes where each process is a Gluten+Velox query processor. Spark scales by using many linux processes for query processing. In this case this means that the __drivers__ are outside of Velox and Gluten and is defined by the Spark configuration and number of workers. Gluten takes on the role of creating and exposing the I/O thread pool count to Spark as configuration and then injecting the I/O thread pool into Velox for parallel I/O.

---
## Built-In TpchBenchmark
As previously stated, this tool uses the in-process multi-threaded executor model. This tool exposes quite a few benchmark options as command line options. It also exposes many Velox internal options. This document will only cover a subset of the options and possible best known values for them.

The setup used in experiments leading to this document was an AWS instance ri6-8xlarge (32 vCPUs; 256GB RAM). The values (or formulas) below are based on these experiments.

### ___TpchBenchmark Tool Optimizations in Velox___
The tool exposes the folowing options (Note: use a single dash not a double dash):
* _num_drivers_ - As described above this represent the number of drivers or executors used to process the TPC-H queries.
* _num_io_threads_ - This represents the number of I/O threads used per __driver__ for I/O.
* _cache_gb_ - How much memory is used for caching data locally in this process. Memory caching cannot be shared in the multiple process executor model.
* _num_splits_per_file_ - This is a rowgroup optimization for the stored dataset for benchmarking.

__NOTE:__ _There is a limitation on the implementation of the AWS SDK that will cause failures if the __drivers__ threads times I/O threads grow much beyond 350 threads. This only really effects the Multi-Threaded __drivers__ applications like the benchmark tool. It is only known to be an issue when running against AWS S3. However, the error is coming from libcurl so its possible other SDK's could also be affected._

Velox exposes other options used for tuning that are of interest:
* _max_coalesce_bytes_ - Size of coalesced data, has small improvements as size grows.
* _max_coalesce_distance_bytes_ - Maximum gap bytes between data that can coalesced. Larger may mean more fetched data but at greater bytes/sec.

### ___Optimizations___

<table border="1">
    <tr><th>Option Name</th><th>Single Process<br />( including tool)</th><th>Multi-Process</th></tr>
    <tr><td>num_drivers</td><td>max(20, vCPUs<super>*</super> X 3 / 4)</td><td>NA</td></tr>
    <tr><td>num_io_threads</td><td>max(16, vCPUs<super>*</super> X 3 / 8)</td><td>vCPUs<super>*</super></td></tr>
    <tr><td>cache_gb</td><td>50% System RAM</td><td>NA (default = 0)</td></tr>
    <tr><td>num_splits_per_file</td><td>Row Group Size of Data</td><td>NA</td></tr>
    <tr><td>max_coalesce_bytes</td><td>Minimum of 90MB</td><td>Minimum of 90MB</td></tr>
    <tr>
        <td>max_coalesce_distance_bytes</td>
        <td>Workload dependent<super>**</super></td>
        <td>Workload dependent<super>**</super></td>
    </tr>
    <tr>
        <td colspan="3" style="font-size: 8pt">
&nbsp;&nbsp;*&nbsp;<b>vCPUs</b> = (cores * hyper-threads)<br />
**&nbsp;Wide tables and few columns retrieved per row can lead to many I/O requests, suggest increasing this value based on testing and needs to reduce I/O requests.
        </td>
    </tr>
</table>

## TBD: Discuss optimization in detail here...

---
## Appendix A: Patch to Build TpchBenchmark Executable
```diff
diff --git a/CMakeLists.txt b/CMakeLists.txt
index f43b8618..a777bbb8 100644
--- a/CMakeLists.txt
+++ b/CMakeLists.txt
@@ -40,7 +40,7 @@ message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
 option(
   VELOX_BUILD_TESTING
   "Enable Velox tests. This will enable all other build options automatically."
-  ON)
+  OFF)
 option(
   VELOX_BUILD_MINIMAL
   "Build a minimal set of components only. This will override other build options."
@@ -70,13 +70,13 @@ option(VELOX_ENABLE_EXAMPLES
        "Build examples. This will enable VELOX_ENABLE_EXPRESSION automatically."
        OFF)
 option(VELOX_ENABLE_SUBSTRAIT "Build Substrait-to-Velox converter." OFF)
-option(VELOX_ENABLE_BENCHMARKS "Enable Velox top level benchmarks." OFF)
+option(VELOX_ENABLE_BENCHMARKS "Enable Velox top level benchmarks." ON)
 option(VELOX_ENABLE_BENCHMARKS_BASIC "Enable Velox basic benchmarks." OFF)
-option(VELOX_ENABLE_S3 "Build S3 Connector" OFF)
+option(VELOX_ENABLE_S3 "Build S3 Connector" ON)
 option(VELOX_ENABLE_HDFS "Build Hdfs Connector" OFF)
-option(VELOX_ENABLE_PARQUET "Enable Parquet support" OFF)
+option(VELOX_ENABLE_PARQUET "Enable Parquet support" ON)
 option(VELOX_ENABLE_ARROW "Enable Arrow support" OFF)
-option(VELOX_ENABLE_CCACHE "Use ccache if installed." ON)
+option(VELOX_ENABLE_CCACHE "Use ccache if installed." OFF)
 
 option(VELOX_BUILD_TEST_UTILS "Builds Velox test utilities" OFF)
 option(VELOX_BUILD_PYTHON_PACKAGE "Builds Velox Python bindings" OFF)
diff --git a/Makefile b/Makefile
index e21f6b33..9578c377 100644
--- a/Makefile
+++ b/Makefile
@@ -80,7 +80,7 @@ cmake:					#: Use CMake to create a Makefile build system
 		${EXTRA_CMAKE_FLAGS}
 
 build:					#: Build the software based in BUILD_DIR and BUILD_TYPE variables
-	cmake --build $(BUILD_BASE_DIR)/$(BUILD_DIR) -j $(NUM_THREADS)
+	cmake --build $(BUILD_BASE_DIR)/$(BUILD_DIR) -j $(NUM_THREADS) --target velox_tpch_benchmark
 
 debug:					#: Build with debugging symbols
 	$(MAKE) cmake BUILD_DIR=debug BUILD_TYPE=Debug
diff --git a/velox/benchmarks/tpch/CMakeLists.txt b/velox/benchmarks/tpch/CMakeLists.txt
index fc865154..a8a7e573 100644
--- a/velox/benchmarks/tpch/CMakeLists.txt
+++ b/velox/benchmarks/tpch/CMakeLists.txt
@@ -27,6 +27,7 @@ target_link_libraries(
   velox_hive_connector
   velox_exception
   velox_memory
+  velox_s3fs
   velox_process
   velox_serialization
   velox_encode
```

---
## Appendix B: TpchBenchmark Tool Help Output
```bash
$ ../_build/release/velox/benchmarks/tpch/velox_tpch_benchmark --help
```
```
velox_tpch_benchmark: This program benchmarks TPC-H queries. Run 'velox_tpch_benchmark -helpon=TpchBenchmark' for available options.


  Flags from ./src/logging.cc:
    -alsologtoemail (log messages go to these email addresses in addition to
      logfiles) type: string default: ""
    -alsologtostderr (log messages go to stderr in addition to logfiles)
      type: bool default: false
    -colorlogtostderr (color messages logged to stderr (if supported by
      terminal)) type: bool default: false
    -drop_log_memory (Drop in-memory buffers of log contents. Logs can grow
      very quickly and they are rarely read before they need to be evicted from
      memory. Instead, drop them from memory as soon as they are flushed to
      disk.) type: bool default: true
    -log_backtrace_at (Emit a backtrace when logging at file:linenum.)
      type: string default: ""
    -log_dir (If specified, logfiles are written into this directory instead of
      the default logging directory.) type: string default: ""
    -log_link (Put additional links to the log files in this directory)
      type: string default: ""
    -log_prefix (Prepend the log prefix to the start of each log line)
      type: bool default: true
    -logbuflevel (Buffer log messages logged at this level or lower (-1 means
      don't buffer; 0 means buffer INFO only; ...)) type: int32 default: 0
    -logbufsecs (Buffer log messages for at most this many seconds) type: int32
      default: 30
    -logemaillevel (Email log messages logged at this level or higher (0 means
      email all; 3 means email FATAL only; ...)) type: int32 default: 999
    -logfile_mode (Log file mode/permissions.) type: int32 default: 436
    -logmailer (Mailer used to send logging email) type: string
      default: "/bin/mail"
    -logtostderr (log messages go to stderr instead of logfiles) type: bool
      default: false
    -max_log_size (approx. maximum log file size (in MB). A value of 0 will be
      silently overridden to 1.) type: int32 default: 1800
    -minloglevel (Messages logged at a lower level than this don't actually get
      logged anywhere) type: int32 default: 0
    -stderrthreshold (log messages at or above this level are copied to stderr
      in addition to logfiles.  This flag obsoletes --alsologtostderr.)
      type: int32 default: 2
    -stop_logging_if_full_disk (Stop attempting to log to disk if the disk is
      full.) type: bool default: false

  Flags from ./src/utilities.cc:
    -symbolize_stacktrace (Symbolize the stack trace in the tombstone)
      type: bool default: true

  Flags from ./src/vlog_is_on.cc:
    -v (Show all VLOG(m) messages for m <= this. Overridable by --vmodule.)
      type: int32 default: 0
    -vmodule (per-module verbose level. Argument is a comma-separated list of
      <module name>=<log level>. <module name> is a glob pattern, matched
      against the filename base (that is, name ignoring .cc/.h./-inl.h). <log
      level> overrides any value given by --v.) type: string default: ""



  Flags from /build/gflags-WDCpEz/gflags-2.2.2/src/gflags.cc:
    -flagfile (load flags from file) type: string default: ""
    -fromenv (set flags from the environment [use 'export FLAGS_flag1=value'])
      type: string default: ""
    -tryfromenv (set flags from the environment if present) type: string
      default: ""
    -undefok (comma-separated list of flag names that it is okay to specify on
      the command line even if the program does not define a flag with that
      name.  IMPORTANT: flags in this list that have arguments MUST use the
      flag=value format) type: string default: ""

  Flags from /build/gflags-WDCpEz/gflags-2.2.2/src/gflags_completions.cc:
    -tab_completion_columns (Number of columns to use in output for tab
      completion) type: int32 default: 80
    -tab_completion_word (If non-empty, HandleCommandLineCompletions() will
      hijack the process and attempt to do bash-style command line flag
      completion on this value.) type: string default: ""

  Flags from /build/gflags-WDCpEz/gflags-2.2.2/src/gflags_reporting.cc:
    -help (show help on all flags [tip: all flags can have two dashes])
      type: bool default: false currently: true
    -helpfull (show help on all flags -- same as -help) type: bool
      default: false
    -helpmatch (show help on modules whose name contains the specified substr)
      type: string default: ""
    -helpon (show help on the modules named by this flag value) type: string
      default: ""
    -helppackage (show help on all modules in the main package) type: bool
      default: false
    -helpshort (show help on only the main module for this program) type: bool
      default: false
    -helpxml (produce an xml version of help) type: bool default: false
    -version (show version and build info and exit) type: bool default: false



  Flags from /home/paul/code/velox-fork/folly/folly/Benchmark.cpp:
    -benchmark (Run benchmarks.) type: bool default: false
    -bm_estimate_time (Estimate running time) type: bool default: false
    -bm_json_verbose (File to write verbose JSON format (for BenchmarkCompare /
      --bm_relative_to). NOTE: this is written independent of the above --json
      / --bm_relative_to.) type: string default: ""
    -bm_max_iters (Maximum # of iterations we'll try for each benchmark.)
      type: int64 default: 1073741824
    -bm_max_secs (Maximum # of seconds we'll spend on each benchmark.)
      type: int32 default: 1
    -bm_max_trials (Maximum number of trials (iterations) executed for each
      benchmark.) type: uint32 default: 1000
    -bm_min_iters (Minimum # of iterations we'll try for each benchmark.)
      type: int32 default: 1
    -bm_min_usec (Minimum # of microseconds we'll accept for each benchmark.)
      type: int64 default: 100
    -bm_profile (Run benchmarks with constant number of iterations) type: bool
      default: false
    -bm_profile_iters (Number of iterations for profiling) type: int64
      default: 1000
    -bm_regex (Only benchmarks whose names match this regex will be run.)
      type: string default: ""
    -bm_relative_to (Print benchmark results relative to an earlier dump (via
      --bm_json_verbose)) type: string default: ""
    -bm_result_width_chars (Width of results table in characters) type: uint32
      default: 76
    -json (Output in JSON format.) type: bool default: false



  Flags from /home/paul/code/velox-fork/folly/folly/detail/MemoryIdler.cpp:
    -folly_memory_idler_purge_arenas (if enabled, folly memory-idler purges
      jemalloc arenas on thread idle) type: bool default: true



  Flags from /home/paul/code/velox-fork/folly/folly/executors/CPUThreadPoolExecutor.cpp:
    -dynamic_cputhreadpoolexecutor (CPUThreadPoolExecutor will dynamically
      create and destroy threads) type: bool default: true

  Flags from /home/paul/code/velox-fork/folly/folly/executors/IOThreadPoolExecutor.cpp:
    -dynamic_iothreadpoolexecutor (IOThreadPoolExecutor will dynamically create
      threads) type: bool default: true

  Flags from /home/paul/code/velox-fork/folly/folly/executors/ThreadPoolExecutor.cpp:
    -threadtimeout_ms (Idle time before ThreadPoolExecutor threads are joined)
      type: int64 default: 60000



  Flags from /home/paul/code/velox-fork/folly/folly/init/Init.cpp:
    -logging (Logging configuration) type: string default: ""



  Flags from /home/paul/code/velox-fork/folly/folly/synchronization/Hazptr.cpp:
    -folly_hazptr_use_executor (Use an executor for hazptr asynchronous
      reclamation) type: bool default: true



  Flags from /home/paul/code/velox-fork/velox/benchmarks/tpch/TpchBenchmark.cpp:
    -cache_gb (GB of process memory for cache and query.. if non-0, uses mmap
      to allocator and in-process data cache.) type: int32 default: 0
    -clear_ram_cache (Clear RAM cache before each query.Flushes in process and
      OS file system cache (if root on Linux)) type: bool default: false
    -clear_ssd_cache (Clears SSD cache before each query) type: bool
      default: false
    -data_format (Data format) type: string default: "parquet"
    -data_path (Root path of TPC-H data. Data layout must follow Hive-style
      partitioning. Example layout for '-data_path=/data/tpch10'
             /data/tpch10/customer
             /data/tpch10/lineitem
             /data/tpch10/nation
             /data/tpch10/orders
             /data/tpch10/part
             /data/tpch10/partsupp
             /data/tpch10/region
             /data/tpch10/supplier
      If the above are directories, they contain the data files for each table.
      If they are files, they contain a file system path for each data file,
      one per line. This allows running against cloud storage or HDFS)
      type: string default: ""
    -full_sorted_stats (Add full stats to the report on  --test_flags_file)
      type: bool default: true
    -include_custom_stats (Include custom statistics along with execution
      statistics) type: bool default: false
    -include_results (Include results in the output) type: bool default: false
    -io_meter_column_pct (Percentage of lineitem columns to include in IO meter
      query. The columns are sorted by name and the n% first are scanned)
      type: int32 default: 0
    -num_drivers (Number of drivers) type: int32 default: 4
    -num_io_threads (Threads for speculative IO) type: int32 default: 8
    -num_repeats (Number of times to run each query) type: int32 default: 1
    -num_splits_per_file (Number of splits per file) type: int32 default: 10
    -run_query_verbose (Run a given query and print execution statistics)
      type: int32 default: -1
    -ssd_cache_gb (Size of local SSD cache in GB) type: int32 default: 0
    -ssd_checkpoint_interval_gb (Checkpoint every n GB new data in cache)
      type: int32 default: 8
    -ssd_path (Directory for local SSD cache) type: string default: ""
    -test_flags_file (Path to a file containing gflafs and values to try.
      Produces results for each flag combination sorted on performance)
      type: string default: ""
    -use_native_parquet_reader (Use Native Parquet Reader) type: bool
      default: true
    -warmup_after_clear (Runs one warmup of the query before measured run. Use
      to run warm after clearing caches.) type: bool default: false



  Flags from /home/paul/code/velox-fork/velox/common/caching/SsdFile.cpp:
    -ssd_odirect (Use O_DIRECT for SSD cache IO) type: bool default: true
    -ssd_verify_write (Read back data after writing to SSD) type: bool
      default: false



  Flags from /home/paul/code/velox-fork/velox/connectors/hive/HiveConnector.cpp:
    -num_file_handle_cache (Max number of file handles to cache.) type: int32
      default: 20000



  Flags from /home/paul/code/velox-fork/velox/dwio/common/BufferedInput.cpp:
    -wsVRLoad (Use WS VRead API to load) type: bool default: false

  Flags from /home/paul/code/velox-fork/velox/dwio/common/CachedBufferedInput.cpp:
    -cache_prefetch_min_pct (Minimum percentage of actual uses over references
      to a column for prefetching. No prefetch if > 100) type: int32
      default: 80
    -max_coalesced_bytes (Maximum size of single coalesced IO) type: int64
      default: 134217728

  Flags from /home/paul/code/velox-fork/velox/dwio/common/Options.cpp:
    -max_coalesce_distance_bytes (Maximum distance in which coalesce will
      combine requests.) type: int32 default: 524288



  Flags from /home/paul/code/velox-fork/velox/dwio/parquet/reader/ParquetReader.cpp:
    -parquet_prefetch_rowgroups (Number of next row groups to prefetch. 1 means
      prefetch the next row group before decoding the current one) type: int32
      default: 1



  Flags from /home/paul/code/velox-fork/velox/exec/TableScan.cpp:
    -split_preload_per_driver (Prefetch split metadata) type: int32 default: 2



  Flags from /home/paul/code/velox-fork/velox/expression/Expr.cpp:
    -force_eval_simplified (Whether to overwrite queryCtx and force the use of
      simplified expression evaluation path.) type: bool default: false



  Flags from /home/paul/code/velox-fork/velox/flag_definitions/flags.cpp:
    -avx2 (Enables use of AVX2 when available) type: bool default: true
    -bmi2 (Enables use of BMI2 when available) type: bool default: true
    -max_block_value_set_length (Max entries per column that the block
      meta-record stores for pre-flight filtering optimization) type: int32
      default: 5
    -memory_usage_aggregation_interval_millis (Interval to compute aggregate
      memory usage for all nodes) type: int32 default: 2
    -velox_enable_memory_usage_track_in_default_memory_pool (If true, enable
      memory usage tracking in the default memory pool) type: bool
      default: false
    -velox_exception_system_stacktrace_enabled (Enable the stacktrace for
      system type of VeloxException) type: bool default: true
    -velox_exception_system_stacktrace_rate_limit_ms (Min time interval in
      milliseconds between stack traces captured in system type of
      VeloxException; off when set to 0 (the default)) type: int32 default: 0
    -velox_exception_user_stacktrace_enabled (Enable the stacktrace for user
      type of VeloxException) type: bool default: false
    -velox_exception_user_stacktrace_rate_limit_ms (Min time interval in
      milliseconds between stack traces captured in user type of
      VeloxException; off when set to 0 (the default)) type: int32 default: 0
    -velox_memory_leak_check_enabled (If true, check fails on any memory leaks
      in memory pool and memory manager) type: bool default: false
    -velox_memory_num_shared_leaf_pools (Number of shared leaf memory pools per
      process) type: int32 default: 32
    -velox_memory_pool_mb (Size of file cache/operator working memory in MB)
      type: int32 default: 4096
    -velox_save_input_on_expression_any_failure_path (Enable saving input
      vector and expression SQL on any failure during expression evaluation.
      Specifies the directory to use for storing the vectors and expression SQL
      strings.) type: string default: ""
    -velox_save_input_on_expression_system_failure_path (Enable saving input
      vector and expression SQL on system failure during expression evaluation.
      Specifies the directory to use for storing the vectors and expression SQL
      strings. This flag is ignored if
      velox_save_input_on_expression_any_failure_path is set.) type: string
      default: ""
    -velox_suppress_memory_capacity_exceeding_error_message (If true, suppress
      the verbose error message in memory capacity exceeded exception. This is
      only used by test to control the test error output size) type: bool
      default: false
    -velox_time_allocations (Record time and volume for large allocation/free)
      type: bool default: true
```

---
