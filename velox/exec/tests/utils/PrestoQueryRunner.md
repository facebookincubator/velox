# How To Use PrestoQueryRunner

Currently, Aggregation Fuzzer verifies results against DuckDB. However, not all
functions are available in DuckDB and sometimes semantics don't match. It would
be nice to use PrestoJava as a source of truth to verify test results.

We've added PrestoJava to Velox in [#7628](https://github.com/facebookincubator/velox/pull/7628),
and we can use `PrestoQueryRunner` instead of `DuckQueryRunner` now.

**Note: We currently need to change the code to use it. Once `PrestoQueryRunner`
works well, we will always use it and remove `DuckQueryRunner`.**

The following are the steps on how to use `PrestoQueryRunner`, you can also refer to
the [PrestoQueryRunnerTest](https://github.com/facebookincubator/velox/blob/main/velox/exec/tests/PrestoQueryRunnerTest.cpp).

- Launch a Presto all-in-one instance or cluster, e.g. HiveQueryRunner in Presto Java
- Make code change in [AggregationFuzzerTest](https://github.com/facebookincubator/velox/blob/main/velox/exec/tests/AggregationFuzzerTest.cpp),
remove the `duckQueryRunner` variable and add a new `PrestoQueryRunner` variable named `prestoQueryRunner`.
- Use the `prestoQueryRunner` in the `main` method.

# Detail Code diff

The code diff is as follows,

```C++
diff --git a/velox/exec/tests/AggregationFuzzerTest.cpp b/velox/exec/tests/AggregationFuzzerTest.cpp
index 1ecdcf960..1b1e05972 100644
--- a/velox/exec/tests/AggregationFuzzerTest.cpp
+++ b/velox/exec/tests/AggregationFuzzerTest.cpp
@@ -28,6 +28,7 @@
 #include "velox/functions/prestosql/registration/RegistrationFunctions.h"
 #include "velox/vector/FlatVector.h"
 #include "velox/vector/tests/utils/VectorMaker.h"
+#include "velox/exec/tests/utils/PrestoQueryRunner.h"

 DEFINE_int64(
     seed,
@@ -509,15 +510,8 @@ int main(int argc, char** argv) {

   size_t initialSeed = FLAGS_seed == 0 ? std::time(nullptr) : FLAGS_seed;

-  auto duckQueryRunner =
-      std::make_unique<facebook::velox::exec::test::DuckQueryRunner>();
-  duckQueryRunner->disableAggregateFunctions({
-      "skewness",
-      // DuckDB results on constant inputs are incorrect. Should be NaN,
-      // but DuckDB returns some random value.
-      "kurtosis",
-      "entropy",
-  });
+  auto queryRunner =
+      std::make_unique<facebook::velox::exec::test::PrestoQueryRunner>("http://127.0.0.1:8080", "hive");

   // List of functions that have known bugs that cause crashes or failures.
   static const std::unordered_set<std::string> skipFunctions = {
@@ -585,5 +579,5 @@ int main(int argc, char** argv) {
       facebook::velox::exec::test::getCustomInputGenerators();
   options.timestampPrecision =
       facebook::velox::VectorFuzzer::Options::TimestampPrecision::kMilliSeconds;
-  return Runner::run(initialSeed, std::move(duckQueryRunner), options);
+  return Runner::run(initialSeed, std::move(queryRunner), options);
 }
```

# References
- https://github.com/facebookincubator/velox/issues/6595
- https://github.com/facebookincubator/velox/pull/7385
- https://github.com/facebookincubator/velox/pull/7568
- https://github.com/facebookincubator/velox/pull/7628
- https://github.com/facebookincubator/velox/blob/main/velox/exec/tests/PrestoQueryRunnerTest.cpp
- https://github.com/facebookincubator/velox/blob/main/velox/exec/tests/AggregationFuzzerTest.cpp

