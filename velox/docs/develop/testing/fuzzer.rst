=================================
Expression and Aggregation Fuzzer
=================================

Velox allows users to define UDFs (user-defined functions) and UDAFs
(user-defined aggregate functions) and provides a fuzzer tools to test the
engine and UDFs thoroughly. These tools are being used to test builtin Presto
and Spark functions and have discovered numerous bugs caused by corner cases
that are difficult to cover in unit tests.

The Expression Fuzzer tests the expression evaluation engine and UDFs by
generating random expressions and evaluating these on random input vectors.
Each generated expression may contain multiple sub-expressions and each input
vector can have random and potentially nested encodings.

To ensure that evaluation engine and UDFs handle vector encodings correctly, the
expression fuzzer evaluates each expression twice and asserts the results to be
the same: using regular evaluation path and using simplified evaluation that
flattens all input vectors before evaluating an expression.

The Aggregation Fuzzer tests the HashAggregation operator, the StreamingAggregation
operator and UDAFs by generating random aggregations and evaluating these on
random input vectors.

The Aggregation Fuzzer tests global aggregations (no grouping keys), group-by
aggregations (one or more grouping keys), distinct aggregations(no aggregates),
aggregations with and without masks.

The results of aggregations using functions supported by DuckDB are compared
with DuckDB results.

For each aggregation, Fuzzer generates multiple logically equivalent plans and
verifies that results match. These plans are:

- Single aggregation (raw input, final result).
- Partial -> Final aggregation.
- Partial -> Intermediate -> Final aggregation.
- Partial -> LocalExchange(grouping keys) -> Final aggregation.
- All of the above using flattened input vectors.

In addition, to test StreamingAggregation operator, Fuzzer generates plans
using OrderBy and StreamingAggregation.

- OrderBy(grouping keys) -> Single streaming aggregation (raw input, final result).
- OrderBy(grouping keys) -> Partial streaming -> Final streaming aggregation.
- OrderBy(grouping keys) -> Partial streaming -> Intermediate streaming
  -> Final streaming aggregation.
- OrderBy(grouping keys) -> Partial streaming -> LocalMerge(grouping keys)
  -> Final streaming aggregation.
- All of the above using flattened input vectors.

When testing aggregate functions whose results depend on the order of inputs
(e.g. map_agg, map_union, arbitrary, etc.), the Fuzzer verifies that all plans
succeed or fail with compatible user exceptions. When plans succeed, the Fuzzer
verifies that number of result rows is the same across all plans.

At the end of the run, Fuzzer prints out statistics that show what has been
tested:

.. code-block::

    ==============================> Done with iteration 5683
    Total functions tested: 31
    Total masked aggregations: 1011 (17.79%)
    Total global aggregations: 500 (8.80%)
    Total group-by aggregations: 4665 (82.07%)
    Total distinct aggregations: 519 (9.13%)
    Total aggregations verified against DuckDB: 2537 (44.63%)
    Total failed aggregations: 1061 (18.67%)

How to integrate
---------------------------------------

To integrate with the Expression Fuzzer, create a test, register all scalar
functions supported by the engine, and call ``FuzzerRunner::run()`` defined in
`FuzzerRunner.h`_. See `ExpressionFuzzerTest.cpp`_.

.. _FuzzerRunner.h: https://github.com/facebookincubator/velox/blob/main/velox/expression/tests/ExpressionFuzzer.h

.. _ExpressionFuzzerTest.cpp: https://github.com/facebookincubator/velox/blob/main/velox/expression/tests/ExpressionFuzzerTest.cpp

Functions with known bugs can be excluded from testing using a skip-list.

Integration with Aggregation Fuzzer is similar. Create a test, register all
aggregate functions supported by the engine, and call
``AggregationFuzzerRunner::run()`` defined in `AggregationFuzzerRunner.h`_. See
`AggregationFuzzerTest.cpp`_.

.. _AggregationFuzzerRunner.h: https://github.com/facebookincubator/velox/blob/main/velox/exec/tests/AggregationFuzzer.h

.. _AggregationFuzzerTest.cpp: https://github.com/facebookincubator/velox/blob/main/velox/exec/tests/AggregationFuzzerTest.cpp

Aggregation Fuzzer allows to indicate functions whose results depend on the
order of inputs and optionally provide an expression to apply to the result to
make it stable. For example, the results of array_agg can be stabilized by
applying array_sort on top: array_sort(array_map(x)) and the results of map_agg
can be stabilized using array_sort(map_keys(map_agg(k, v))). Order-dependent
functions are tested to ensure no crashes or failures. The results of
order-dependent functions with stabilizing expressions are further verified for
correctness by ensuring that results of logically equivalent plans match.

How to run
----------------------------

Fuzzers support a number of powerful command line arguments.

* ``–-steps``: How many iterations to run. Each iteration generates and evaluates one expression or aggregation. Default is 10.

* ``–-duration_sec``: For how long to run in seconds. If both ``-–steps`` and ``-–duration_sec`` are specified, –duration_sec takes precedence.

* ``–-seed``: The seed to generate random expressions and input vectors with.

* ``–-v=1``: Verbose logging (from `Google Logging Library <https://github.com/google/glog#setting-flags>`_).

* ``–-only``: A comma-separated list of functions to use in generated expressions.

* ``–-batch_size``: The size of input vectors to generate. Default is 100.

In addition, Aggregation Fuzzer supports:

* ``--num_batches``: The number of input vectors of size `--batch_size` to generate. Default is 10.

If running from CLion IDE, add ``--logtostderr=1`` to see the full output.

How to reproduce failures
-------------------------------------

When Fuzzer test fails, a seed number and the evaluated expression are
printed to the log. An example is given below. Developers can use ``--seed``
with this seed number to rerun the exact same expression with the same inputs,
and use a debugger to investigate the issue. For the example below, the command
to reproduce the error would be ``velox/expression/tests/velox_expression_fuzzer_test --seed 1188545576``.

::

    I0819 18:37:52.249965 1954756 ExpressionFuzzer.cpp:685] ==============================> Started iteration 38
    (seed: 1188545576)
    I0819 18:37:52.250263 1954756 ExpressionFuzzer.cpp:578]
    Executing expression: in("c0",10 elements starting at 0 {120, 19, -71, null, 27, ...})
    I0819 18:37:52.250350 1954756 ExpressionFuzzer.cpp:581] 1 vectors as input:
    I0819 18:37:52.250401 1954756 ExpressionFuzzer.cpp:583] 	[FLAT TINYINT: 100 elements, 6 nulls]
    E0819 18:37:52.252044 1954756 Exceptions.h:68] Line: velox/expression/tests/ExpressionFuzzer.cpp:153, Function:compareVectors, Expression: vec1->equalValueAt(vec2.get(), i, i)Different results at idx '78': 'null' vs. '1', Source: RUNTIME, ErrorCode: INVALID_STATE
    terminate called after throwing an instance of 'facebook::velox::VeloxRuntimeError'
    ...

Note that changes to the set of all UDFs to test with invalidates this
reproduction, which can be affected by the skip function list, the ``--only``
argument, or the base commit, etc. This is because the chosen UDFs in the
expression are determined by both the seed and the pool of all UDFs to choose
from. So make sure you use the same configuration when reproducing a failure.

Accurate on-disk reproduction
-----------------------------

Sometimes developers may want to capture an issue and investigate later,
possibly by someone else using a different machine. Using ``--seed`` is not
sufficient to accurately reproduce the failure in this scenario. This could be
cased by different behaviors of random generator on different platforms,
additions/removals of UDFs from the list, and etc. To have an accurate
reproduction of a fuzzer failure regardless of environments you can record the
input vector and expression to files and replay these later.

1. Run Fuzzer using ``--seed`` and ``--repro_persist_path`` flags to save the input vector and expression to files in the specified directory. Add "--persist_and_run_once" if the issue is not an exception failure but a crash failure.

2. Run Expression Runner using generated files.

``--repro_persist_path <path/to/directory>`` flag tells the Fuzzer to save the
input vector, initial result vector, expression SQL, and other relevant data to files in a new directory saved within
the specified directory. It also prints out the exact paths for these. Fuzzer uses :doc:`VectorSaver <../debugging/vector-saver>`
for storing vectors on disk while preserving encodings.

ExpressionRunner needs at the very least a path to input vector and path to expression SQL to run.
However, you might need more files to reproduce the issue. All of which will be present in the directory
that the fuzzer test generated. You can directly point the ExpressionRunner to that directory using --fuzzer_repro_path
where it will pick up all the files automatically or you can specify each explicitly using other startup flags.
ExpressionRunner supports the following flags:

* ``--fuzzer_repro_path`` directory path where all input files (required to reproduce a failure) that are generated by the Fuzzer are expected to reside. ExpressionRunner will automatically pick up all the files from this folder unless they are explicitly specified via their respective startup flag.

* ``--input_path`` path to input vector that was created by the Fuzzer

* ``--sql_path`` path to expression SQL that was created by the Fuzzer

* ``--complex_constant_path`` optional path to complex constants that aren't accurately expressable in SQL (Array, Map, Structs, ...). This is used with SQL file to reproduce the exact expression, not needed when the expression doesn't contain complex constants.

* ``--lazy_column_list_path`` optional path for the file stored on-disk which contains a vector of column indices that specify which columns of the input row vector should be wrapped in lazy. This is used when the failing test included input columns that were lazy vector.

* ``--result_path`` optional path to result vector that was created by the Fuzzer. Result vector is used to reproduce cases where Fuzzer passes dirty vectors to expression evaluation as a result buffer. This ensures that functions are implemented correctly, taking into consideration dirty result buffer.

* ``--mode`` run mode. One of "verify", "common" (default), "simplified".

    - ``verify`` evaluates the expression using common and simplified paths and compares the results. This is identical to a fuzzer run.

    - ``common`` evaluates the expression using common path and prints the results to stdout.

    - ``simplified`` evaluates the expression using simplified path and prints the results to stdout.

    - ``query`` evaluate SQL query specified in --sql or --sql_path and print out results. If --input_path is specified, the query may reference it as table 't'.

* ``--num_rows`` optional number of rows to process in common and simplified modes. Default: 10. 0 means all rows. This flag is ignored in 'verify' mode.

* ``--store_result_path`` optional directory path for storing the results of evaluating SQL expression or query in 'common', 'simplified' or 'query' modes.

Example command:

::

    velox/expression/tests:velox_expression_runner_test --input_path "/path/to/input" --sql_path "/path/to/sql" --result_path "/path/to/result"

To assist debugging workload, ExpressionRunner supports ``--sql`` to specify
SQL expression on the command line. ``--sql`` option can be used standalone to
evaluate constant expression or together with ``--input_path`` to evaluate
expression on a vector. ``--sql`` and ``--sql_path`` flags are mutually
exclusive. If both are specified, ``--sql`` is used while ``--sql_path`` is
ignored. ``--sql`` option allow to specify multiple comma-separated SQL
expressions.

::

    $ velox/expression/tests:velox_expression_runner_test --sql "pow(2, 3), ceil(1.3)"

    I1101 11:32:51.955689 2306506 ExpressionRunner.cpp:127] Evaluating SQL expression(s): pow(2, 3), ceil(1.3)
    Result: ROW<_col0:DOUBLE,_col1:DOUBLE>
    8 | 2

    $ velox/expression/tests:velox_expression_runner_test --sql "pow(2, 3)"

    Evaluating SQL expression(s): pow(2, 3)
    Result: ROW<_col0:DOUBLE>
    8

    $ velox/expression/tests:velox_expression_runner_test --sql "array_sort(array[3,6,1,null,2])"
    Building: finished in 0.3 sec (100%) 817/3213 jobs, 0/3213 updated

    Evaluating SQL expression(s): array_sort(array[3,6,1,null,2])
    Result: ROW<_col0:ARRAY<INTEGER>>
    [1,2,3,6,null]

    $ velox/expression/tests:velox_expression_runner_test --sql "array_sort(array[3,6,1,null,2]), filter(array[1, 2, 3, 4], x -> (x % 2 == 0))"

    Evaluating SQL expression(s): array_sort(array[3,6,1,null,2]), filter(array[1, 2, 3, 4], x -> (x % 2 == 0))
    Result: ROW<_col0:ARRAY<INTEGER>,_col1:ARRAY<INTEGER>>
    [1,2,3,6,null] | [2,4]
