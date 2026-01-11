# Velox4J: Java Bindings for Velox

![badge](https://github.com/facebookincubator/velox/actions/workflows/java-build-java.yml/badge.svg?branch=main)
![badge](https://github.com/facebookincubator/velox/actions/workflows/java-build-cpp.yml/badge.svg?branch=main)

## Introduction

### What is Velox4J?

Velox4J is the Java bindings for Velox. It enables JVM applications to directly invoke Velox's
functionalities without writing and maintaining any C++ / JNI code.

## Design

Velox4J is designed within the following manners:

### Portable

Velox4J is designed to be portable. The eventual goal is to make one Velox4J release to be
shipped onto difference platforms without rebuilding the Jar file.

### Seamless Velox API Mapping

Velox4J directly adopts Velox's existing JSON serde framework and implements the following
JSON-serializable Velox components in Java-side:

- Data types
- Query plans
- Expressions
- Connectors

With the help of Velox's own JSON serde, there will be no re-interpreting layer for query plans
in Velox4J's C++ code base. Which means, the Java side Velox components defined in Velox4J's
Java code will be 1-on-1 mapped to Velox's associated components. The design makes Velox4J's
code base even small, and any new Velox features easy to add to Velox4J.

### Compatible With Arrow Java

Velox4J is compatible with [Apache Arrow's Java implementation](https://arrow.apache.org/java/). Built-in utilities converting between
Velox4J's RowVector / BaseVector and Arrow Java's VectorSchemaRoot / Table / FieldVector are provided.

## Prerequisites

### Platform

The project and its releases are now only tested on the following CPU architectures:

- x86-64

and on the following operating systems:

- Linux

Supports for platforms not on the above list will not be guaranteed to have by the main stream code
of Velox4J at the time. But certainly, contributions are always welcomed if anyone tends to involve.

### Build Toolchains

The minimum toolchain versions for building Velox4J:

- GCC 11
- JDK 11

## Build From Source

```shell
cd java/velox4j/
mvn clean install
```

## Get Started

The following is a brief example of using Velox4J to execute a query:

```java

public static void main(String[] args) {
  // 1. Initialize Velox4J.
  Velox4j.initialize();

  // 2. Define the plan output schema.
  final RowType outputType = new RowType(List.of(
      "n_nationkey",
      "n_name",
      "n_regionkey",
      "n_comment"
  ), List.of(
      new BigIntType(),
      new VarCharType(),
      new BigIntType(),
      new VarCharType()
  ));

  // 3. Create a table scan node.
  final TableScanNode scanNode = new TableScanNode(
      "plan-id-1",
      outputType,
      new HiveTableHandle(
          "connector-hive",
          "table-1",
          false,
          List.of(),
          null,
          outputType,
          Map.of()
      ),
      toAssignments(outputType)
  );

  // 4. Build the query.
  final Query query = new Query(scanNode, Config.empty(), ConnectorConfig.empty());

  // 5. Create a Velox4J session.
  final MemoryManager memoryManager = MemoryManager.create(AllocationListener.NOOP);
  final Session session = Velox4j.newSession(memoryManager);

  // 6. Execute the query. A Velox serial task will be returned.
  final SerialTask task = session.queryOps().execute(query);

  // 7. Add a split associating with the table scan node to the task, this makes
  // the scan read a local file "/tmp/nation.parquet".
  final File file = new File("/tmp/nation.parquet");
  final ConnectorSplit split = new HiveConnectorSplit(
      "connector-hive",
      0,
      false,
      file.getAbsolutePath(),
      FileFormat.PARQUET,
      0,
      file.length(),
      Map.of(),
      OptionalInt.empty(),
      Optional.empty(),
      Map.of(),
      Optional.empty(),
      Map.of(),
      Map.of(),
      Map.of(),
      Optional.empty(),
      Optional.empty()
  );
  task.addSplit(scanNode.getId(), split);
  task.noMoreSplits(scanNode.getId());

  // 8. Create a Java iterator from the Velox task.
  final Iterator<RowVector> itr = UpIterators.asJavaIterator(task);

  // 9. Collect and print results.
  while (itr.hasNext()) {
    final RowVector rowVector = itr.next(); // 9.1. Get next RowVector returned by Velox.
    final VectorSchemaRoot vsr = Arrow.toArrowTable(new RootAllocator(), rowVector).toVectorSchemaRoot(); // 9.2. Convert the RowVector into Arrow format (an Arrow VectorSchemaRoot in this case).
    System.out.println(vsr.contentToTSVString()); // 9.3. Print the arrow table to stdout.
    vsr.close(); // 9.4. Release the Arrow VectorSchemaRoot.
  }

  // 10. Close the Velox4J session.
  session.close();
  memoryManager.close();
}
```

Code of the `toAssignment` utility method used above:

```java
private static List<Assignment> toAssignments(RowType rowType) {
  final List<Assignment> list = new ArrayList<>();
  for (int i = 0; i < rowType.size(); i++) {
    final String name = rowType.getNames().get(i);
    final Type type = rowType.getChildren().get(i);
    list.add(new Assignment(name,
        new HiveColumnHandle(name, ColumnType.REGULAR, type, type, List.of())));
  }
  return list;
}
```

## Coding Style

Velox4J's code conforms to Java coding style from Google Java format and C++ coding style from Velox.

You can run the following command to fix all the code style issues during development, including both
the C++ code and Java code:

```shell
cd java/velox4j/
bash scripts/gha/format/format.sh -fix
```

Note, Docker environment is required to run the script.

If you only need to check the code format without fixing them, use the subcommand`-check` instead:

```shell
bash scripts/gha/format/format.sh -check
```

Specifically, the following script from outside Velox folder is reused to fix license header format issues:

```shell
scripts/check.py header main --fix
```
