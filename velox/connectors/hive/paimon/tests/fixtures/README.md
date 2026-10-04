# Paimon append fixtures

`append/` contains Parquet files produced by Apache Paimon
`e28c6582c6c864b82c22b6bb4768c092aeafdebd` (`2.2-SNAPSHOT`). The C++
tests read these checked-in files without Java, a catalog, or a planner.

`GenerateAppend.java` creates an append table with `bucket=-1`,
`file.format=parquet`, Snappy compression, `write-only=true`, and partition `p`.
The schema covers INTEGER, BIGINT, DOUBLE, BOOLEAN, and VARCHAR, including a
case-sensitive `Payload` name. `schema.json` records the Paimon schema; only the
machine-specific `path` option is removed.

`manifest.json` records actual planned DataSplits and DataFileMeta values, file
SHA-256 hashes, and rows returned by the Paimon Java reader for each split:

- Snapshot 3 has eight rows in four files. Two files belong to partition `east`;
  NULL and empty-string partitions each have one file. Duplicate rows, nullable
  values, and both BIGINT limits are preserved.
- Snapshot 4 replaces `east` with a caller-planned COW result: remove `id=3`,
  update `id=4`, and retain both copies of `id=2`. Other partitions are unchanged.
  This snapshot has seven rows. The fixture does not implement a Velox writer
  or global commit.
- `empty.parquet` is a valid zero-row file created with Paimon's format writer.
  It is an explicit executor EOF test input; Paimon's table writer normally
  omits empty files.

The C++ helpers normalize this metadata into `PaimonTableHandle`, column handles,
and versioned splits. The manifest is fixture metadata, not Paimon's serialized
Java DataSplit or the connector's wire format. Table options unrelated to task
execution stay outside the execution handle. A table configured with `bucket=-1`
can produce a DataSplit with bucket ID 0; those are different fields.

To regenerate, use a checkout or exported source tree at the exact commit above,
Maven, and JDK 11 or newer (`javac --release 8` compiles the generator). Build in
that source tree, using an explicit Maven repository directory:

```sh
mvn -B -pl paimon-bundle -am -Pfast-build -Dmaven.test.skip=true \
  -Dmaven.repo.local=/absolute/path/to/m2 package
```

Then run from the Velox repository, supplying fresh work and output directories:

```sh
python3 velox/connectors/hive/paimon/tests/fixtures/regenerate.py \
  --paimon /absolute/path/to/paimon-build \
  --maven-repo /absolute/path/to/m2 \
  --work-dir /absolute/path/to/fresh-generator-work \
  --output /absolute/path/to/fresh-fixtures
```

The script uses production jars from that local Paimon build, verifies file
hashes, and formats the metadata for review. Compare oracle rows and replace
`append/` only after reviewing the generated result. Paimon file identities,
creation times, and file bytes can vary between runs; the query results and
snapshot semantics are reproducible.
