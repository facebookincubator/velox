# ADBC Connector

## Motivation

Velox has no connector for relational databases such as MySQL or PostgreSQL.
Presto solves this with a family of JDBC-based connectors (mysql, postgresql,
bigquery) built on a shared `base-jdbc` module. The equivalent
vendor-independent standard for native code is Arrow ADBC (Arrow Database
Connectivity): a stable C API implemented by per-database drivers (MySQL,
PostgreSQL, SQLite, Snowflake, Flight SQL) that returns results as Arrow data.

ADBC is a natural fit for Velox because query results arrive via the Arrow C
data interface (`ArrowArrayStream`), and Velox already ships an Arrow C-ABI
bridge (`velox/vector/arrow/Bridge.h`) that imports Arrow arrays into Velox
vectors with minimal copying.

This document describes a single generic `AdbcConnector` that works with any
ADBC driver. Database-specific behavior lives in the driver, selected through
connector configuration — analogous to how Presto's JDBC connectors delegate
to a JDBC driver jar.

## Goals (v1)

- Read-only table scans through any ADBC driver.
- Column projection pushdown via generated SQL.
- Raw SQL pass-through (scan the result of an arbitrary query).
- Hermetic unit tests that do not require a running database.

## Non-goals (v1)

- Writes (`DataSink`), filter/limit/aggregation pushdown, parallel splits,
  dynamic filters, and catalog/schema discovery. See Future work.

## Dependency: vendored ADBC driver manager

The ADBC driver manager is a small, dependency-free Apache-2.0 licensed
library (one `.cc` plus two headers) that loads driver shared libraries via
`dlopen` and exposes the uniform ADBC C API. It is explicitly designed to be
vendored.

We vendor it into `velox/external/adbc/` (following `velox/external/md5`,
`velox/external/date`) from the `apache-arrow-adbc-14` release tag:

- `adbc.h` — the ADBC C API (structs and status codes).
- `adbc_driver_manager.h` / `adbc_driver_manager.cc` — the driver manager.

Release 14 is the newest tag whose driver manager has no third-party
dependencies (release 15+ adds driver-manifest support that requires
toml++). The vendored copy is only built when the connector is enabled.

## Build integration

New CMake option `VELOX_ENABLE_ADBC_CONNECTOR`, default `OFF` (consistent
with other connectors that add dependencies, e.g. `VELOX_ENABLE_S3`). When
`ON`, builds `velox/external/adbc` and `velox/connectors/adbc`.

## Components

All code lives in `velox/connectors/adbc/`, namespace
`facebook::velox::connector::adbc`, modeled on the TPC-H connector layout.

### AdbcConfig

Wraps the connector `ConfigBase`. Keys:

| Key | Meaning |
| --- | --- |
| `adbc.driver` | Driver shared library name or path (e.g. `adbc_driver_mysql`). Required unless a driver init function is supplied programmatically. |
| `adbc.entrypoint` | Optional driver entrypoint symbol; the driver manager derives a default from the library name. |
| `adbc.identifier-quote` | Quote string for identifiers in generated SQL. Default `"`; MySQL uses a backtick. |
| `adbc.option.<name>` | Passed through as `AdbcDatabaseSetOption(<name>, value)`, e.g. `adbc.option.uri`, `adbc.option.username`, `adbc.option.password`. |

The pass-through prefix keeps the connector fully generic: new driver options
require no Velox changes.

### AdbcTableHandle / AdbcColumnHandle

`AdbcTableHandle` identifies what to scan: either a table name or a raw SQL
query (exactly one must be set). `AdbcColumnHandle` carries the column name.
Both implement `serialize()` / `create()` / `registerSerDe()` like their TPC-H
counterparts.

### AdbcConnectorSplit

A remote database table is not splittable the way files are, so each scan
produces exactly one split (the same choice Presto's JDBC connectors make).
The split carries no data beyond the connector id; it exists to trigger
`DataSource::addSplit`.

### AdbcConnector / AdbcConnectorFactory

Registered under factory name `"adbc"`. The connector owns one
`AdbcDatabase`, initialized eagerly in the constructor so misconfiguration
fails at connector registration rather than first query. An `AdbcDatabase` is
shareable across connections per the ADBC spec.

The constructor accepts an optional `AdbcDriverInitFunc`. When set, the
driver manager uses this in-process entry point instead of `dlopen`. This
serves embedders that link drivers statically and is the injection point for
tests.

### AdbcDataSource

Per-split flow:

1. `addSplit`: create an `AdbcConnection` and `AdbcStatement`; set the SQL
   text; `AdbcStatementExecuteQuery` yields an `ArrowArrayStream`.
2. `next(size)`: pull the next chunk from the stream, import it with
   `importFromArrowAsOwner` (the vector takes ownership of the Arrow
   buffers), and return it as a `RowVector` cast to the output type. Returns
   `nullptr` when the stream is exhausted. The `size` hint is ignored in v1;
   chunk sizes are driver-determined.
3. Track `completedRows_` / `completedBytes_` from returned vectors.

SQL generation: for a table-name handle, `SELECT <q>c1<q>, <q>c2<q> FROM
<q>table<q>` using the configured quote and the projected columns (this is
the projection pushdown). For a query handle the text is used verbatim; the
projected columns must match the query's result schema.

Schema handling: the imported chunk's column names and types must match the
scan's output type; mismatches raise a user error naming the column. Result
columns are matched by name (case-insensitive) so drivers that reorder or
alias columns are handled.

`addDynamicFilter` raises `VELOX_NYI`.

### Error handling

A small RAII layer in the `.cpp` wraps `AdbcDatabase`, `AdbcConnection`,
`AdbcStatement`, and `ArrowArrayStream`, releasing them in destructors. Every
ADBC call goes through a checker that converts a non-OK `AdbcStatusCode`
into a `VELOX_FAIL` carrying the status name and the `AdbcError` message
(with `AdbcError::release` called). Connection failures surface as user
errors; unexpected driver states as runtime errors.

## Memory

Imported vectors reference Arrow buffers allocated by the driver, not by a
Velox `MemoryPool`; only conversion allocations (e.g. out-of-order strings)
hit the pool. This mirrors other `importFromArrowAsOwner` users and is
acceptable for v1; per-scan memory accounting of driver buffers is future
work.

## Testing

A fake in-process ADBC driver (`tests/AdbcTestDriver.{h,cpp}`) implements the
minimal driver surface (database/connection/statement lifecycle plus
`ExecuteQuery`). It serves configurable `RowVector` batches by exporting them
with `exportToArrow`, and records the SQL text it receives. It is injected
through the connector's `AdbcDriverInitFunc` hook — no shared library, no
network, no database.

Tests cover:

- Round trip: vectors served by the fake driver come back equal through a
  `TableScan` plan (multiple batches, multiple types, nulls).
- Generated SQL text for table-name handles (projection, identifier quoting)
  and verbatim text for query handles.
- Schema mismatch, missing column, and driver-error propagation.
- Split and handle serde round trips.

An opt-in integration test against a real driver (e.g. SQLite or MySQL via
`adbc.driver`) can be added later behind an environment variable; v1 keeps CI
hermetic.

## Future work

- Filter pushdown by translating `SubfieldFilters` into SQL `WHERE` clauses.
- Limit and aggregate pushdown.
- Parallel splits via key-range sharding.
- `DataSink` for `INSERT` via `AdbcStatementBind`.
- Driver-buffer memory accounting.
- Substrait plans (`AdbcStatementSetSubstraitPlan`) instead of SQL text.
