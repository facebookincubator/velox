============================
Scanning and Filter Pushdown
============================

DWIO pushes filter evaluation as close to the data as possible: first at the
row-group / stride level using statistics, then inline during column decoding.
This section covers the key abstractions.

ScanSpec
========

``ScanSpec`` (``dwio/common/ScanSpec.h``) is a tree-shaped descriptor that
mirrors the column tree of the schema. Each node carries:

.. list-table::
   :widths: 20 45
   :header-rows: 1

   * - Field
     - Description
   * - ``filter``
     - A ``common::Filter`` applied to this column during decoding (e.g.
       ``BigintRange``, ``BytesRange``, ``IsNull``).
   * - ``projectOut``
     - Whether this column should appear in the output ``RowVector``.
   * - ``channel``
     - The output column index in the result ``RowVector``.
   * - ``constantValue``
     - A literal value to use instead of reading from the file (e.g. for
       partition columns).

**Adaptive filter reordering.** ``ScanSpec`` tracks per-filter selectivity at
runtime. Columns with higher rejection rates are evaluated first so that
subsequent columns decode fewer rows.

**Dynamic filter pushdown.** During execution the connector may receive new
filters (e.g. from a hash-join build side). It calls
``ScanSpec::setFilter()`` followed by ``RowReader::resetFilterCaches()`` to
inject the filter into subsequent row groups.

SelectiveColumnReader
=====================

``SelectiveColumnReader`` (``dwio/common/SelectiveColumnReader.h``) is the
core of the read path. Each column in the file gets its own reader instance.
The key methods are:

.. list-table::
   :widths: 15 50
   :header-rows: 1

   * - Method
     - Behavior
   * - ``read()``
     - Decodes rows and applies the column's filter inline via a
       ``ColumnVisitor``. Rows that fail the filter are dropped immediately
       without materializing values.
   * - ``getValues()``
     - Extracts the passing rows into the output vector. Only called for
       columns that are projected out.

**SIMD bulk path.** For simple filters on flat integer/float columns the
reader uses a bulk SIMD path that processes multiple values per instruction.

**Dictionary filter caching.** When a column uses dictionary encoding, the
reader evaluates the filter against dictionary entries once and caches a
bitmap of passing entries. Subsequent rows are tested against the bitmap
instead of re-evaluating the filter.

Row Group Skipping
==================

Before decoding any rows in a row group (DWRF stride or Parquet row group),
DWIO tests column-level statistics against the scan filters:

1. ``RowReader::filterRowGroups()`` calls ``FormatData::filterRowGroups()``
   for each column that has a filter.
2. ``MetadataFilter`` (``dwio/common/MetadataFilter.h``) combines the
   per-column results according to the AND/OR tree of the overall predicate.
   A row group is skipped only if the combined result proves no rows can pass.
3. Skipped row groups avoid all I/O and decoding for that range of rows.

Mutation
========

``Mutation`` (``dwio/common/Mutation.h``) supports row-level modifications
during the scan:

* **Row deletes.** A ``deletedRows`` bitmask marks rows that should be
  excluded from the output (e.g. for ACID delete files).
* **Random sampling.** ``RandomSkipTracker`` probabilistically skips rows to
  implement sampling scans.
* **Delta column updates.** Columns can be replaced or augmented with values
  from a delta file.

ColumnSelector (Deprecated)
===========================

.. note::

   ``ColumnSelector`` is deprecated and should not be used in future designs.
   It is still used in current code by the legacy bulk reader path. New code
   should use ``ScanSpec`` for column projection.

``ColumnSelector`` (``dwio/common/ColumnSelector.h``) handles column
projection:

* Columns can be selected by **name** or by **Hive column ID**.
* Supports schema evolution: when the file schema has a different column
  order or a subset of the requested columns, ``ColumnSelector`` maps between
  the file schema and the scan schema.
* Projected columns that do not exist in the file produce ``NULL`` vectors.
