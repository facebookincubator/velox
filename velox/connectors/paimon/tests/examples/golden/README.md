# Paimon golden tables

Paimon Java 1.4.1 wrote each table here in Nimble, planned its splits and read
them. Each case directory holds:

- `table.json`: the columns as Hive type strings, their field IDs, the primary
  and partition keys, and the table options.
- `splits.json`: the splits Paimon Java planned, as `PaimonConnectorSplit` JSON
  with paths relative to the case directory.
- `expected.json`: the rows Paimon Java read from those splits, one JSON array
  per row. Decimals are strings, binary values are base64, and maps are arrays
  of [key, value] pairs.
- `data/` and `index/`: the data files and deletion-vector index files the
  splits reference.

`PaimonGoldenTableTest` reads the splits with Velox and compares the rows with
`expected.json`. File names and creation times change whenever the tables are
regenerated.
