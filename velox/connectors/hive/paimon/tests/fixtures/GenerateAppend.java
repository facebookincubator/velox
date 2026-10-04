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

import org.apache.paimon.catalog.Catalog;
import org.apache.paimon.catalog.CatalogContext;
import org.apache.paimon.catalog.CatalogFactory;
import org.apache.paimon.catalog.Identifier;
import org.apache.paimon.data.BinaryString;
import org.apache.paimon.data.GenericRow;
import org.apache.paimon.data.InternalRow;
import org.apache.paimon.format.FileFormat;
import org.apache.paimon.format.FormatWriter;
import org.apache.paimon.fs.Path;
import org.apache.paimon.fs.PositionOutputStream;
import org.apache.paimon.io.DataFileMeta;
import org.apache.paimon.options.Options;
import org.apache.paimon.schema.Schema;
import org.apache.paimon.table.FileStoreTable;
import org.apache.paimon.table.sink.BatchTableCommit;
import org.apache.paimon.table.sink.BatchTableWrite;
import org.apache.paimon.table.sink.BatchWriteBuilder;
import org.apache.paimon.table.source.DataSplit;
import org.apache.paimon.table.source.ReadBuilder;
import org.apache.paimon.table.source.Split;
import org.apache.paimon.types.DataTypes;
import org.apache.paimon.utils.JsonSerdeUtil;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** Offline only. Build against Paimon e28c6582c6c864b82c22b6bb4768c092aeafdebd. */
public final class GenerateAppend {
    private static final String COMMIT = "e28c6582c6c864b82c22b6bb4768c092aeafdebd";
    private static final Map<String, String> COPIED = new LinkedHashMap<>();
    private static java.nio.file.Path output;

    private static Map<String, Object> object(Object... entries) {
        Map<String, Object> result = new LinkedHashMap<>();
        for (int i = 0; i < entries.length; i += 2) {
            result.put((String) entries[i], entries[i + 1]);
        }
        return result;
    }

    private static BinaryString string(String value) {
        return value == null ? null : BinaryString.fromString(value);
    }

    private static GenericRow row(int id, String value, Double amount, Boolean active, Long big, String p) {
        return GenericRow.of(id, string(value), amount, active, big, string(p));
    }

    private static void write(FileStoreTable table, boolean overwrite, InternalRow... rows) throws Exception {
        BatchWriteBuilder builder = table.newBatchWriteBuilder();
        if (overwrite) {
            builder.withOverwrite(Collections.singletonMap("p", "east"));
        }
        try (BatchTableWrite write = builder.newWrite(); BatchTableCommit commit = builder.newCommit()) {
            for (InternalRow row : rows) {
                write.write(row);
            }
            commit.commit(write.prepareCommit());
        }
    }

    private static List<Object> values(InternalRow row) {
        return Arrays.asList(
                row.isNullAt(0) ? null : row.getInt(0),
                row.isNullAt(1) ? null : row.getString(1).toString(),
                row.isNullAt(2) ? null : row.getDouble(2),
                row.isNullAt(3) ? null : row.getBoolean(3),
                row.isNullAt(4) ? null : row.getLong(4),
                row.isNullAt(5) ? null : row.getString(5).toString());
    }

    private static String hash(java.nio.file.Path path) throws Exception {
        byte[] digest = MessageDigest.getInstance("SHA-256").digest(Files.readAllBytes(path));
        StringBuilder result = new StringBuilder();
        for (byte value : digest) {
            result.append(String.format("%02x", value & 255));
        }
        return result.toString();
    }

    private static Map<String, Object> export(FileStoreTable table) throws Exception {
        ReadBuilder read = table.newReadBuilder();
        List<Object> splits = new ArrayList<>();
        List<Object> allRows = new ArrayList<>();
        long snapshot = -1;
        for (Split input : read.newScan().plan().splits()) {
            DataSplit split = (DataSplit) input;
            snapshot = split.snapshotId();
            String partition = split.partition().isNullAt(0) ? null : split.partition().getString(0).toString();
            List<Object> files = new ArrayList<>();
            for (DataFileMeta file : split.dataFiles()) {
                java.nio.file.Path source = Paths.get(new Path(split.bucketPath(), file.fileName()).toUri().getPath());
                String name = COPIED.get(source.toString());
                if (name == null) {
                    name = "data-" + COPIED.size() + ".parquet";
                    Files.copy(source, output.resolve(name));
                    COPIED.put(source.toString(), name);
                }
                files.add(object("filePath", name, "fileSize", file.fileSize(), "rowCount", file.rowCount(),
                        "schemaId", file.schemaId(), "fileFormat", "parquet", "level", file.level(),
                        "minSequenceNumber", file.minSequenceNumber(), "maxSequenceNumber", file.maxSequenceNumber(),
                        "deleteRowCount", file.deleteRowCount().orElse(null), "creationTimeMs", file.creationTimeEpochMillis(),
                        "fileType", "DATA", "sourceType", file.fileSource().orElseThrow(IllegalStateException::new).name(),
                        "sha256", hash(output.resolve(name))));
            }
            List<Object> rows = new ArrayList<>();
            read.newRead().createReader(split).forEachRemaining(row -> rows.add(values(row)));
            allRows.addAll(rows);
            splits.add(object("snapshotId", snapshot, "bucket", split.bucket(), "partition", partition,
                    "rawConvertible", split.rawConvertible(), "files", files, "rows", rows));
        }
        return object("snapshotId", snapshot, "splits", splits, "rows", allRows);
    }

    public static void main(String[] args) throws Exception {
        if (args.length != 2) {
            throw new IllegalArgumentException("Usage: GenerateAppend <new warehouse directory> <new fixture directory>");
        }
        output = Paths.get(args[1]);
        Files.createDirectories(output);
        if (Files.exists(output.resolve("manifest.json"))) {
            throw new IllegalArgumentException("Use a fresh fixture directory");
        }
        try (Catalog catalog = CatalogFactory.createCatalog(CatalogContext.create(new Path(args[0])))) {
            catalog.createDatabase("fixture", false);
            Identifier id = Identifier.create("fixture", "append");
            Schema schema = Schema.newBuilder()
                    .column("id", DataTypes.INT())
                    .column("Payload", DataTypes.STRING())
                    .column("amount", DataTypes.DOUBLE())
                    .column("active", DataTypes.BOOLEAN())
                    .column("big", DataTypes.BIGINT())
                    .column("p", DataTypes.STRING())
                    .partitionKeys("p")
                    .option("bucket", "-1")
                    .option("file.format", "parquet")
                    .option("file.compression", "snappy")
                    .option("write-only", "true")
                    .build();
            catalog.createTable(id, schema, false);
            FileStoreTable table = (FileStoreTable) catalog.getTable(id);
            write(table, false,
                    row(1, "alpha", 1.25, true, 10000000000L, "east"),
                    row(2, "dup", 2.5, false, 20000000000L, "east"),
                    row(2, "dup", 2.5, false, 20000000000L, "east"),
                    row(3, null, null, null, null, "east"));
            write(table, false,
                    row(4, "beta", -3.5, true, Long.MIN_VALUE, "east"),
                    row(5, "", 0.0, false, Long.MAX_VALUE, "east"));
            write(table, false,
                    row(6, "null-partition", 6.0, true, 0L, null),
                    row(7, "empty-partition", 7.0, false, 7L, ""));
            Map<String, Object> append = export(table);

            // Model a caller-planned append COW rewrite: remove id=3, update
            // id=4, retain both id=2 rows, and replace only partition east.
            write(table, true,
                    row(1, "alpha", 1.25, true, 10000000000L, "east"),
                    row(2, "dup", 2.5, false, 20000000000L, "east"),
                    row(2, "dup", 2.5, false, 20000000000L, "east"),
                    row(4, "beta-updated", -3.5, true, Long.MIN_VALUE, "east"),
                    row(5, "", 0.0, false, Long.MAX_VALUE, "east"));
            Map<String, Object> cow = export(table);

            // Table writers elide empty files. Generate a valid zero-row file
            // with Paimon's format writer to exercise the executor's EOF path.
            java.nio.file.Path empty = output.resolve("empty.parquet");
            try (PositionOutputStream stream = table.fileIO().newOutputStream(new Path(empty.toString()), false);
                    FormatWriter writer = FileFormat.fromIdentifier("parquet", new Options())
                            .createWriterFactory(table.rowType()).create(stream, "snappy")) {
                // No rows.
            }
            Map<String, Object> emptyFile = object("filePath", "empty.parquet", "fileSize", Files.size(empty),
                    "rowCount", 0, "schemaId", table.schema().id(), "fileFormat", "parquet", "level", 0,
                    "minSequenceNumber", 0, "maxSequenceNumber", 0, "deleteRowCount", 0,
                    "creationTimeMs", 0, "fileType", "DATA", "sourceType", "APPEND", "sha256", hash(empty));
            Map<String, Object> result = object("paimonCommit", COMMIT, "paimonVersion", "2.2-SNAPSHOT",
                    "schemaId", table.schema().id(), "append", append, "cow", cow, "emptyFile", emptyFile);
            Files.write(output.resolve("manifest.json"), JsonSerdeUtil.toJson(result).getBytes(StandardCharsets.UTF_8));
            Files.write(output.resolve("schema.json"), table.schema().toString().getBytes(StandardCharsets.UTF_8));
        }
    }
}
