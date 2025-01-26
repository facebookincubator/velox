<!--
  ~ Licensed to the Apache Software Foundation (ASF) under one
  ~ or more contributor license agreements.  See the NOTICE file
  ~ distributed with this work for additional information
  ~ regarding copyright ownership.  The ASF licenses this file
  ~ to you under the Apache License, Version 2.0 (the
  ~ "License"); you may not use this file except in compliance
  ~ with the License.  You may obtain a copy of the License at
  ~
  ~   http://www.apache.org/licenses/LICENSE-2.0
  ~
  ~ Unless required by applicable law or agreed to in writing,
  ~ software distributed under the License is distributed on an
  ~ "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
  ~ KIND, either express or implied.  See the License for the
  ~ specific language governing permissions and limitations
  ~ under the License.
  -->

`old_list_structure.parquet` is generated with parquet-java version 1.14.3.
It contains a `LIST<LIST<INT32>>` column with a single value `[[1, 2], [3, 4]]`
using the legacy two-level structure encoding for list type.

The file is created by the following Java code:
```java
package org.example;

import org.apache.avro.Schema;
import org.apache.avro.generic.GenericData;
import org.apache.avro.generic.GenericRecord;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.parquet.avro.AvroParquetWriter;
import org.apache.parquet.hadoop.ParquetWriter;

import java.io.IOException;
import java.util.Arrays;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class TwoLevelList {

  public static void main(String[] args) {
    Schema schema = new Schema.Parser().parse("{"
      + "\"type\":\"record\","
      + "\"name\":\"my_record\","
      + "\"fields\":["
      + "  {"
      + "    \"name\":\"a\","
      + "    \"type\":{\"type\":\"array\", \"items\":{\"type\":\"array\", \"items\":\"int\"}}"
      + "  }"
      + "]"
      + "}");

    GenericRecord record = new GenericData.Record(schema);

    // Write [[1, 2], [3, 4]] to the avro record
    record.put("a", Stream.of(Arrays.asList(1, 2), Arrays.asList(3, 4))
      .map(list -> {
        Schema innerListType = schema.getField("a").schema().getElementType();
        GenericData.Array<Integer> innerList = new GenericData.Array<>(list.size(), innerListType);
        innerList.addAll(list);
        return innerList;
      }).collect(Collectors.toList()));

    Path file = new Path("/tmp/old_list_structure.parquet");
    Configuration conf = new Configuration();
    conf.set("parquet.avro.write-old-list-structure", "true");  // this is the default value
    try (ParquetWriter<GenericRecord> writer = AvroParquetWriter.<GenericRecord>builder(file)
      .withSchema(schema)
      .withConf(conf)
      .build()) {
      writer.write(record);
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

}
```

Here is the file metadata printed by parquet-cli:
```
File path:  /tmp/old_list_structure.parquet
Created by: parquet-mr version 1.14.3 (build b5e376a2caee767a11e75b783512b14cf8ca90ec)
Properties:
  parquet.avro.schema: {"type":"record","name":"my_record","fields":[{"name":"a","type":{"type":"array","items":{"type":"array","items":"int"}}}]}
    writer.model.name: avro
Schema:
message my_record {
  required group a (LIST) {
    repeated group array (LIST) {
      repeated int32 array;
    }
  }
}


Row group 0:  count: 1  53.00 B records  start: 4  total(compressed): 53 B total(uncompressed):53 B 
--------------------------------------------------------------------------------
               type      encodings count     avg size   nulls   min / max
a.array.array  INT32     _   _     4         13.25 B    0       "1" / "4"
```
