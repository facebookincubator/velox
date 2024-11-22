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

`map_no_value.parquet` is generated with parquet-rs version 53.2.0
using the following code:
```
    fn main() {
        use crate::data_type::Int32Type;
        use crate::file::properties::{EnabledStatistics, WriterProperties};
        use crate::file::writer::SerializedFileWriter;
        use crate::schema::parser::parse_message_type;
        use std::sync::Arc;

        let schema = "
            message schema {
                REQUIRED group my_map (MAP) {
                    REPEATED group key_value {
                        REQUIRED INT32 key;
                        OPTIONAL INT32 value;
                    }
                }
                REQUIRED group my_map_no_v (MAP) {
                    REPEATED group key_value {
                        REQUIRED INT32 key;
                    }
                }
                REQUIRED group my_list (LIST) {
                    REPEATED group list {
                        REQUIRED INT32 element;
                    }
                }
            }
            ";
        let schema = Arc::new(parse_message_type(schema).unwrap());

        // Write Parquet file to buffer
        let mut file = std::fs::File::create("/tmp/map_no_value.parquet").unwrap();
        let props = Arc::new(
            WriterProperties::builder()
                .set_statistics_enabled(EnabledStatistics::None)
                .build(),
        );
        let mut file_writer = SerializedFileWriter::new(&mut file, schema, props).unwrap();
        let mut row_group_writer = file_writer.next_row_group().unwrap();

        // Write column my_map.key_value.key
        let mut column_writer = row_group_writer.next_column().unwrap().unwrap();
        column_writer
            .typed::<Int32Type>()
            .write_batch(
                &[1, 2, 3, 4, 5, 6, 7, 8, 9],
                Some(&[1, 1, 1, 1, 1, 1, 1, 1, 1]),
                Some(&[0, 1, 1, 0, 1, 1, 0, 1, 1]),
            )
            .unwrap();
        column_writer.close().unwrap();

        // Write column my_map.key_value.value
        let mut column_writer = row_group_writer.next_column().unwrap().unwrap();
        column_writer
            .typed::<Int32Type>()
            .write_batch(
                &[],
                Some(&[1, 1, 1, 1, 1, 1, 1, 1, 1]),
                Some(&[0, 1, 1, 0, 1, 1, 0, 1, 1]),
            )
            .unwrap();
        column_writer.close().unwrap();

        // Write column my_map_no_v.key_value.key
        let mut column_writer = row_group_writer.next_column().unwrap().unwrap();
        column_writer
            .typed::<Int32Type>()
            .write_batch(
                &[1, 2, 3, 4, 5, 6, 7, 8, 9],
                Some(&[1, 1, 1, 1, 1, 1, 1, 1, 1]),
                Some(&[0, 1, 1, 0, 1, 1, 0, 1, 1]),
            )
            .unwrap();
        column_writer.close().unwrap();

        // Write column my_list.list.element
        let mut column_writer = row_group_writer.next_column().unwrap().unwrap();
        column_writer
            .typed::<Int32Type>()
            .write_batch(
                &[1, 2, 3, 4, 5, 6, 7, 8, 9],
                Some(&[1, 1, 1, 1, 1, 1, 1, 1, 1]),
                Some(&[0, 1, 1, 0, 1, 1, 0, 1, 1]),
            )
            .unwrap();
        column_writer.close().unwrap();

        // Finalize Parquet file
        row_group_writer.close().unwrap();
        file_writer.close().unwrap();
    }
```

It contains a MAP with all null values, a second MAP without a `values` field, and
an equivalent LIST repeating the MAP keys. The first column is 3 MAP rows:
```
{1 -> null, 2 -> null, 3 -> null}
{4 -> null, 5 -> null, 6 -> null}
{7 -> null, 8 -> null, 9 -> null}
```

The last two columns comprise 3 equivalent rows of `list<Integer>`:
```
[1, 2, 3]
[4, 5, 6]
[7, 8, 9]
```

Here is the file metadata printed by parquet-cli:
```
File path:  map_no_value.parquet
Created by: parquet-rs version 53.2.0
Properties: (none)
Schema:
message schema {
  required group my_map (MAP) {
    repeated group key_value {
      required int32 key;
      optional int32 value;
    }
  }
  required group my_map_no_v (MAP) {
    repeated group key_value {
      required int32 key;
    }
  }
  required group my_list (LIST) {
    repeated group list {
      required int32 element;
    }
  }
}


Row group 0:  count: 3  105.00 B records  start: 4  total(compressed): 315 B total(uncompressed):315 B
--------------------------------------------------------------------------------
                           type      encodings count     avg size   nulls   min / max
my_map.key_value.key       INT32     _ RR_     9         10.00 B            
my_map.key_value.value     INT32     _ RR_     9         5.00 B             
my_map_no_v.key_value.key  INT32     _ RR_     9         10.00 B            
my_list.list.element       INT32     _ RR_     9         10.00 B
