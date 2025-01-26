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
# "Bad Data" files

These are files used for reproducing various bugs that have been reported.

* PARQUET-1481.parquet: tests a case where a schema Thrift value has been
  corrupted.
* ARROW-RS-GH-6229-DICTHEADER.parquet: tests a case where the number of values
  stored in dictionary page header is negative.
* ARROW-RS-GH-6229-LEVELS.parquet: tests a case where a page has insufficient 
  repetition levels.
* ARROW-GH-41321.parquet: test case of https://github.com/apache/arrow/issues/41321
  where decoded rep / def levels is less than num_values in page_header.
* ARROW-GH-41317.parquet: test case of https://github.com/apache/arrow/issues/41317
  where all columns have not the same size.
* ARROW-GH-43605.parquet: dictionary index page uses rle encoding but 0 as rle bit-width.
