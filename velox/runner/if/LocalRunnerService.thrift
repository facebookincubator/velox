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

namespace cpp2 facebook.velox.runner

struct i128 {
  1: i64 msb;
  2: i64 lsb;
}

struct Timestamp {
  1: i64 seconds;
  2: i64 nanos;
}

union ScalarValue {
  1: bool boolValue;
  2: byte tinyintValue;
  3: i16 smallintValue;
  4: i32 integerValue;
  5: i64 bigintValue;
  6: float realValue;
  7: double doubleValue;
  8: string varcharValue;
  9: binary varbinaryValue;
  10: Timestamp timestampValue;
  11: i128 hugeintValue;
}

struct Array {
  1: list<Value> values;
}

struct Map {
  1: map<Value, Value> values;
}

struct Row {
  1: list<string> filedNames;
  2: list<Value> fieldValues;
}

union ComplexValue {
  1: Array arrayValue;
  2: Map mapValue;
  3: Row rowValue;
}

struct Value {
  1: optional ScalarValue scalarValue;
  2: optional ComplexValue complexValue;
  3: bool isNull;
}

struct Column {
  1: list<Value> rows;
}

struct ResultBatch {
  1: list<Column> columns;
  2: list<string> columnNames;
  3: list<string> columnTypes;
  4: i32 numRows;
}

struct ExecutePlanRequest {
  1: string serializedPlan;
  2: string queryId;
  3: i32 numWorkers = 4;
  4: i32 numDrivers = 2;
}

struct ExecutePlanResponse {
  1: list<ResultBatch> resultBatches;
  2: string output;
  3: bool success;
  4: optional string errorMessage;
}

service LocalRunnerService {
  ExecutePlanResponse executePlan(1: ExecutePlanRequest request);
}
