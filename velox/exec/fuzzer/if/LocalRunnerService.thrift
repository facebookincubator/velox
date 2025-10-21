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

// This file defines a Thrift service for executing Velox query plans remotely.
// It provides a type system for representing query results and a service interface
// for executing serialized query plans with configurable parallelism.

namespace cpp2 facebook.velox.runner

// Represents a HUGEINT value by splitting it into most significant and least
// significant components.
struct i128 {
  1: i64 msb;
  2: i64 lsb;
}

// Represents a timestamp value with seconds and nanoseconds components.
struct Timestamp {
  1: i64 seconds;
  2: i64 nanos;
}

// A tagged union representing all supported scalar (primitive) data types.
// Only one field will be set at a time, corresponding to the actual type of the value.
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

// Represents an ARRAY type, containing an ordered list of values.
// All values in the array are of the same type.
struct Array {
  1: list<Value> values;
}

// Represents a MAP type, containing key-value pairs.
// Keys and values can be of any supported type.
struct Map {
  1: map<Value, Value> values;
}

// Represents a ROW (struct) type, containing an ordered list of field values.
// Each field can have a different type.
struct Row {
  1: list<Value> fieldValues;
}

// A tagged union representing complex (nested) data types.
// Only one field will be set at a time, corresponding to the actual complex type.
union ComplexValue {
  1: Array arrayValue;
  2: Map mapValue;
  3: Row rowValue;
}

// Represents a single value of any supported data type.
// A value can be:
// - A scalar (primitive) value
// - A complex (nested) value
// - NULL (indicated by isNull = true)
struct Value {
  1: optional ScalarValue scalarValue;
  2: optional ComplexValue complexValue;
  3: bool isNull;
}

// Represents a single column of data in columnar format.
// Contains all values for one column across multiple rows.
struct Column {
  1: list<Value> rows;
}

// Represents a batch of rows in columnar format.
// This is the fundamental unit of data transfer, containing multiple columns
// and metadata about the schema.
struct Batch {
  1: list<Column> columns;
  2: list<string> columnNames;
  3: list<string> columnTypes;
  4: i32 numRows;
}

// Request to execute a serialized Velox query plan.
struct ExecutePlanRequest {
  1: string serializedPlan;
  2: string queryId;
  3: i32 numWorkers = 4;
  4: i32 numDrivers = 2;
}

// Response from executing a query plan.
struct ExecutePlanResponse {
  1: list<Batch> results;
  2: string output;
  3: bool success;
  4: optional string errorMessage;
}

// Service for executing Velox query plans locally.
// This service enables remote execution of serialized query plans with
// configurable parallelism, returning results in a structured format.
service LocalRunnerService {
  // Inputs a Thrift request and executes a serialized Velox query plan and
  // returns the results as a Thrift response.
  ExecutePlanResponse execute(1: ExecutePlanRequest request);
}
