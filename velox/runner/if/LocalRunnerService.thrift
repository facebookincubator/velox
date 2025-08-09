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

/**
 * Represents a scalar value in a result row.
 */
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
  10: i64 timestampValue;
  11: i32 dateValue;
}

/**
 * Represents a cell in a result row.
 */
struct Cell {
  1: optional ScalarValue value;
  2: bool isNull = false;
}

/**
 * Represents a row in the result set.
 */
struct ResultRow {
  1: list<Cell> cells;
}

/**
 * Represents a batch of rows in the result set.
 */
struct ResultBatch {
  1: list<ResultRow> rows;
  2: list<string> columnNames;
  3: list<string> columnTypes;
}

/**
 * Request to execute a query plan.
 */
struct ExecutePlanRequest {
  1: string serializedPlan;
  2: string queryId;
  3: i32 numWorkers = 4;
  4: i32 numDrivers = 2;
}

/**
 * Response from executing a query plan.
 */
struct ExecutePlanResponse {
  1: list<ResultBatch> resultBatches;
  2: string output;
  3: bool success;
  4: optional string errorMessage;
}

/**
 * Service for executing Velox query plans.
 */
service LocalRunnerService {
  /**
   * Execute a serialized query plan and return the results.
   */
  ExecutePlanResponse executePlan(1: ExecutePlanRequest request);
}
