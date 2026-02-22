/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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
// Results are returned using Presto's binary serialization format for efficient
// data transfer.

namespace cpp2 facebook.velox.runner

// Represents a batch of rows using Presto's binary serialization format.
// The serialized data can be deserialized using PrestoVectorSerde to reconstruct
// the original RowVector.
struct Batch {
  // Binary serialized RowVector data in Presto format
  1: binary serializedData;
  // Column names in the RowVector
  2: list<string> columnNames;
  // Column type strings in the RowVector
  3: list<string> columnTypes;
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
