# Copyright 2014-present Facebook. All Rights Reserved.

namespace cpp2 facebook.velox.functions.remote

cpp_include "folly/io/IOBuf.h"

typedef binary (cpp2.type = "folly::IOBuf") IOBuf

/// The format used to serialize buffers/payloads.
enum PageFormat {
  PRESTO_PAGE = 1,
  SPARK_UNSAFE_ROW = 2,
}

/// Identifies the remote function being called.
struct RemoteFunctionHandle {
  /// The function name
  1: string name;

  /// The function return and argument types. The types are serialized using
  /// Velox's type serialization format.
  2: string returnType;
  3: list<string> argumentTypes;
}

/// A page of data to be sent to, or got as a return from a remote function
/// execution.
struct RemoteFunctionPage {
  /// The serialization format used to encode the payload.
  1: PageFormat pageFormat;

  /// The actual data.
  2: IOBuf payload;

  /// The number of logical rows in this page.
  3: i64 rowCount;
}

/// The parameters passed to the remote thrift call.
struct RemoteFunctionRequest {
  /// Function handle to identify te function being called.
  1: RemoteFunctionHandle functionHandle;

  /// The page containing serialized input parameters.
  2: RemoteFunctionPage inputs;

  /// Whether the function is supposed to throw an exception if errors are
  /// found, or capture exceptions and return back to the user. This is used
  /// to implement special forms (if the function is inside a try() construct,
  /// for example).
  ///
  /// TODO: the format to return serialized exceptions back to the client needs
  /// to be defined and implemented.
  3: bool throwOnError;
}

/// Statistics that may be returned from server to client.
struct RemoteFunctionStats {
  1: map<string, string> stats;
}

/// Structured returned from server to client. Contains the serialized output
/// page and optional statistics.
struct RemoteFunctionResponse {
  1: RemoteFunctionPage result;
  2: optional RemoteFunctionStats remoteFunctionStats;
}

/// Service definition.
service RemoteFunctionService {
  RemoteFunctionResponse invokeFunction(1: RemoteFunctionRequest request);
}
