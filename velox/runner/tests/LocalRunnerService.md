# LocalRunnerService

This service provides an HTTP interface to the LocalRunner functionality, allowing you to run Velox queries via network requests. It accepts serialized PlanNode objects and executes them using the LocalRunner.

## Building the Service

To build the LocalRunnerService, use the following command:

```bash
buck build //velox/runner/tests:velox_local_runner_service
```

## Running the Service

After building, you can run the service with:

```bash
./buck-out/gen/velox/runner/tests/velox_local_runner_service
```

By default, the service listens on port 8080. You can change this using the `--port` flag:

```bash
./buck-out/gen/velox/runner/tests/velox_local_runner_service --port 9090
```

### Available Command Line Options

- `--port`: Port to listen on (default: 8080)
- `--registry`: Function registry to use for query evaluation. Currently supported values are "presto" and "spark". Default is "presto".
- `--num_workers`: Number of workers to use for query execution (default: 4)
- `--num_drivers`: Number of drivers per worker (default: 2)

## API Usage

The service exposes a single HTTP endpoint that accepts POST requests with JSON payloads.

### Execute a Query

**Endpoint**: `/` (root)

**Method**: POST

**Request Body**:

```json
{
  "serialized_plan": "<serialized PlanNode JSON>",
  "query_id": "optional-query-id",
  "num_workers": 4,
  "num_drivers": 2
}
```

- `serialized_plan`: (required) A JSON string containing the serialized PlanNode
- `query_id`: (optional) A unique identifier for the query (default: "query")
- `num_workers`: (optional) Number of workers to use (default: value from --num_workers flag)
- `num_drivers`: (optional) Number of drivers per worker (default: value from --num_drivers flag)

**Response**:

```json
{
  "status": "success",
  "output": "<captured stdout>",
  "results": [
    [
      { "column1": "value1", "column2": 123 },
      { "column1": "value2", "column2": 456 }
    ]
  ],
  "stats": {
    "num_tasks": 1
  }
}
```

- `status`: "success" or "error"
- `output`: Captured stdout during query execution
- `results`: Array of result batches, each containing an array of row objects
- `stats`: Statistics about the query execution

### Error Response

```json
{
  "status": "error",
  "message": "Error message"
}
```

## Example

```bash
curl -X POST http://localhost:8080/ \
  -H "Content-Type: application/json" \
  -d '{
    "serialized_plan": "{\"id\":\"0\",\"type\":\"values\",\"names\":[\"a\",\"b\"],\"values\":[[1,\"x\"],[2,\"y\"],[3,\"z\"]]}"
  }'
```

This example sends a simple VALUES plan node that returns a small result set.
