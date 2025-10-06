# CuDF Exchange

The CuDF Exchange is a replacement for the inter-worker velox exchange that happens between tasks ending in a partitioned output operator and tasks having an exchange operator as source node. In Velox, this exchange is partially implemented; both the exchange server and the actual exchange source object implementations are outside the scope of Velox.

The velox cudf exchange implements all the necessary components to efficiently transfer cudf-vectors between tasks. At the core is a UCXX based transfer that directly copies the raw vector data from GPU memory to GPU memory.

This code is experimental and in its early stages.