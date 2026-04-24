# Query-Scoped Registries in Velox

March 2026

## Motivation

Velox uses static mutable variables to store global registries for connectors,
functions, types, serializers, and operator translators. This design has two
problems:

1. **Thread safety.** Half of the registries have no synchronization. Concurrent
   registration from multiple threads is a data race.

2. **Single global scope.** All queries in a process share the same registries.
   It is impossible for different queries to use different connectors, function
   sets, or type systems. This limits embedding scenarios (e.g., Axiom running
   queries locally with custom connectors) and multi-tenant deployments where
   tenants may register different UDFs.

This document proposes replacing global static registries with query-scoped
registries, using a generic layered lookup mechanism. `QueryCtx` carries
per-query registry overrides, and each subsystem's lookup functions check
these overrides before falling back to the global default.

## Approach Preview

Global registration calls remain unchanged:

```cpp
connector::registerConnector(hiveConnector);
functions::prestosql::registerAllScalarFunctions();
```

Queries that need overrides create a per-query registry via the subsystem's API:

```cpp
auto registry = ConnectorRegistry::create(&ConnectorRegistry::global());
ConnectorRegistry::registerScoped(*registry, customHiveConnector);
queryCtx->setRegistry(RegistryKey::kConnectors, registry);
```

Subsystem lookups take a `QueryCtx&` parameter:

```cpp
// Before:
connector_ = connector::getConnector(connectorId);
// After:
connector_ = ConnectorRegistry::get(*queryCtx, connectorId);
```

The lookup checks per-query overrides first, then falls back to the global
registry. See [Proposed Design](#proposed-design) for details.

## Current State

### Engine Registries

These are used by the Velox execution engine to resolve plan nodes, evaluate
expressions, and serialize data. They are accessed via free functions
(`getConnector`, `getVectorSerde`, etc.) that read from function-local static
variables.

At a glance:

- **Connectors** (1 registry) — maps connector IDs to connector instances.
- **Functions** (6 registries) — scalar, aggregate, window functions, special forms, expression rewrites.
- **Types** (3 registries) — custom type factories and opaque type mappings.
- **Serialization** (2 registries) — default and named vector serdes.
- **Operators** (1 registry) — plan node translators.
- **Listeners** (3 registries) — task, split, and ExprSet listeners.
- **Filesystems** (1 registry) — maps URI schemes to filesystem implementations.
- **Exchange sources** (1 registry) — factories for creating exchange sources for inter-task data transfer.
- **Scan trackers** (1 registry) — tracks active scans for cache coordination.

#### Connector Registry

Maps connector IDs to connector instances. Used by `TableScan`, `TableWriter`,
and `IndexLookupJoin` operators to resolve data sources and sinks.

| Item | Detail |
|------|--------|
| Storage | `static unordered_map<string, shared_ptr<Connector>>` |
| Location | `velox/connectors/Connector.cpp:22` |
| Access | `registerConnector()`, `unregisterConnector()`, `getConnector()`, `getAllConnectors()` |
| Thread-safe | No |

#### Function Registries

Map function names to their implementations, signatures, and metadata. Used by
expression evaluation to resolve and invoke scalar, aggregate, and window
functions. Also includes special forms (e.g., `if`, `switch`, `try`) that
override default function call semantics, and expression rewrites that transform
expression trees before evaluation.

| Registry | Location | Access |
|----------|----------|--------|
| Vector functions | `velox/expression/VectorFunction.cpp:58` | `registerVectorFunction()`, `registerStatefulVectorFunction()`, `getVectorFunction()`, `resolveVectorFunction()` |
| Simple functions | `velox/expression/SimpleFunctionRegistry.cpp:23` | `registerSimpleFunction()`, `simpleFunctions()`, `resolveFunction()` |
| Aggregate functions | `velox/exec/Aggregate.cpp:41` | `registerAggregateFunction()`, `getAggregateFunctionEntry()`, `Aggregate::create()` |
| Window functions | `velox/exec/WindowFunction.cpp:23` | `registerWindowFunction()`, `getWindowFunctionSignatures()`, `WindowFunction::create()` |
| Special forms | `velox/expression/SpecialFormRegistry.cpp:18` | `registerFunctionCallToSpecialForm()`, `specialFormRegistry()` |
| Expression rewrites | `velox/expression/ExprRewriteRegistry.h:46` | `ExprRewriteRegistry::registerRewrite()`, `ExprRewriteRegistry::rewrite()` |

All function registries except window functions use `folly::Synchronized`
for thread safety. The window functions registry is an unprotected
`static unordered_map`.

#### Type Registries

Register custom and opaque types. Custom type factories resolve type names
(e.g., "json", "timestamp with time zone") to `Type` instances during parsing,
and provide custom CAST operator implementations used during expression
evaluation. Opaque type registries maintain bidirectional mappings between
string aliases and `std::type_index` for cross-process serialization.

| Registry | Location | Access |
|----------|----------|--------|
| Custom type factories | `velox/type/Type.cpp:~1124` | `registerCustomType()`, `getTypeFactory()`, `getCustomTypeCastOperator()` |
| Opaque type name-to-index | `velox/type/Type.cpp:~1133` | `registerOpaqueType()`, `getTypeIdForOpaqueTypeAlias()` |
| Opaque type index-to-name | `velox/type/Type.cpp:~1134` | `registerOpaqueType()`, `getOpaqueAliasForTypeId()` |

All three use unprotected `static unordered_map`.

#### Serialization Registries

Store vector serialization/deserialization implementations. Used for data
exchange between tasks (shuffle) and spill-to-disk.

| Registry | Location | Access |
|----------|----------|--------|
| Default vector serde | `velox/vector/VectorStream.cpp:64` | `registerVectorSerde()`, `getVectorSerde()` |
| Named vector serdes | `velox/vector/VectorStream.cpp:69` | `registerNamedVectorSerde()`, `getNamedVectorSerde()` |

Neither has synchronization. The default serde is a single `unique_ptr`; the
named serdes use an `unordered_map`.

#### Operator Registry

Maps plan node types to operator implementations. Used by the execution engine
to translate a plan tree into a pipeline of runnable operators.

| Registry | Location | Access |
|----------|----------|--------|
| Plan node translators | `velox/exec/Operator.cpp:142` | `Operator::registerOperator()`, `Operator::unregisterAllOperators()` |

Uses an unprotected `static vector<unique_ptr<PlanNodeTranslator>>`.

#### Listener Registries

Hooks for observability and diagnostics. Task listeners observe task lifecycle
events, split listeners observe split processing, and ExprSet listeners observe
expression evaluation.

| Registry | Location | Access |
|----------|----------|--------|
| Task listeners | `velox/exec/Task.cpp:86` | `registerTaskListener()`, `unregisterTaskListener()` |
| Split listener factories | `velox/exec/Task.cpp:92` | `registerSplitListenerFactory()`, `unregisterSplitListenerFactory()` |
| ExprSet listeners | `velox/expression/Expr.cpp:60` | `registerExprSetListener()`, `unregisterExprSetListener()` |

All three use `folly::Synchronized`.

#### Filesystem Registry

Maps URI schemes (e.g., `file:`, `s3://`, `hdfs://`) to filesystem
implementations. Used by connectors and spill-to-disk to read and write files.

| Registry | Location | Access |
|----------|----------|--------|
| Filesystem factories | `velox/common/file/FileSystems.cpp:33` | `registerFileSystem()`, `registerLocalFileSystem()`, `getFileSystem()` |

Uses an unprotected `static vector` of (scheme-matcher, factory) pairs.
Filesystems are resolved by iterating in registration order until a matcher
returns true.

#### Exchange Source Registry

Factories for creating exchange sources that transfer data between tasks
(shuffle). Each factory inspects the task ID and returns a source if it can
handle that scheme, or nullptr to pass to the next factory.

| Registry | Location | Access |
|----------|----------|--------|
| Exchange source factories | `velox/exec/ExchangeSource.cpp:35` | `ExchangeSource::registerFactory()`, `ExchangeSource::create()` |

Uses an unprotected `static vector<Factory>`.

#### Scan Tracker Registry

Tracks active scan operations for cache coordination across all connectors.

| Registry | Location | Access |
|----------|----------|--------|
| Scan trackers | `velox/connectors/Connector.cpp:67` | `Connector::getTracker()`, `Connector::unregisterTracker()` |

Uses `folly::Synchronized`. Defined as a static member of the `Connector` base
class.

#### Summary

Of the 19 engine registries:
- **10 are NOT thread-safe**: connector instances, window functions, all 3 type
  registries, both serde registries, operator translators, filesystem factories,
  exchange source factories.
- **9 are thread-safe**: vector/simple/aggregate functions, special forms,
  expression rewrites, all 3 listener registries, scan trackers.

### Connector-Internal Registries

These are implementation details of specific connector plugins. They are not
accessed by the engine — they are used internally by connector implementations
to resolve file formats, storage backends, and credentials.

Velox includes several connector implementations: HiveConnector (with Iceberg
and Paimon derivatives), TpchConnector, TpcdsConnector. Additional connectors
exist outside of the Velox repository. Of these, only the Hive connector
family has internal registries:

| Registry | Location | Access |
|----------|----------|--------|
| Reader factories | `velox/dwio/common/ReaderFactory.cpp:26` | `registerReaderFactory()`, `getReaderFactory()` |
| Writer factories | `velox/dwio/common/WriterFactory.cpp:25` | `registerWriterFactory()`, `getWriterFactory()` |
| S3 credential factories | `velox/connectors/hive/storage_adapters/s3fs/S3FileSystem.cpp:43` | `registerS3CredentialsProviderFactory()` |
| GCS credential factories | `velox/connectors/hive/storage_adapters/gcs/GcsFileSystem.cpp:38` | `registerGcsCredentialsProviderFactory()` |
| Azure client factories | `velox/connectors/hive/storage_adapters/abfs/AzureClientProviderFactories.cpp:21` | `registerAzureClientProviderFactory()` |
| S3 filesystem instances | `velox/connectors/hive/storage_adapters/s3fs/RegisterS3FileSystem.cpp:26` | via `registerS3FileSystem()` |
| GCS filesystem instances | `velox/connectors/hive/storage_adapters/gcs/RegisterGcsFileSystem.cpp:26` | via `registerGcsFileSystem()` |

Reader and writer factories are not thread-safe. The storage adapter registries
use `folly::Synchronized`.

The engine should not dictate the structure of connector-internal registries.
However, it should provide guidelines and reusable building blocks (see
Proposed Design) so that connector authors avoid repeating the pattern of
unsynchronized global statics.

### Axiom Registries

Axiom layers its own registries on top of Velox:

| Registry | Location | Access |
|----------|----------|--------|
| Connector metadata | `axiom/connectors/ConnectorMetadata.cpp:184` | `ConnectorMetadata::registerMetadata()`, `ConnectorMetadata::metadata()` |
| Function metadata | `axiom/optimizer/FunctionRegistry.cpp:236` | `FunctionRegistry::instance()`, `FunctionRegistry::registerPrestoFunctions()` |

Neither has synchronization.

Axiom maintains a parallel connector metadata registry because each connector
needs optimizer-specific metadata (table schemas, statistics, write operations)
that the Velox `Connector` interface does not provide. Registration follows a
dual pattern:

```cpp
// Register with Velox engine.
velox::connector::registerConnector(connector);

// Register with Axiom optimizer.
ConnectorMetadata::registerMetadata(connectorId, metadata);
```

Axiom's function registry stores optimizer-specific metadata (lambda functions,
reversible function pairs, aggregate empty result resolvers) that is separate
from Velox's function registries.

Both Axiom registries have the same problems as the unsynchronized Velox
registries: no thread safety and single global scope.

## Proposed Design

### Generic Layered Lookup: `ScopedRegistry<K, V>`

A reusable template that supports layered override: query-local entries take
precedence, with fallback to a parent scope (typically the global default).

```cpp
/// Layered key-value registry. Lookups check the local scope first, then
/// fall back to the parent chain. Supports arbitrary nesting (global ->
/// session -> query). All operations are synchronized.
///
/// @tparam K the key type. Must be copyable, hashable, and equality-comparable.
/// @tparam V the value type. Stored internally as shared_ptr<V>.
template <typename K, typename V>
class ScopedRegistry {
 public:
  using VPtr = std::shared_ptr<V>;

  /// Create a root registry (no parent).
  ScopedRegistry() : parent_(nullptr) {}

  /// Create a derived scope that falls back to 'parent' when a key is not
  /// found locally. The parent must outlive this registry.
  explicit ScopedRegistry(const ScopedRegistry* parent) : parent_(parent) {}

  /// Insert an entry in the local scope. Returns true if the key was newly
  /// inserted, false if it already existed and was overwritten. Throws if
  /// the key already exists unless 'overwrite' is true.
  bool insert(K key, VPtr value, bool overwrite = false) {
    return local_.withWLock([&](auto& map) {
      auto [it, inserted] = map.emplace(std::move(key), std::move(value));
      if (!inserted) {
        VELOX_CHECK(overwrite, "Key '{}' already registered", it->first);
        it->second = std::move(value);
      }
      return inserted;
    });
  }

  /// Look up a key. Check local scope first, then walk the parent chain.
  /// Returns nullptr if not found.
  VPtr find(const K& key) const {
    auto result = local_.withRLock([&](const auto& map) -> VPtr {
      auto it = map.find(key);
      if (it != map.end()) {
        return it->second;
      }
      return nullptr;
    });
    if (result) {
      return result;
    }
    return parent_ ? parent_->find(key) : nullptr;
  }

 private:
  folly::Synchronized<folly::F14FastMap<K, VPtr>> local_;
  const ScopedRegistry* parent_;
};
```

All operations are synchronized. Even a single query executes on multiple
threads, so any registry must be safe for concurrent access.

`ScopedRegistry` supports two usage modes:

- **Override mode** (parent != nullptr): The query-scoped registry inherits all
  entries from the parent (typically the global default) and selectively
  overrides specific entries. Lookups that miss locally fall through to the
  parent. Use this when the query needs mostly the same registrations as the
  global scope with a few changes.

- **Isolation mode** (parent == nullptr): The query-scoped registry is
  self-contained. Only explicitly registered entries are visible — no fallback.
  Use this when the query needs full control over its registry contents (e.g.,
  running PrestoSQL vs SparkSQL queries with entirely different function sets).

This template handles all map-shaped registries. The few non-map registries
(default vector serde — a single value; operator translators and listeners —
ordered vectors) need separate treatment, potentially `ScopedValue<T>` and
`ScopedList<T>` variants.

### Engine Registries on `QueryCtx`

`QueryCtx` lives in `velox/core/`, which has no dependency on `velox/connectors/`,
`velox/exec/`, or `velox/expression/`. Adding typed registry fields to `QueryCtx`
would create circular dependencies or pull heavy transitive deps into
`velox/core/`.

Instead, `QueryCtx` provides a generic mechanism to store per-query registry
overrides. Each subsystem owns its registry type and scoping logic. `QueryCtx`
just carries an opaque bag of overrides keyed by subsystem:

```cpp
/// Well-known registry keys. Each subsystem defines its own key.
/// Subsystems outside of velox/ (e.g., Axiom) define additional keys
/// in their own headers.
struct RegistryKey {
  static constexpr std::string_view kConnectors = "connectors";
  static constexpr std::string_view kVectorFunctions = "vectorFunctions";
  static constexpr std::string_view kSimpleFunctions = "simpleFunctions";
  static constexpr std::string_view kAggregateFunctions = "aggregateFunctions";
  static constexpr std::string_view kWindowFunctions = "windowFunctions";
  static constexpr std::string_view kCustomTypes = "customTypes";
  static constexpr std::string_view kVectorSerde = "vectorSerde";
  static constexpr std::string_view kFilesystems = "filesystems";
  static constexpr std::string_view kExchangeSources = "exchangeSources";
  // ...
};

class QueryCtx {
 public:
  /// Store a per-query registry override. Returns true if the key was newly
  /// inserted, false if it already existed and was overwritten. Throws if
  /// the key already exists unless 'overwrite' is true.
  template <typename T>
  bool setRegistry(
      std::string_view key,
      std::shared_ptr<T> registry,
      bool overwrite = false) {
    return registries_.withWLock([&](auto& map) {
      Entry entry{std::move(registry), std::type_index(typeid(T))};
      auto [it, inserted] =
          map.emplace(std::string(key), std::move(entry));
      if (!inserted) {
        VELOX_CHECK(overwrite, "Registry '{}' already set", key);
        it->second = std::move(entry);
      }
      return inserted;
    });
  }

  /// Retrieve a per-query registry override. Returns nullptr if no override
  /// was set for this key. Asserts that the stored type matches T exactly
  /// (inheritance is not supported — store and retrieve using the same type).
  template <typename T>
  std::shared_ptr<T> registry(std::string_view key) const {
    return registries_.withRLock([&](const auto& map) -> std::shared_ptr<T> {
      auto it = map.find(std::string(key));
      if (it != map.end()) {
        VELOX_CHECK(
            it->second.type == std::type_index(typeid(T)),
            "Registry type mismatch for key '{}': expected {}, got {}",
            key, typeid(T).name(), it->second.type.name());
        return std::static_pointer_cast<T>(it->second.registry);
      }
      return nullptr;
    });
  }

 private:
  struct Entry {
    std::shared_ptr<void> registry;
    std::type_index type;
  };

  folly::Synchronized<folly::F14FastMap<std::string, Entry>> registries_;
};
```

Each subsystem wraps its registry behind a class with static methods for
creating registries, registering entries, and looking up values. Callers
never construct `ScopedRegistry` directly.

```cpp
// In velox/connectors/Connector.h:
class ConnectorRegistry {
 public:
  using Registry = ScopedRegistry<std::string, Connector>;

  /// Create a per-query registry. If 'parent' is provided, lookups fall
  /// back to it. Pass nullptr for isolation mode (no fallback).
  static std::shared_ptr<Registry> create(
      const Registry* parent = nullptr);

  /// Return the global registry (root scope).
  static Registry& global();

  /// Register a connector in the global registry.
  static void registerGlobal(std::shared_ptr<Connector> connector);

  /// Register a connector in a specific registry.
  static void registerScoped(
      Registry& registry,
      std::shared_ptr<Connector> connector);

  /// Look up a connector. Checks per-query override on QueryCtx first,
  /// falls back to the global registry if no override is set.
  static std::shared_ptr<Connector> get(
      const core::QueryCtx& queryCtx,
      const std::string& connectorId);
};
```

This approach preserves the existing dependency graph — `velox/core/` does not
need to know about connector, function, or operator types. Each subsystem
depends on `velox/core/` (for `QueryCtx`), not the other way around.

For backward compatibility, existing global free functions
(`registerConnector`, `getConnector`, etc.) are preserved and reimplemented
to delegate to the new registry classes:

```cpp
// Existing API — preserved, delegates to ConnectorRegistry.
void registerConnector(std::shared_ptr<Connector> connector) {
  ConnectorRegistry::registerGlobal(std::move(connector));
}
```

### Example: End-to-End Flow

#### Process startup: populate global registries

```cpp
// These calls remain unchanged — they delegate to the new registry classes.
connector::registerConnector(hiveConnector);
connector::registerConnector(tpchConnector);
functions::prestosql::registerAllScalarFunctions();
filesystems::registerLocalFileSystem();
dwrf::registerDwrfReaderFactory();
```

#### Query setup: create per-query overrides (optional)

```cpp
// Most queries use global defaults — no overrides needed.
auto queryCtx = QueryCtx::create(executor, queryConfig);

// A query that needs a custom connector (e.g., different S3 endpoint).
// The per-query registry uses the global as its parent for fallback.
auto registry = ConnectorRegistry::create(&ConnectorRegistry::global());
ConnectorRegistry::registerScoped(*registry, customHiveConnector);
queryCtx->setRegistry(RegistryKey::kConnectors, registry);

// A query that needs additional UDFs:
auto funcRegistry = VectorFunctionRegistry::create(
    &VectorFunctionRegistry::global());
VectorFunctionRegistry::registerScoped(*funcRegistry, "my_udf", myUdfEntry);
queryCtx->setRegistry(RegistryKey::kVectorFunctions, funcRegistry);
```

#### Query execution: lookup with fallback

```cpp
// In operator code (e.g., TableScan):
// Before:
connector_ = connector::getConnector(connectorId);
// After:
connector_ = ConnectorRegistry::get(*queryCtx, connectorId);

// ConnectorRegistry::get checks QueryCtx for a per-query registry
// override, falls back to the global registry if none is set.
```

#### Axiom: query-scoped optimizer registries

```cpp
// Axiom defines its own registry class following the same pattern.
auto registry = ConnectorMetadataRegistry::create();
ConnectorMetadataRegistry::registerScoped(*registry, "hive", hiveMetadata);
ConnectorMetadataRegistry::registerScoped(*registry, "tpch", tpchMetadata);

queryCtx->setRegistry(AxiomRegistryKey::kConnectorMetadata, registry);
```

### Connector-Internal Registry Guidelines

The engine does not dictate the internal structure of connectors. However,
connector authors should follow these guidelines:

1. **Do not use static mutable state.** Store registries as members of the
   `Connector` subclass, not in function-local statics or global variables.

2. **Use `ScopedRegistry` for internal registries** that may need layered
   override (e.g., reader/writer factories, credential providers).

3. **Accept per-query configuration via `ConnectorQueryCtx`.** Static
   connector-wide settings go in the constructor config. Query-varying settings
   go in session properties.

4. **Thread safety.** All registries must be synchronized. Even a single query
   executes on multiple threads.

For the Hive connector specifically, this means moving reader/writer factory
registration from global statics (`ReaderFactory.cpp`, `WriterFactory.cpp`)
into the `HiveConnector` instance.

### Axiom Migration

Axiom's two registries should follow the same pattern:

- **Connector metadata.** Currently a parallel global registry. Should be moved
  to query-scoped access using the same `QueryCtx` override mechanism. Velox
  cannot depend on optimizer-level abstractions, so this stays as a separate
  registry.

- **Function metadata.** Currently a global singleton. Should be query-scoped,
  paralleling the Velox function registries.

## Migration Plan

### Phase 1: Thread Safety (Non-breaking)

Wrap the 10 unsynchronized engine registries in `folly::Synchronized`. No API
changes — just adds safety for concurrent access.

### Phase 2: `ScopedRegistry` Primitive

Implement `ScopedRegistry<K, V>` in `velox/common/`. Add unit tests. This is a
pure addition with no migration required.

### Phase 3: Query-Scoped Engine Registries

Add the generic registry override mechanism to `QueryCtx`. Introduce registry
classes for each subsystem (e.g., `ConnectorRegistry`) with static methods for
global registration, scoped registration, and lookup. Reimplement existing
global free functions (`registerConnector`, `getConnector`, etc.) to delegate
to the new registry classes for backward compatibility.

### Phase 4: Connector-Internal Migration

Move Hive connector's reader/writer factories from global statics into the
`HiveConnector` instance. Publish connector authoring guidelines.

### Phase 5: Axiom Migration

Move Axiom's connector metadata and function metadata registries to
query-scoped access.

## Open Questions

1. **Listener registries.** Task, split, and ExprSet listeners are currently
   global and thread-safe. Should they be query-scoped? Listeners are
   observability hooks — it is unclear whether per-query listener sets are
   useful.

2. **Operator translators.** These are an ordered vector, not a map. A
   `ScopedList<T>` variant could concatenate local + parent entries, but the
   ordering semantics need thought (local-first? interleaved by priority?).

3. **Default vector serde.** This is a single value, not a map. Rather than
   introducing a `ScopedValue<T>` variant, the default serde could be replaced
   with a configuration option specifying the serde name, resolved via the
   named serdes registry.


