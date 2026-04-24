# gpu_portable

A build-time code generator that lifts a Velox CPU function into
**JIT-compilable CUDA source code**. The generator runs as part of the
normal CMake build, emits a C++ header per function, and the header
exposes one thing: a getter that returns a string of CUDA source ready
to hand to cuDF's `transform_extended` (or any other NVRTC-based JIT
entry point).

The goal is to keep the CPU and GPU implementations of **simple** functions
in sync by having a single source of truth.

## Typical flow

1. The developer writes one `velox_add_gpu_portable(...)` registration
   in `CMakeLists.txt`, pointing at the CPU function to lift.
2. At build time, the generator emits
   `${CMAKE_BINARY_DIR}/velox/experimental/cudf/gpu_portable/<OUTPUT>.h`
   containing a single inline function `<WRAPPER_NAME>_source(...)` that
   returns a CUDA source string.
3. Consumer code includes that header and passes the source string to
   cuDF's JIT path:

```cpp
#include "velox/experimental/cudf/gpu_portable/Round.h"

// gpu_portable::velox_round_double_source(scale) returns the CUDA
// source for a self-contained `__device__` function. Hand it to
// cudf::transform_extended so NVRTC compiles and runs it against the
// input column.
return cudf::transform_extended(
    inputs,
    gpu_portable::velox_round_double_source(scale),
    outputType,
    cudf::udf_source_type::CUDA,
    /* ... other cuDF args ... */);
```

That is the entire intended use: produce the source string, feed it to
cuDF.

## The registration

### Shape of a typical CPU function being lifted

```cpp
// In some Velox header, e.g. velox/functions/prestosql/ArithmeticImpl.h:

template <typename T, typename U, bool Flag = false>   // template params
R cpu_function(
    T runtimeArg,     // per-row input
    U planArg);       // value known at plan time (e.g. a SQL literal)
// Returns R.
```

### Fields

| Field | What it is | What it corresponds to in the CPU function |
| --- | --- | --- |
| `SOURCE` | Absolute path to the Velox header. | The file where the function is defined. |
| `FUNC` | Name of the function to lift. | `cpu_function` above. |
| `TEMPLATE_TYPES` | `Tname=ConcreteType` entries, one per template type parameter. | Binds `T`, `U` to concrete types. |
| `TEMPLATE_VALUES` | `name=literal` entries, one per template non-type parameter. | Binds `Flag` to a concrete value. |
| `OUT_PARAM` | Declaration of the out-pointer on the emitted `__device__` function, e.g. `"R* out"`. | The return value of the CPU function is written here instead of returned. |
| `INPUT_PARAMS` | Declarations of per-row inputs on the emitted `__device__` function, e.g. `"T runtimeArg"`. | Each CPU parameter whose value varies per row. |
| `BAKED_PARAMS` | Declarations of values that are constant for the whole kernel launch, e.g. `"U planArg"`. The name must match the CPU parameter name. | Each CPU parameter whose value is known at plan time. The generator splices its value as a literal into the source at call time so NVRTC can constant-fold. |
| `WRAPPER_NAME` | Name of the emitted `__device__` function. The getter is called `<WRAPPER_NAME>_source`. | Chosen by the registration; typically `velox_<func>_<type>`. |
| `OUTPUT` | Name of the generated header file. | Chosen by the registration, e.g. `Round.h`. |

### What the generator emits

The header at `${CMAKE_BINARY_DIR}/velox/experimental/cudf/gpu_portable/<OUTPUT>`
contains one function:

```cpp
namespace facebook::velox::cudf_velox::gpu_portable {
inline std::string <WRAPPER_NAME>_source(<BAKED_PARAMS>);
}
```

Calling the getter returns the full CUDA source for a self-contained
`__device__` function with signature
`void <WRAPPER_NAME>(<OUT_PARAM>, <INPUT_PARAMS>)`. The source has no
`#include` lines, no Velox references, and no templates, so NVRTC
compiles it in cuDF's context with no additional setup.

### Where to include it

`${PROJECT_BINARY_DIR}` is on the include path of the cuDF expression
target, so the include line matches the source-tree path:

```cpp
#include "velox/experimental/cudf/gpu_portable/Round.h"
```

## End-to-end example: round

The CPU source is `velox/functions/prestosql/ArithmeticImpl.h::round`:

```cpp
template <typename TNum, typename TDecimals, bool alwaysRoundNegDec = false>
TNum round(const TNum& number, const TDecimals& decimals = 0);
```

Registration in `velox/experimental/cudf/gpu_portable/CMakeLists.txt`:

```cmake
velox_add_gpu_portable(
  OUTPUT           Round.h
  SOURCE           ${CMAKE_SOURCE_DIR}/velox/functions/prestosql/ArithmeticImpl.h
  FUNC             round
  WRAPPER_NAME     velox_round_double
  TEMPLATE_TYPES   TNum=double TDecimals=int
  TEMPLATE_VALUES  alwaysRoundNegDec=false
  OUT_PARAM        "double* out"
  INPUT_PARAMS     "double number"
  BAKED_PARAMS     "int decimals")
```

The build produces `Round.h` containing:

```cpp
namespace facebook::velox::cudf_velox::gpu_portable {
inline std::string velox_round_double_source(int decimals);
}
```

Calling `velox_round_double_source(2)` returns a CUDA source string
defining `__device__ void velox_round_double(double* out, double number)`
in which every reference to `decimals` resolves to the literal `2`.
NVRTC can constant-fold the decimal-handling branches on that basis.

The caller in `velox/experimental/cudf/expression/ExpressionEvaluator.cpp`
feeds the string to `cudf::transform_extended`:

```cpp
#include "velox/experimental/cudf/gpu_portable/Round.h"

if (inputCol.type().id() == cudf::type_id::FLOAT64) {
  const cudf::transform_input transformInputs[] = {inputCol};
  return cudf::transform_extended(
      transformInputs,
      gpu_portable::velox_round_double_source(scale_),
      cudf::data_type{cudf::type_id::FLOAT64},
      cudf::udf_source_type::CUDA,
      std::nullopt,
      cudf::null_aware::NO,
      std::nullopt,
      cudf::output_nullability::PRESERVE,
      stream, mr);
}
// For other input types, fall through to a different cuDF path.
```

The caller owns how the source string is wired into surrounding
dispatch: deciding which input types use the JIT path, pulling baked
values out of the Velox `Expr`, registering a SQL function if that is
the intent, and so on.

## Guarantees

- The emitted `__device__` function has the same semantics as the bound
  CPU instantiation, bit-identical to the CPU result where NVRTC's
  libdevice matches host libm and within a few ULPs elsewhere.
- The emitted CUDA source stands on its own and compiles inside cuDF's
  `transform_extended` context without additional headers or macros.
- The codegen is fully build-time. Any change to `SOURCE`, the
  extractor, or the registration triggers a regeneration on the next
  build. The generated header is never checked into the tree.
- Each failure point (venv provisioning, pip install, extractor run, C++
  compile of the generated header) fails loudly with the stage name and
  the path of the offending file.

## Debugging

Two files are worth looking at when a registration misbehaves:

- `${CMAKE_BINARY_DIR}/.../gpu_portable/<WRAPPER_NAME>.cu` is the
  intermediate CUDA source produced by the first stage. Transform bugs
  usually show up as a syntactic oddity here.
- `${CMAKE_BINARY_DIR}/.../gpu_portable/<OUTPUT>` is the final header
  with the source baked into the string getter.

To force regeneration, delete either file and rebuild. To reprovision
the Python environment (e.g. after editing `requirements.txt`), delete
`<gpu_portable build dir>/venv/.installed.stamp` and reconfigure.

## How it works

The generator runs in two stages, both driven by `extractor.py`:

1. Parse the CPU function's source with tree-sitter (a syntax parser
   for C++), apply a small set of tree-editing transforms to adapt the
   function for the CUDA runtime compiler, and write the resulting
   self-contained `__device__` function to
   `${CMAKE_BINARY_DIR}/.../<WRAPPER_NAME>.cu`.
2. Wrap that file in a string getter and emit the final header at
   `${CMAKE_BINARY_DIR}/.../<OUTPUT>`.

Both stages run inside a Python virtual environment that CMake
provisions once at configure time from `requirements.txt`. The venv
path is `${CMAKE_BINARY_DIR}/velox/experimental/cudf/gpu_portable/venv`.

### The transforms

NVRTC (the CUDA runtime compiler that cuDF invokes) accepts a subset of
what a host C++ compiler accepts. The first stage applies one transform
per incompatibility between that subset and the Velox CPU source. Each
transform edits only the nodes the incompatibility forces it to touch;
everything else is preserved verbatim, so the emitted `__device__`
function's semantics match the chosen CPU instantiation.

- **substitute_types** replaces each template type identifier in the
  body with the concrete type bound in `TEMPLATE_TYPES`. Needed because
  `transform_extended` compiles a concrete function, not a template.
- **substitute_values** does the same for template non-type parameters
  bound in `TEMPLATE_VALUES`. Same reason.
- **rewrite_returns** converts every `return expr;` in the body to
  `*out = expr; return;`, where `out` is the name taken from
  `OUT_PARAM`. Needed because `transform_extended` writes results
  through an out-pointer rather than via a return value.
- **strip_std_cmath** drops the `std::` qualifier from calls to cmath
  functions in a fixed whitelist (`round`, `pow`, `trunc`, `isfinite`,
  `fabs`, and others). Needed because NVRTC exposes these in the global
  namespace via CUDA's math intrinsics, but not reliably under `std::`.
- **pow_int_promotion** promotes integer literals passed as the first
  argument of `pow` to doubles, e.g. `pow(10, d)` becomes
  `pow(10.0, d)`. Needed because `pow(int, int)` is ambiguous under
  NVRTC (multiple overloads match).
- **Bridge lines for baked params** are synthesized at the top of the
  body, one per `BAKED_PARAMS` entry, of the form
  `const TYPE NAME = <literal>;`. This binds the baked literal, spliced
  at call time by the second stage, to the name the CPU body already
  uses.

When a new incompatibility surfaces (a future NVRTC version changes
behavior, or a new CPU function uses an idiom the current transforms
don't cover), the fix is to add another narrowly scoped transform in
the same style.
