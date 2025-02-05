***********************************
Dynamic Loading of Velox Extensions
***********************************

This generic utility adds extensibility features to load User Defined Functions (UDFs) without having to fork and build Velox, through the use of shared libraries. Support for connectors and types will be added in the future.

This mechanism relies on ABI compatibility, meaning it is only guarenteed to work if the shared libraries and the Velox build are based on the same Velox version.
Using shared libraries built against a different version of Velox may result in undefined behavior or runtime errors due to ABI mismatches.

Getting Started
===============

1. **Create a C++ file for your dynamic library**

   For dynamically loaded function registration, the format followed mirrors that of built-in function registration with some noted differences. Using `DynamicTestFunction.cpp` as an example, the function uses the `extern "C"` keyword to protect against name mangling. A `registry()` function call is also necessary.

   Make sure to also include the necessary header file:

   .. code-block:: cpp

      #include "velox/common/dynamic_registry/DynamicUdf.h"

   Example template for a function with no arguments returning a BIGINT:

   .. code-block:: cpp

      #include "velox/common/dynamic_registry/DynamicUdf.h"

      namespace example_namespace {

      template <typename T>
      struct DynamicFunction {
        FOLLY_ALWAYS_INLINE bool call(int64_t& result) {
          // Code to calculate result.
        }
      };
      }

      extern "C" {
      void registry() {
        facebook::velox::registerFunction<
            example_namespace::DynamicFunction,
            int64_t>({"your_function_name"});
      }
      }

2. **Register functions dynamically by creating `.dylib` (MacOS) or `.so` (Linux) shared libraries**

   These shared libraries may be made using a CMakeLists file like the following:

   .. code-block:: cmake

      add_library(name_of_dynamic_fn SHARED TestFunction.cpp)
      target_link_libraries(name_of_dynamic_fn PRIVATE fmt::fmt glog::glog xsimd)
      target_link_options(name_of_dynamic_fn PRIVATE "-Wl,-undefined,dynamic_lookup")

   - The `fmt::fmt` and `xsimd` libraries are required for all necessary symbols to be defined when loading `TestFunction.cpp` dynamically.
   - Additionally, `glog::glog` is currently required on MacOS.
   - The `target_link_options` allows for symbols to be resolved at runtime on MacOS.
   - On Linux, `glog::glog` and the `target_link_options` are optional.

Notes
=====

- In Velox, a function's signature is determined solely by its name and argument types. The return type is not taken into account. As a result, if a function with an identical signature is added but with a different return type, it will overwrite the existing function.
- Function overloading is supported. Therefore, multiple functions can share the same name as long as they differ in the number or types of arguments.

