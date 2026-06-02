/*
 * Minimal reproduction for cuDF compiled binary ops on fixed_point when one
 * operand is a column and the other is a scalar.
 *
 * Null-aware ops (NULL_MAX, NULL_MIN, NULL_EQUALS) call column_device_view::
 * is_valid() on the scalar-as-column wrapper.  That wrapper reinterprets the
 * scalar's 1-byte device bool validity as a column bitmask, so bit_is_set reads
 * 4 bytes and compute-sanitizer reports an OOB access.
 *
 * Non-null-aware ops (ADD, EQUAL) do not read scalar validity inside the kernel
 * and serve as controls.
 *
 * Build (inside the Velox+cuDF container, from an existing release build tree):
 *   cmake --build /opt/velox-build/release --target cudf_nullmax_scalar_repro -j
 *
 * Run under compute-sanitizer:
 *   compute-sanitizer --tool memcheck \
 *     /opt/velox-build/release/velox/cudf-binop-scalar-repro/cudf_nullmax_scalar_repro
 */

#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using decimal64 = numeric::decimal64;
using scale_type = numeric::scale_type;

#define CUDA_CHECK(expr)                                                       \
  do {                                                                         \
    cudaError_t const _err = (expr);                                           \
    if (_err != cudaSuccess) {                                                 \
      throw std::runtime_error(std::string("CUDA error at ") + __FILE__ + ":" + \
                               std::to_string(__LINE__) + ": " +               \
                               cudaGetErrorString(_err));                      \
    }                                                                          \
  } while (0)

rmm::cuda_stream_view stream() {
  return cudf::get_default_stream();
}

rmm::device_async_resource_ref mr() {
  return cudf::get_current_device_resource_ref();
}

std::unique_ptr<cudf::column> make_decimal64_column(
    std::vector<int64_t> const& unscaled,
    scale_type scale) {
  cudf::data_type type{cudf::type_id::DECIMAL64, scale};
  auto col = cudf::make_fixed_point_column(
      type,
      static_cast<cudf::size_type>(unscaled.size()),
      cudf::mask_state::ALL_VALID,
      stream(),
      mr());
  CUDA_CHECK(cudaMemcpyAsync(
      col->mutable_view().data<int64_t>(),
      unscaled.data(),
      unscaled.size() * sizeof(int64_t),
      cudaMemcpyHostToDevice,
      stream().value()));
  stream().synchronize();
  return col;
}

void run_case(char const* name, cudf::binary_operator op, bool expect_sanitizer_issue) {
  std::cout << "\n=== " << name << " ===\n";
  if (expect_sanitizer_issue) {
    std::cout << "Expected: compute-sanitizer OOB read when is_valid() touches "
                 "scalar bool validity as bitmask\n";
  } else {
    std::cout << "Expected: clean under compute-sanitizer (control)\n";
  }

  // DECIMAL(10, 2): unscaled values 1.00, 5.00, 3.00
  auto col = make_decimal64_column({100, 500, 300}, scale_type{-2});
  auto scalar =
      cudf::make_fixed_point_scalar<decimal64>(500, scale_type{-2}, stream(), mr());

  cudf::data_type out_type;
  if (op == cudf::binary_operator::EQUAL ||
      op == cudf::binary_operator::NULL_EQUALS ||
      op == cudf::binary_operator::NULL_NOT_EQUALS) {
    out_type = cudf::data_type{cudf::type_id::BOOL8};
  } else {
    out_type = cudf::binary_operation_fixed_point_output_type(
        op, col->view().type(), scalar->type());
  }

  auto result =
      cudf::binary_operation(col->view(), *scalar, op, out_type, stream(), mr());
  stream().synchronize();
  CUDA_CHECK(cudaGetLastError());

  std::cout << "Host-visible completion: " << result->size() << " rows, type "
            << static_cast<int>(result->type().id()) << "\n";
}

void print_usage(char const* argv0) {
  std::cerr
      << "Usage: " << argv0 << " [--only null_max|null_min|null_equals|equal|add]\n"
      << "\n"
      << "Exercises cuDF column-vs-scalar fixed_point binary_operation paths.\n"
      << "Run the full binary under compute-sanitizer memcheck to see OOB reads\n"
      << "on null-aware ops.\n";
}

}  // namespace

int main(int argc, char** argv) {
  std::string only;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      return 0;
    }
    if (arg == "--only" && i + 1 < argc) {
      only = argv[++i];
      continue;
    }
    std::cerr << "Unknown argument: " << arg << "\n";
    print_usage(argv[0]);
    return 1;
  }

  try {
    std::cout << "cudf null-aware column-vs-scalar fixed_point repro\n";
    std::cout << "CUDA device: ";
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    std::cout << prop.name << " (device " << device << ")\n";

    auto run_if = [&](char const* tag, auto&& fn) {
      if (only.empty() || only == tag) {
        fn();
      }
    };

    run_if("null_max", [&] {
      run_case(
          "NULL_MAX  column vs scalar (greatest-like)",
          cudf::binary_operator::NULL_MAX,
          true);
    });
    run_if("null_min", [&] {
      run_case(
          "NULL_MIN  column vs scalar (least-like)",
          cudf::binary_operator::NULL_MIN,
          true);
    });
    run_if("null_equals", [&] {
      run_case(
          "NULL_EQUALS column vs scalar",
          cudf::binary_operator::NULL_EQUALS,
          true);
    });
    run_if("equal", [&] {
      run_case(
          "EQUAL column vs scalar (divide zero-check path; control)",
          cudf::binary_operator::EQUAL,
          false);
    });
    run_if("add", [&] {
      run_case(
          "ADD column vs scalar (control)",
          cudf::binary_operator::ADD,
          false);
    });

    std::cout << "\nAll selected cases finished on the host.\n";
    std::cout << "Re-run under: compute-sanitizer --tool memcheck " << argv[0]
              << "\n";
    return 0;
  } catch (std::exception const& ex) {
    std::cerr << "Fatal: " << ex.what() << "\n";
    return 1;
  }
}
