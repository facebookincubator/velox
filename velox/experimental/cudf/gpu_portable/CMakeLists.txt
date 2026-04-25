# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Generates self-contained __device__ headers by lifting simple Velox CPU
# functions via a tree-sitter-based extractor. Each registration produces a
# header in ${CMAKE_CURRENT_BINARY_DIR}/; consumers depend on the
# velox_cudf_gpu_portable_gen aggregate target and include the headers
# through ${CMAKE_BINARY_DIR}.

find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(GPU_PORTABLE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(GPU_PORTABLE_BIN ${CMAKE_CURRENT_BINARY_DIR})
set(GPU_PORTABLE_VENV ${GPU_PORTABLE_BIN}/venv)
set(GPU_PORTABLE_PY ${GPU_PORTABLE_VENV}/bin/python3)
set(GPU_PORTABLE_REQS ${GPU_PORTABLE_DIR}/requirements.txt)
set(GPU_PORTABLE_EXTR ${GPU_PORTABLE_DIR}/extractor.py)
set(GPU_PORTABLE_STAMP ${GPU_PORTABLE_VENV}/.installed.stamp)

# Provision the venv and install pinned deps. Reruns when requirements.txt
# is newer than the stamp so version bumps propagate on next configure.
if(NOT EXISTS ${GPU_PORTABLE_STAMP} OR ${GPU_PORTABLE_REQS} IS_NEWER_THAN ${GPU_PORTABLE_STAMP})
  message(
    STATUS
    "gpu_portable: provisioning venv at ${GPU_PORTABLE_VENV} "
    "using ${Python3_EXECUTABLE}"
  )
  file(REMOVE_RECURSE ${GPU_PORTABLE_VENV})
  execute_process(
    COMMAND ${Python3_EXECUTABLE} -m venv ${GPU_PORTABLE_VENV}
    RESULT_VARIABLE _rc
    ERROR_VARIABLE _err
  )
  if(NOT _rc EQUAL 0)
    message(
      FATAL_ERROR
      "gpu_portable: failed to create venv at ${GPU_PORTABLE_VENV}\n"
      "  interpreter: ${Python3_EXECUTABLE}\n"
      "  exit code: ${_rc}\n"
      "  stderr:\n${_err}\n"
      "Check that Python3 venv support is installed "
      "(e.g. python3-venv on Debian/Ubuntu)."
    )
  endif()

  message(STATUS "gpu_portable: installing deps from ${GPU_PORTABLE_REQS}")
  execute_process(
    COMMAND
      ${GPU_PORTABLE_PY} -m pip install --disable-pip-version-check --require-virtualenv -r
      ${GPU_PORTABLE_REQS}
    RESULT_VARIABLE _rc
    OUTPUT_VARIABLE _out
    ERROR_VARIABLE _err
  )
  if(NOT _rc EQUAL 0)
    message(
      FATAL_ERROR
      "gpu_portable: pip install failed\n"
      "  requirements: ${GPU_PORTABLE_REQS}\n"
      "  venv:         ${GPU_PORTABLE_VENV}\n"
      "  exit code:    ${_rc}\n"
      "  stdout:\n${_out}\n"
      "  stderr:\n${_err}\n"
      "This step needs network access at configure time. To retry "
      "by hand:\n"
      "  ${GPU_PORTABLE_PY} -m pip install -r ${GPU_PORTABLE_REQS}"
    )
  endif()

  # Verify imports cleanly so downstream build steps don't fail cryptically.
  execute_process(
    COMMAND ${GPU_PORTABLE_PY} -c "import tree_sitter, tree_sitter_cpp"
    RESULT_VARIABLE _rc
    ERROR_VARIABLE _err
  )
  if(NOT _rc EQUAL 0)
    message(
      FATAL_ERROR
      "gpu_portable: tree-sitter deps installed but not importable\n"
      "  venv:   ${GPU_PORTABLE_VENV}\n"
      "  stderr:\n${_err}"
    )
  endif()

  file(TOUCH ${GPU_PORTABLE_STAMP})
  message(STATUS "gpu_portable: venv ready")
endif()

# Registers one generated header. Each call creates a rule that produces
# <OUTPUT> in the build tree by running the extractor's wrap + embed stages.
#
# Named args:
#   OUTPUT           header file name (relative to this dir's build dir).
#   SOURCE           absolute path to the Velox CPU source file.
#   FUNC             C++ function name inside SOURCE to lift.
#   WRAPPER_NAME     name of the emitted __device__ function; the getter
#                    that returns its source string is named
#                    <WRAPPER_NAME>_source.
#   OUT_PARAM        the __device__ wrapper's out-pointer parameter,
#                    e.g. "double* out". The CPU return value is written
#                    to this pointer.
#   INPUT_PARAMS     (list) per-row inputs of the __device__ wrapper,
#                    e.g. "double number".
#   BAKED_PARAMS     (list) values known at call time, spliced into the
#                    source as literals so NVRTC can constant-fold. Each
#                    entry "TYPE NAME" adds a TYPE NAME parameter to the
#                    emitted getter. The name should match the name of
#                    the corresponding CPU function parameter.
#   TEMPLATE_TYPES   (list) picks the concrete type for each template
#                    type parameter of the CPU function, e.g.
#                    "TNum=double".
#   TEMPLATE_VALUES  (list) picks the concrete value for each template
#                    non-type parameter, e.g. "alwaysRoundNegDec=false".
function(velox_add_gpu_portable)
  # FRAGMENT_GROUP: when set, the call emits a getter-only fragment (no
  # boilerplate) instead of a standalone header. The fragment path is
  # accumulated under the named group's global property; a later call to
  # velox_combine_gpu_portable_fragments(GROUP <same name> OUTPUT <h>)
  # collects all the group's fragments into one combined header. Used to
  # land template-unrolled instantiations of the same CPU function in a
  # single header (the general pattern: "1 registration -> 1 header"
  # holds, except for template unrolling where it's "N variants -> 1
  # combined header").
  set(
    one_value
    OUTPUT
    SOURCE
    FUNC
    WRAPPER_NAME
    OUT_PARAM
    OUT_ASSIGN_NAME
    OUT_WRAP
    FRAGMENT_GROUP
  )
  set(
    multi_value
    TEMPLATE_TYPES
    TEMPLATE_VALUES
    INPUT_PARAMS
    BAKED_PARAMS
    REPLACE_EXPRS
    INPUT_ADAPTS
  )
  cmake_parse_arguments(ARG "" "${one_value}" "${multi_value}" ${ARGN})

  foreach(
    required
    OUTPUT
    SOURCE
    FUNC
    WRAPPER_NAME
    OUT_PARAM
  )
    if(NOT ARG_${required})
      message(FATAL_ERROR "velox_add_gpu_portable: missing required argument ${required}")
    endif()
  endforeach()
  if(ARG_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR "velox_add_gpu_portable: unknown arguments: ${ARG_UNPARSED_ARGUMENTS}")
  endif()
  if(NOT EXISTS ${ARG_SOURCE})
    message(FATAL_ERROR "velox_add_gpu_portable(${ARG_OUTPUT}): SOURCE not found: ${ARG_SOURCE}")
  endif()

  set(wrapped ${GPU_PORTABLE_BIN}/${ARG_WRAPPER_NAME}.cu)
  if(ARG_FRAGMENT_GROUP)
    set(header ${GPU_PORTABLE_BIN}/${ARG_OUTPUT}.frag)
  else()
    set(header ${GPU_PORTABLE_BIN}/${ARG_OUTPUT})
  endif()

  set(
    wrap_args
    --input
    "${ARG_SOURCE}"
    --func
    "${ARG_FUNC}"
    --wrapper-name
    "${ARG_WRAPPER_NAME}"
    --out-param
    "${ARG_OUT_PARAM}"
    --output
    "${wrapped}"
  )
  foreach(p IN LISTS ARG_TEMPLATE_TYPES)
    list(APPEND wrap_args --template-type "${p}")
  endforeach()
  foreach(p IN LISTS ARG_TEMPLATE_VALUES)
    list(APPEND wrap_args --template-value "${p}")
  endforeach()
  foreach(p IN LISTS ARG_INPUT_PARAMS)
    list(APPEND wrap_args --input-param "${p}")
  endforeach()
  foreach(p IN LISTS ARG_BAKED_PARAMS)
    list(APPEND wrap_args --baked-param "${p}")
  endforeach()
  if(ARG_OUT_ASSIGN_NAME)
    list(APPEND wrap_args --out-assign-name "${ARG_OUT_ASSIGN_NAME}")
  endif()
  if(ARG_OUT_WRAP)
    list(APPEND wrap_args --out-wrap "${ARG_OUT_WRAP}")
  endif()
  foreach(p IN LISTS ARG_REPLACE_EXPRS)
    list(APPEND wrap_args --replace-expr "${p}")
  endforeach()
  foreach(p IN LISTS ARG_INPUT_ADAPTS)
    list(APPEND wrap_args --input-adapt "${p}")
  endforeach()

  set(
    embed_args
    --source
    "${wrapped}"
    --wrapper-name
    "${ARG_WRAPPER_NAME}"
    --output
    "${header}"
  )
  foreach(p IN LISTS ARG_BAKED_PARAMS)
    list(APPEND embed_args --baked-param "${p}")
  endforeach()
  if(ARG_FRAGMENT_GROUP)
    list(APPEND embed_args --fragment)
  endif()

  add_custom_command(
    OUTPUT ${header}
    BYPRODUCTS ${wrapped}
    COMMAND ${GPU_PORTABLE_PY} ${GPU_PORTABLE_EXTR} wrap ${wrap_args}
    COMMAND ${GPU_PORTABLE_PY} ${GPU_PORTABLE_EXTR} embed ${embed_args}
    DEPENDS ${ARG_SOURCE} ${GPU_PORTABLE_EXTR} ${GPU_PORTABLE_STAMP}
    COMMENT "[gpu_portable] ${ARG_FUNC} -> ${ARG_OUTPUT}"
    VERBATIM
  )

  if(ARG_FRAGMENT_GROUP)
    set_property(GLOBAL APPEND
      PROPERTY VELOX_CUDF_GPU_PORTABLE_FRAGMENTS_${ARG_FRAGMENT_GROUP}
      ${header})
  else()
    set_property(GLOBAL APPEND PROPERTY VELOX_CUDF_GPU_PORTABLE_HEADERS ${header})
  endif()
endfunction()

# Combines all fragments accumulated under FRAGMENT_GROUP into a single
# header. Run once per group, after all per-variant velox_add_gpu_portable
# calls have registered their fragments.
function(velox_combine_gpu_portable_fragments)
  set(one_value GROUP OUTPUT SOURCE_PATH)
  cmake_parse_arguments(ARG "" "${one_value}" "" ${ARGN})
  foreach(required GROUP OUTPUT)
    if(NOT ARG_${required})
      message(FATAL_ERROR
        "velox_combine_gpu_portable_fragments: missing ${required}")
    endif()
  endforeach()

  get_property(fragments GLOBAL
    PROPERTY VELOX_CUDF_GPU_PORTABLE_FRAGMENTS_${ARG_GROUP})
  if(NOT fragments)
    message(FATAL_ERROR
      "velox_combine_gpu_portable_fragments: group '${ARG_GROUP}' has no fragments")
  endif()

  set(combined ${GPU_PORTABLE_BIN}/${ARG_OUTPUT})
  set(combine_args --output "${combined}")
  if(ARG_SOURCE_PATH)
    list(APPEND combine_args --source-path "${ARG_SOURCE_PATH}")
  endif()
  foreach(f IN LISTS fragments)
    list(APPEND combine_args --fragment "${f}")
  endforeach()

  add_custom_command(
    OUTPUT ${combined}
    COMMAND ${GPU_PORTABLE_PY} ${GPU_PORTABLE_EXTR} combine ${combine_args}
    DEPENDS ${fragments} ${GPU_PORTABLE_EXTR} ${GPU_PORTABLE_STAMP}
    COMMENT "[gpu_portable] combine ${ARG_GROUP} -> ${ARG_OUTPUT}"
    VERBATIM
  )

  set_property(GLOBAL APPEND PROPERTY
    VELOX_CUDF_GPU_PORTABLE_HEADERS ${combined})
endfunction()

# --- Registrations ---

velox_add_gpu_portable(
  OUTPUT           Round.h
  SOURCE           ${CMAKE_SOURCE_DIR}/velox/functions/prestosql/ArithmeticImpl.h
  FUNC             round
  WRAPPER_NAME     velox_round_double
  TEMPLATE_TYPES   TNum=double TDecimals=int
  TEMPLATE_VALUES  alwaysRoundNegDec=false
  OUT_PARAM        "double* out"
  INPUT_PARAMS     "double number"
  BAKED_PARAMS     "int decimals"
)

# Lift DecimalUtil::divideWithRoundUp. C++ templates don't survive into
# the JIT'd source: each (R, A, B) instantiation needs its own concrete
# emission, even though the body is identical across them. This is a
# general pattern -- whenever you lift a templated CPU function and need
# more than one instantiation on the GPU, expect to wrap the per-variant
# velox_add_gpu_portable() call in a small helper function like the one
# below to fold the per-variant boilerplate. The variants get accumulated
# under FRAGMENT_GROUP and stitched into one combined header by the
# trailing velox_combine_gpu_portable_fragments() call -- so the final
# output stays as one DecimalDivide.h despite N instantiations.
#
# Semantic deltas vs the CPU source that the transforms bake in:
#  - overflow wraps silently (CPU throws via checkedMultiply),
#  - division by zero is undefined (CPU throws VELOX_USER_CHECK_NE; the
#    caller must ensure b != 0).
function(velox_add_decimal_divide variant in_int out_int in_dec out_dec)
  velox_add_gpu_portable(
    OUTPUT           DecimalDivide_${variant}.h
    FRAGMENT_GROUP   decimal_divide
    SOURCE           ${CMAKE_SOURCE_DIR}/velox/type/DecimalUtil.h
    FUNC             divideWithRoundUp
    WRAPPER_NAME     velox_decimal_divide_${variant}
    TEMPLATE_TYPES   R=${out_int} A=${in_int} B=${in_int}
    OUT_PARAM        "numeric::${out_dec}* out"
    INPUT_PARAMS     "numeric::${in_dec} a_in" "numeric::${in_dec} b_in"
    INPUT_ADAPTS     "${in_int} a=a_in.value()" "${in_int} b=b_in.value()"
    OUT_WRAP         "numeric::${out_dec}{VALUE, numeric::scale_type{0}}"
    BAKED_PARAMS     "bool noRoundUp" "__int128_t pow10_aRescale"
    OUT_ASSIGN_NAME  r
    REPLACE_EXPRS    "DecimalUtil::kPowersOfTen[aRescale]=pow10_aRescale"
  )
endfunction()

velox_add_decimal_divide(int64_int64   int64_t    int64_t    decimal64  decimal64)
velox_add_decimal_divide(int64_int128  int64_t    __int128_t decimal64  decimal128)
velox_add_decimal_divide(int128_int128 __int128_t __int128_t decimal128 decimal128)

velox_combine_gpu_portable_fragments(
  GROUP        decimal_divide
  OUTPUT       DecimalDivide.h
  SOURCE_PATH  velox/type/DecimalUtil.h::divideWithRoundUp
)

# --- Aggregate codegen target ---

get_property(_gen_headers GLOBAL PROPERTY VELOX_CUDF_GPU_PORTABLE_HEADERS)
add_custom_target(velox_cudf_gpu_portable_gen DEPENDS ${_gen_headers})
