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

set(GPU_PORTABLE_DIR   ${CMAKE_CURRENT_SOURCE_DIR})
set(GPU_PORTABLE_BIN   ${CMAKE_CURRENT_BINARY_DIR})
set(GPU_PORTABLE_VENV  ${GPU_PORTABLE_BIN}/venv)
set(GPU_PORTABLE_PY    ${GPU_PORTABLE_VENV}/bin/python3)
set(GPU_PORTABLE_REQS  ${GPU_PORTABLE_DIR}/requirements.txt)
set(GPU_PORTABLE_EXTR  ${GPU_PORTABLE_DIR}/extractor.py)
set(GPU_PORTABLE_STAMP ${GPU_PORTABLE_VENV}/.installed.stamp)

# Provision the venv and install pinned deps. Reruns when requirements.txt
# is newer than the stamp so version bumps propagate on next configure.
if(NOT EXISTS ${GPU_PORTABLE_STAMP}
   OR ${GPU_PORTABLE_REQS} IS_NEWER_THAN ${GPU_PORTABLE_STAMP})
  message(STATUS
    "gpu_portable: provisioning venv at ${GPU_PORTABLE_VENV} "
    "using ${Python3_EXECUTABLE}")
  file(REMOVE_RECURSE ${GPU_PORTABLE_VENV})
  execute_process(
    COMMAND ${Python3_EXECUTABLE} -m venv ${GPU_PORTABLE_VENV}
    RESULT_VARIABLE _rc
    ERROR_VARIABLE _err)
  if(NOT _rc EQUAL 0)
    message(FATAL_ERROR
      "gpu_portable: failed to create venv at ${GPU_PORTABLE_VENV}\n"
      "  interpreter: ${Python3_EXECUTABLE}\n"
      "  exit code: ${_rc}\n"
      "  stderr:\n${_err}\n"
      "Check that Python3 venv support is installed "
      "(e.g. python3-venv on Debian/Ubuntu).")
  endif()

  message(STATUS
    "gpu_portable: installing deps from ${GPU_PORTABLE_REQS}")
  execute_process(
    COMMAND ${GPU_PORTABLE_PY} -m pip install
            --disable-pip-version-check --require-virtualenv
            -r ${GPU_PORTABLE_REQS}
    RESULT_VARIABLE _rc
    OUTPUT_VARIABLE _out
    ERROR_VARIABLE  _err)
  if(NOT _rc EQUAL 0)
    message(FATAL_ERROR
      "gpu_portable: pip install failed\n"
      "  requirements: ${GPU_PORTABLE_REQS}\n"
      "  venv:         ${GPU_PORTABLE_VENV}\n"
      "  exit code:    ${_rc}\n"
      "  stdout:\n${_out}\n"
      "  stderr:\n${_err}\n"
      "This step needs network access at configure time. To retry "
      "by hand:\n"
      "  ${GPU_PORTABLE_PY} -m pip install -r ${GPU_PORTABLE_REQS}")
  endif()

  # Verify imports cleanly so downstream build steps don't fail cryptically.
  execute_process(
    COMMAND ${GPU_PORTABLE_PY} -c
            "import tree_sitter, tree_sitter_cpp"
    RESULT_VARIABLE _rc
    ERROR_VARIABLE  _err)
  if(NOT _rc EQUAL 0)
    message(FATAL_ERROR
      "gpu_portable: tree-sitter deps installed but not importable\n"
      "  venv:   ${GPU_PORTABLE_VENV}\n"
      "  stderr:\n${_err}")
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
  set(one_value   OUTPUT SOURCE FUNC WRAPPER_NAME OUT_PARAM)
  set(multi_value TEMPLATE_TYPES TEMPLATE_VALUES INPUT_PARAMS BAKED_PARAMS)
  cmake_parse_arguments(ARG "" "${one_value}" "${multi_value}" ${ARGN})

  foreach(required OUTPUT SOURCE FUNC WRAPPER_NAME OUT_PARAM)
    if(NOT ARG_${required})
      message(FATAL_ERROR
        "velox_add_gpu_portable: missing required argument ${required}")
    endif()
  endforeach()
  if(ARG_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR
      "velox_add_gpu_portable: unknown arguments: ${ARG_UNPARSED_ARGUMENTS}")
  endif()
  if(NOT EXISTS ${ARG_SOURCE})
    message(FATAL_ERROR
      "velox_add_gpu_portable(${ARG_OUTPUT}): SOURCE not found: ${ARG_SOURCE}")
  endif()

  set(wrapped ${GPU_PORTABLE_BIN}/${ARG_WRAPPER_NAME}.cu)
  set(header  ${GPU_PORTABLE_BIN}/${ARG_OUTPUT})

  set(wrap_args
      --input        "${ARG_SOURCE}"
      --func         "${ARG_FUNC}"
      --wrapper-name "${ARG_WRAPPER_NAME}"
      --out-param    "${ARG_OUT_PARAM}"
      --output       "${wrapped}")
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

  set(embed_args
      --source       "${wrapped}"
      --wrapper-name "${ARG_WRAPPER_NAME}"
      --output       "${header}")
  foreach(p IN LISTS ARG_BAKED_PARAMS)
    list(APPEND embed_args --baked-param "${p}")
  endforeach()

  add_custom_command(
    OUTPUT  ${header}
    BYPRODUCTS ${wrapped}
    COMMAND ${GPU_PORTABLE_PY} ${GPU_PORTABLE_EXTR} wrap  ${wrap_args}
    COMMAND ${GPU_PORTABLE_PY} ${GPU_PORTABLE_EXTR} embed ${embed_args}
    DEPENDS ${ARG_SOURCE} ${GPU_PORTABLE_EXTR} ${GPU_PORTABLE_STAMP}
    COMMENT "[gpu_portable] ${ARG_FUNC} -> ${ARG_OUTPUT}"
    VERBATIM)

  set_property(GLOBAL APPEND PROPERTY
    VELOX_CUDF_GPU_PORTABLE_HEADERS ${header})
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
  BAKED_PARAMS     "int decimals")

# --- Aggregate codegen target ---

get_property(_gen_headers GLOBAL PROPERTY VELOX_CUDF_GPU_PORTABLE_HEADERS)
add_custom_target(velox_cudf_gpu_portable_gen DEPENDS ${_gen_headers})
