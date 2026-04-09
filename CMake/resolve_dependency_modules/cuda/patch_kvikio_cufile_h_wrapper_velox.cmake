# Idempotent: inject velox_cufile_batch_stream_compat.h after <cufile.h> in KvikIO.
if(NOT DEFINED KVIKIO_CUFILE_H_WRAPPER)
  message(FATAL_ERROR "KVIKIO_CUFILE_H_WRAPPER not set")
endif()
if(NOT EXISTS "${KVIKIO_CUFILE_H_WRAPPER}")
  message(FATAL_ERROR "KvikIO cufile_h_wrapper.hpp not found: ${KVIKIO_CUFILE_H_WRAPPER}")
endif()
file(READ "${KVIKIO_CUFILE_H_WRAPPER}" _velox_kv_w)
if(_velox_kv_w MATCHES "VELOX_CUFILE_BATCH_STREAM_COMPAT")
  return()
endif()
set(
  _velox_marker
  [[
/* VELOX_CUFILE_BATCH_STREAM_COMPAT */
#include "velox_cufile_batch_stream_compat.h"
]]
)
set(_velox_kv_before "${_velox_kv_w}")
string(REPLACE
       "#include <cufile.h>\n#else"
       "#include <cufile.h>\n${_velox_marker}#else"
       _velox_kv_w
       "${_velox_kv_w}")
if(_velox_kv_w STREQUAL "${_velox_kv_before}")
  string(REPLACE
         "#include <cufile.h>\r\n#else"
         "#include <cufile.h>\r\n${_velox_marker}#else"
         _velox_kv_w
         "${_velox_kv_w}")
endif()
if(_velox_kv_w STREQUAL "${_velox_kv_before}")
  message(
    FATAL_ERROR
    "Velox: KvikIO cufile_h_wrapper.hpp patch failed: expected \"#include <cufile.h>\" then \"#else\"."
  )
endif()
file(WRITE "${KVIKIO_CUFILE_H_WRAPPER}" "${_velox_kv_w}")
