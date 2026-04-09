# Insert #include <cstdint> into DuckDB FSST before <algorithm> if missing.
# Robust across tarball line endings and FetchContent patch CWD quirks.
if(NOT DEFINED VELOX_DUCKDB_SRC)
  message(FATAL_ERROR "VELOX_DUCKDB_SRC not set")
endif()
set(_fsst "${VELOX_DUCKDB_SRC}/third_party/fsst/libfsst.hpp")
if(NOT EXISTS "${_fsst}")
  message(FATAL_ERROR "DuckDB FSST header not found: ${_fsst}")
endif()
file(READ "${_fsst}" _hpp)
string(FIND "${_hpp}" "#include <cstdint>" _cstdint_pos)
if(NOT _cstdint_pos EQUAL -1)
  return()
endif()
string(REPLACE "#include <algorithm>"
       "#include <cstdint>\n#include <algorithm>" _hpp2 "${_hpp}")
if(_hpp2 STREQUAL _hpp)
  message(
    FATAL_ERROR
    "Could not patch ${_fsst}: expected literal line '#include <algorithm>'")
endif()
file(WRITE "${_fsst}" "${_hpp2}")
