# Apply DuckDB patches on Windows without git apply.
# This script is run as PATCH_COMMAND via cmake -P.
# Working directory is the DuckDB source root.
# Patches are idempotent (safe to run on already-patched source).

# Patch 1: remove-ccache.patch - Remove ccache/sccache from CMakeLists.txt
file(READ "CMakeLists.txt" content)
string(FIND "${content}" "find_program(CCACHE_PROGRAM ccache)" _idx)
if(NOT _idx EQUAL -1)
  string(REPLACE
    "find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
  set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE \"\${CCACHE_PROGRAM}\")
else()
  find_program(CCACHE_PROGRAM sccache)
  if(CCACHE_PROGRAM)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE \"\${CCACHE_PROGRAM}\")
  endif()
endif()

"
    ""
    content "${content}")
  file(WRITE "CMakeLists.txt" "${content}")
  message(STATUS "Applied remove-ccache patch")
else()
  message(STATUS "remove-ccache patch already applied")
endif()

# Patch 2: re2.patch - Change PUBLIC to PRIVATE for duckdb_re2 include
file(READ "third_party/re2/CMakeLists.txt" content)
string(FIND "${content}" "PUBLIC $<BUILD_INTERFACE:\${CMAKE_CURRENT_SOURCE_DIR}>" _idx)
if(NOT _idx EQUAL -1)
  string(REPLACE
    "PUBLIC $<BUILD_INTERFACE:\${CMAKE_CURRENT_SOURCE_DIR}>"
    "PRIVATE $<BUILD_INTERFACE:\${CMAKE_CURRENT_SOURCE_DIR}>"
    content "${content}")
  file(WRITE "third_party/re2/CMakeLists.txt" "${content}")
  message(STATUS "Applied re2 patch")
else()
  message(STATUS "re2 patch already applied")
endif()

# Patch 3 (MSVC only): Fix DUCKDB_API for static library builds on Windows.
# Without DUCKDB_BUILD_LIBRARY defined, DUCKDB_API becomes __declspec(dllimport),
# causing C2491 errors. For static libs, DUCKDB_API should be empty.
file(READ "src/include/duckdb/common/winapi.hpp" content)
string(FIND "${content}" "__declspec(dllimport)" _idx)
if(NOT _idx EQUAL -1)
  string(REPLACE "__declspec(dllimport)" "" content "${content}")
  file(WRITE "src/include/duckdb/common/winapi.hpp" "${content}")
  message(STATUS "Applied DUCKDB_API static lib patch")
else()
  message(STATUS "DUCKDB_API patch already applied")
endif()
