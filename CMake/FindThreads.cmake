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

# Velox WASM/Emscripten builds compile without -pthread and run single-threaded
# in the browser. Bundled dependencies (notably Boost.Atomic, transitively
# pulled by Boost.Filesystem) unconditionally call find_package(Threads
# REQUIRED), which invokes CMake's built-in FindThreads module and fails
# because emcc rejects -pthread without -sUSE_PTHREADS=1. Pre-creating a
# Threads::Threads target in the parent project does not help: FindThreads
# re-runs the probe regardless. Shim the module: under Emscripten, return a
# stub Threads::Threads target; otherwise delegate to the upstream module.
if(EMSCRIPTEN)
  if(NOT TARGET Threads::Threads)
    add_library(Threads::Threads INTERFACE IMPORTED)
  endif()
  set(Threads_FOUND TRUE)
  set(CMAKE_THREAD_LIBS_INIT "")
  set(CMAKE_HAVE_THREADS_LIBRARY 1)
  set(CMAKE_USE_WIN32_THREADS_INIT 0)
  set(CMAKE_USE_PTHREADS_INIT 1)
  set(THREADS_PREFER_PTHREAD_FLAG ON)
  return()
endif()

include(${CMAKE_ROOT}/Modules/FindThreads.cmake)
