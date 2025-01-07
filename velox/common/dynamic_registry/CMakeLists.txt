velox_add_library(velox_dynamic_function_loader DynamicLibraryLoader.cpp)
velox_link_libraries(velox_dynamic_function_loader PRIVATE velox_exception)

if(${VELOX_BUILD_TESTING})
  add_subdirectory(tests)
endif()