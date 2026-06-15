_find_package(${ARGS})

# Add unnamespaced target for velox
foreach(tgt gtest gtest_main gmock gmock_main)
    if (NOT TARGET ${tgt} AND TARGET GTest::${tgt})
        add_library(${tgt} INTERFACE IMPORTED)
        target_link_libraries(${tgt} INTERFACE GTest::${tgt})
    endif() 
endforeach(tgt)
