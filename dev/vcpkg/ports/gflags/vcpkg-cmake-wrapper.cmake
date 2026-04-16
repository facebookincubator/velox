# find_package(gflags COMPONENTS shared) with namespace is buggy
set(GFLAGS_USE_TARGET_NAMESPACE OFF)

_find_package(${ARGS})

foreach(tgt gflags gflags_shared gflags_static)
    if (NOT TARGET gflags::${tgt} AND TARGET ${tgt})
        add_library(gflags::${tgt} INTERFACE IMPORTED)
        target_link_libraries(gflags::${tgt} INTERFACE ${tgt})
    endif() 
endforeach(tgt)
