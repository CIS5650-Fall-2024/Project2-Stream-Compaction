set(headers
    "common.h"
    "cpu.h"
    "naive.h"
    "efficient.h"
    "efficient-thread-optimized.h"
    "thrust.h"
    "radix-sort.h"
    )

set(sources
    "common.cu"
    "cpu.cu"
    "naive.cu"
    "efficient.cu"
    "efficient-thread-optimized.cu"
    "thrust.cu"
    "radix-sort.cu"
    )

list(SORT headers)
list(SORT sources)

source_group(Headers FILES ${headers})
source_group(Sources FILES ${sources})

add_library(stream_compaction ${sources} ${headers})
