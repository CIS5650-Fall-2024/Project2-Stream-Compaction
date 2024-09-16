set(headers
    "common.h"
    "cpu.h"
    "naive.h"
    "efficient.h"
    "efficient-optimized.h"
    "thrust.h"
    )

set(sources
    "common.cu"
    "cpu.cu"
    "naive.cu"
    "efficient.cu"
    "efficient-optimized.cu"
    "thrust.cu"
    )

list(SORT headers)
list(SORT sources)

source_group(Headers FILES ${headers})
source_group(Sources FILES ${sources})

add_library(stream_compaction ${sources} ${headers})
