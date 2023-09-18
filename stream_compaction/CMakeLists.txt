set(headers
    "common.h"
    "cpu.h"
    "naive.h"
    "efficient.h"
    "thrust.h"
    "efficient_sharedmem.h"
    "radix_sort.h"
    )

set(sources
    "common.cu"
    "cpu.cu"
    "naive.cu"
    "efficient.cu"
    "thrust.cu"
    "efficient_sharedmem.cu"
    "radix_sort.cu"
    )

list(SORT headers)
list(SORT sources)

source_group(Headers FILES ${headers})
source_group(Sources FILES ${sources})

cuda_add_library(stream_compaction ${sources} ${headers})
