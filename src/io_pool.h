/**
 * MnemoCUDA I/O Pool — Persistent thread pool for parallel pread/memcpy.
 */

#ifndef MNEMO_IO_POOL_H
#define MNEMO_IO_POOL_H

#include <stddef.h>
#include <sys/types.h>

#define IO_POOL_MAX 16  // Maximum supported pool threads

typedef struct {
    int fd;
    void *dst;
    const void *src;
    size_t size;
    off_t offset;
    volatile int done;
    size_t bytes_read;  // actual bytes transferred (may be < size on error)
    int error;          // errno on failure, 0 on success
} IOTask;

void io_pool_init(int n_threads);  // n_threads clamped to [1, IO_POOL_MAX]
void io_pool_shutdown(void);
int  io_pool_size(void);           // current pool size (0 if not init'd)
void io_pool_submit_wait(IOTask *tasks, int n);
void io_pool_submit(IOTask *tasks, int n);
void io_pool_wait(int n);

#endif
