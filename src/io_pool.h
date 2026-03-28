/**
 * MnemoCUDA I/O Pool — Persistent thread pool for parallel pread/memcpy.
 */

#ifndef MNEMO_IO_POOL_H
#define MNEMO_IO_POOL_H

#include <stddef.h>
#include <sys/types.h>

#define IO_POOL_SIZE 8

typedef struct {
    int fd;
    void *dst;
    const void *src;
    size_t size;
    off_t offset;
    volatile int done;
} IOTask;

void io_pool_init(void);
void io_pool_shutdown(void);
void io_pool_submit_wait(IOTask *tasks, int n);
void io_pool_submit(IOTask *tasks, int n);
void io_pool_wait(int n);

#endif
