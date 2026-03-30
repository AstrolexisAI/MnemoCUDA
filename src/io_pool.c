/**
 * MnemoCUDA I/O Pool — Persistent thread pool for parallel pread/memcpy.
 *
 * Avoids per-token pthread_create/join overhead by keeping threads alive.
 */

#include "io_pool.h"

#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <pthread.h>
#include <semaphore.h>
#include <stdint.h>

typedef struct {
    pthread_t threads[IO_POOL_MAX];
    IOTask tasks[IO_POOL_MAX];
    sem_t task_ready[IO_POOL_MAX];
    sem_t task_done[IO_POOL_MAX];
    sem_t any_done;           // signaled when any task completes
    volatile int shutdown;
    int size;  // actual number of threads
} IOPool;

static IOPool g_io_pool;
static int g_io_pool_init = 0;

static void *io_pool_worker(void *arg) {
    int id = (int)(intptr_t)arg;
    IOPool *pool = &g_io_pool;
    while (!pool->shutdown) {
        sem_wait(&pool->task_ready[id]);
        if (pool->shutdown) break;
        IOTask *t = &pool->tasks[id];
        t->bytes_read = 0;
        t->error = 0;
        if (t->fd >= 0) {
            // pread_full: retry on EINTR, accumulate until size or EOF/error
            size_t total = 0;
            while (total < t->size) {
                ssize_t n = pread(t->fd, (char *)t->dst + total,
                                  t->size - total, t->offset + (off_t)total);
                if (n < 0) {
                    if (errno == EINTR) continue;
                    t->error = errno;
                    break;
                }
                if (n == 0) break;  // EOF
                total += (size_t)n;
            }
            t->bytes_read = total;
            if (total < t->size && t->error == 0)
                t->error = EIO;  // short read
        } else if (t->src) {
            memcpy(t->dst, t->src, t->size);
            t->bytes_read = t->size;
        }
        t->done = 1;
        sem_post(&pool->task_done[id]);
        sem_post(&pool->any_done);
    }
    return NULL;
}

void io_pool_init(int n_threads) {
    if (g_io_pool_init) return;
    if (n_threads < 1) n_threads = 1;
    if (n_threads > IO_POOL_MAX) n_threads = IO_POOL_MAX;

    memset(&g_io_pool, 0, sizeof(g_io_pool));
    g_io_pool.size = n_threads;
    sem_init(&g_io_pool.any_done, 0, 0);
    for (int i = 0; i < n_threads; i++) {
        sem_init(&g_io_pool.task_ready[i], 0, 0);
        sem_init(&g_io_pool.task_done[i], 0, 0);
        pthread_create(&g_io_pool.threads[i], NULL, io_pool_worker, (void *)(intptr_t)i);
    }
    g_io_pool_init = 1;
}

void io_pool_shutdown(void) {
    if (!g_io_pool_init) return;
    g_io_pool.shutdown = 1;
    for (int i = 0; i < g_io_pool.size; i++)
        sem_post(&g_io_pool.task_ready[i]);
    for (int i = 0; i < g_io_pool.size; i++)
        pthread_join(g_io_pool.threads[i], NULL);
    for (int i = 0; i < g_io_pool.size; i++) {
        sem_destroy(&g_io_pool.task_ready[i]);
        sem_destroy(&g_io_pool.task_done[i]);
    }
    sem_destroy(&g_io_pool.any_done);
    g_io_pool_init = 0;
}

int io_pool_size(void) {
    return g_io_pool_init ? g_io_pool.size : 0;
}

void io_pool_submit_wait(IOTask *tasks, int n) {
    if (n > g_io_pool.size) n = g_io_pool.size;
    for (int i = 0; i < n; i++) {
        g_io_pool.tasks[i] = tasks[i];
        g_io_pool.tasks[i].done = 0;
        sem_post(&g_io_pool.task_ready[i]);
    }
    for (int i = 0; i < n; i++)
        sem_wait(&g_io_pool.task_done[i]);
}

void io_pool_submit(IOTask *tasks, int n) {
    if (n > g_io_pool.size) n = g_io_pool.size;
    for (int i = 0; i < n; i++) {
        g_io_pool.tasks[i] = tasks[i];
        g_io_pool.tasks[i].done = 0;
        sem_post(&g_io_pool.task_ready[i]);
    }
}

void io_pool_wait(int n) {
    if (n > g_io_pool.size) n = g_io_pool.size;
    for (int i = 0; i < n; i++)
        sem_wait(&g_io_pool.task_done[i]);
}

int io_pool_wait_any(int n) {
    if (n > g_io_pool.size) n = g_io_pool.size;
    // Wait for any task to signal completion
    sem_wait(&g_io_pool.any_done);
    // Find which task completed (scan for done=1 that hasn't been consumed yet)
    for (int i = 0; i < n; i++) {
        if (g_io_pool.tasks[i].done) {
            // Mark consumed so subsequent calls skip this task
            g_io_pool.tasks[i].done = 0;
            // Also consume the per-task semaphore so io_pool_wait still works
            sem_trywait(&g_io_pool.task_done[i]);
            return i;
        }
    }
    return -1;  // shouldn't happen
}

int io_pool_task_error(int idx) {
    if (idx < 0 || idx >= g_io_pool.size) return -1;
    return g_io_pool.tasks[idx].error;
}
