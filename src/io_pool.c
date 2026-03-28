/**
 * MnemoCUDA I/O Pool — Persistent thread pool for parallel pread/memcpy.
 *
 * Avoids per-token pthread_create/join overhead by keeping threads alive.
 */

#include "io_pool.h"

#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <semaphore.h>
#include <stdint.h>

typedef struct {
    pthread_t threads[IO_POOL_MAX];
    IOTask tasks[IO_POOL_MAX];
    sem_t task_ready[IO_POOL_MAX];
    sem_t task_done[IO_POOL_MAX];
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
        if (t->fd >= 0) {
            pread(t->fd, t->dst, t->size, t->offset);
        } else if (t->src) {
            memcpy(t->dst, t->src, t->size);
        }
        t->done = 1;
        sem_post(&pool->task_done[id]);
    }
    return NULL;
}

void io_pool_init(int n_threads) {
    if (g_io_pool_init) return;
    if (n_threads < 1) n_threads = 1;
    if (n_threads > IO_POOL_MAX) n_threads = IO_POOL_MAX;

    memset(&g_io_pool, 0, sizeof(g_io_pool));
    g_io_pool.size = n_threads;
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
