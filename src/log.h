/**
 * MnemoCUDA Logging — Leveled logging with timestamps.
 *
 * Usage:
 *   LOG_INFO("Loaded %d tensors", count);
 *   LOG_WARN("Token count %d exceeds limit", n);
 *   LOG_ERROR("CUDA error: %s", msg);
 */

#ifndef MNEMO_LOG_H
#define MNEMO_LOG_H

#include <stdio.h>
#include <time.h>

typedef enum {
    MNEMO_LOG_ERROR = 0,
    MNEMO_LOG_WARN  = 1,
    MNEMO_LOG_INFO  = 2,
    MNEMO_LOG_DEBUG = 3,
} MnemoLogLevel;

// Global log level (default: INFO). Set before loading to filter output.
extern MnemoLogLevel mnemo_log_level;

#define MNEMO_LOG(level, tag, fmt, ...) do { \
    if ((level) <= mnemo_log_level) { \
        struct timespec _ts; clock_gettime(CLOCK_REALTIME, &_ts); \
        struct tm _tm; localtime_r(&_ts.tv_sec, &_tm); \
        fprintf(stderr, "%04d-%02d-%02d %02d:%02d:%02d.%03ld [%s] " fmt "\n", \
                _tm.tm_year+1900, _tm.tm_mon+1, _tm.tm_mday, \
                _tm.tm_hour, _tm.tm_min, _tm.tm_sec, _ts.tv_nsec/1000000, \
                tag, ##__VA_ARGS__); \
    } \
} while(0)

#define LOG_ERROR(fmt, ...) MNEMO_LOG(MNEMO_LOG_ERROR, "ERROR", fmt, ##__VA_ARGS__)
#define LOG_WARN(fmt, ...)  MNEMO_LOG(MNEMO_LOG_WARN,  "WARN",  fmt, ##__VA_ARGS__)
#define LOG_INFO(fmt, ...)  MNEMO_LOG(MNEMO_LOG_INFO,  "INFO",  fmt, ##__VA_ARGS__)
#define LOG_DEBUG(fmt, ...) MNEMO_LOG(MNEMO_LOG_DEBUG, "DEBUG", fmt, ##__VA_ARGS__)

#endif
