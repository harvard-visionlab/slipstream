# slipstream
Slipstream: Frictionless streaming and mmap-accelerated PyTorch dataloading

## Future Enhancements

### Faster S3 chunk downloads with s5cmd

Currently `OptimizedCache.build()` downloads LitData chunks via LitData's internal mechanisms, which can be slow (~2 min for 160 chunks). A potential optimization is to use [s5cmd](https://github.com/peak/s5cmd) for bulk parallel downloads before building the cache.

```bash
# s5cmd can download many files in parallel with high throughput
s5cmd cp "s3://bucket/dataset/chunks/*" /local/cache/chunks/
```

s5cmd has a `sync` command that only downloads missing/changed files:

```bash
# Sync chunks from S3 to local cache (only downloads what's missing)
s5cmd sync 's3://bucket/dataset/chunks/*' /local/cache/chunks/

# Use --size-only for faster comparison (skip modification time check)
s5cmd sync --size-only 's3://bucket/dataset/chunks/*' /local/cache/chunks/
```

This would allow a two-phase approach:
1. **Sync phase**: Use `s5cmd sync` to download all chunks in parallel (much faster than sequential HTTP)
2. **Build phase**: Build OptimizedCache from local chunks (no network latency)

Benefits:
- s5cmd is **32x faster than s3cmd** and **12x faster than aws-cli** for uploads, and can saturate a 40Gbps link for downloads
- `sync` only downloads missing files, making subsequent runs fast
- Chunks are fully local before processing, avoiding cache eviction issues
- Build phase becomes pure local I/O

See: [s5cmd GitHub](https://github.com/peak/s5cmd) | [AWS Blog: Parallelizing S3 Workloads with s5cmd](https://aws.amazon.com/blogs/opensource/parallelizing-s3-workloads-s5cmd/)
