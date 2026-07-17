#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
typedef struct {
    const uint8_t *base;
    size_t size;
    size_t pos;
} gguf_reader;

static uint8_t rd_u8(gguf_reader *r) {
    return r->base[r->pos++];
}

static uint16_t rd_u16(gguf_reader *r) {
    uint16_t v;
    memcpy(&v, r->base + r->pos, 2);
    r->pos += 2;
    return v;
}

static uint32_t rd_u32(gguf_reader *r) {
    uint32_t v;
    memcpy(&v, r->base + r->pos, 4);
    r->pos += 4;
    return v;
}

static uint64_t rd_u64(gguf_reader *r) {
    uint64_t v;
    memcpy(&v, r->base + r->pos, 8);
    r->pos += 8;
    return v;
}

static float rd_f32(gguf_reader *r) {
    uint32_t bits = rd_u32(r);
    float v;
    memcpy(&v, &bits, 4);
    return v;
}

static double rd_f64(gguf_reader *r) {
    uint64_t bits = rd_u64(r);
    double v;
    memcpy(&v, &bits, 8);
    return v;
}

typedef struct {
    char *data;
    uint64_t len;
} gguf_str;

static gguf_str rd_str(gguf_reader *r) {
    uint64_t len = rd_u64(r);
    gguf_str s;
    s.len = len;
    s.data = malloc(len + 1);
    if (len > 0) memcpy(s.data, r->base + r->pos, len);
    s.data[len] = '\0';
    r->pos += len;
    return s;
}
typedef struct {
    uint32_t version;
    uint64_t tensor_count;
    uint64_t kv_count;
} gguf_header;

static gguf_header parse_header(gguf_reader *r) {
    if (r->size < 4 || memcmp(r->base, "GGUF", 4) != 0) {
        fprintf(stderr, "not a GGUF file (bad magic bytes)\n");
        exit(1);
    }
    r->pos = 4;

    gguf_header h;
    h.version = rd_u32(r);
    h.tensor_count = rd_u64(r);
    h.kv_count = rd_u64(r);
    return h;
}
int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s model.gguf [tensor_name]\n", argv[0]);
        return 1;
    }
    const char *path = argv[1];
    const char *tensor_name = (argc >= 3) ? argv[2] : NULL;

    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror("open"); return 1; }

    struct stat st;
    if (fstat(fd, &st) != 0) { perror("fstat"); close(fd); return 1; }

    uint8_t *map = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (map == MAP_FAILED) { perror("mmap"); close(fd); return 1; }

    gguf_reader r = { .base = map, .size = (size_t)st.st_size, .pos = 0 };

    gguf_header h = parse_header(&r);
    printf("=== HEADER ===\n");
    printf("  magic:         GGUF\n");
    printf("  version:       %u\n", h.version);
    printf("  tensor_count:  %llu\n", (unsigned long long)h.tensor_count);
    printf("  kv_count:      %llu\n", (unsigned long long)h.kv_count);

    close(fd);
    return 0;
}