#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

typedef enum {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
} gguf_value_type;

typedef enum {
    GGML_TYPE_F32  = 0,
    GGML_TYPE_F16  = 1,
    GGML_TYPE_Q4_0 = 2,
    GGML_TYPE_Q4_1 = 3,
    GGML_TYPE_Q5_0 = 6,
    GGML_TYPE_Q5_1 = 7,
    GGML_TYPE_Q8_0 = 8,
    GGML_TYPE_Q8_1 = 9,
} ggml_type;


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

static const char *gguf_value_type_name(uint32_t t) {
    switch (t) {
        case GGUF_TYPE_UINT8:   return "u8";
        case GGUF_TYPE_INT8:    return "i8";
        case GGUF_TYPE_UINT16:  return "u16";
        case GGUF_TYPE_INT16:   return "i16";
        case GGUF_TYPE_UINT32:  return "u32";
        case GGUF_TYPE_INT32:   return "i32";
        case GGUF_TYPE_FLOAT32: return "f32";
        case GGUF_TYPE_BOOL:    return "bool";
        case GGUF_TYPE_STRING:  return "str";
        case GGUF_TYPE_ARRAY:   return "array";
        case GGUF_TYPE_UINT64:  return "u64";
        case GGUF_TYPE_INT64:   return "i64";
        case GGUF_TYPE_FLOAT64: return "f64";
        default:                return "unknown";
    }
}


static void gguf_read_value(gguf_reader *r, uint32_t type, int print, int depth) {
    switch (type) {
        case GGUF_TYPE_UINT8: {
            uint8_t v = rd_u8(r);
            if (print) printf("%u", v);
            return;
        }
        case GGUF_TYPE_INT8: {
            int8_t v = (int8_t)rd_u8(r);
            if (print) printf("%d", v);
            return;
        }
        case GGUF_TYPE_BOOL: {
            uint8_t v = rd_u8(r);
            if (print) printf("%s", v ? "true" : "false");
            return;
        }
        case GGUF_TYPE_UINT16: {
            uint16_t v = rd_u16(r);
            if (print) printf("%u", v);
            return;
        }
        case GGUF_TYPE_INT16: {
            int16_t v = (int16_t)rd_u16(r);
            if (print) printf("%d", v);
            return;
        }
        case GGUF_TYPE_UINT32: {
            uint32_t v = rd_u32(r);
            if (print) printf("%u", v);
            return;
        }
        case GGUF_TYPE_INT32: {
            int32_t v = (int32_t)rd_u32(r);
            if (print) printf("%d", v);
            return;
        }
        case GGUF_TYPE_FLOAT32: {
            float v = rd_f32(r);
            if (print) printf("%g", v);
            return;
        }
        case GGUF_TYPE_UINT64: {
            uint64_t v = rd_u64(r);
            if (print) printf("%llu", (unsigned long long)v);
            return;
        }
        case GGUF_TYPE_INT64: {
            int64_t v = (int64_t)rd_u64(r);
            if (print) printf("%lld", (long long)v);
            return;
        }
        case GGUF_TYPE_FLOAT64: {
            double v = rd_f64(r);
            if (print) printf("%g", v);
            return;
        }
        case GGUF_TYPE_STRING: {
            gguf_str s = rd_str(r);
            if (print) printf("\"%s\"", s.data);
            free(s.data);
            return;
        }
        case GGUF_TYPE_ARRAY: {
            uint32_t elem_type = rd_u32(r);
            uint64_t n = rd_u64(r);
            if (print) {
                printf("[%s x%llu]", gguf_value_type_name(elem_type),
                       (unsigned long long)n);
                if (elem_type != GGUF_TYPE_ARRAY && n > 0) {
                    printf(" = [");
                    uint64_t preview = n < 8 ? n : 8;
                    for (uint64_t i = 0; i < preview; i++) {
                        if (i > 0) printf(", ");
                        gguf_read_value(r, elem_type, 1, depth + 1);
                    }
                    if (n > preview) printf(", ...");
                    printf("]");
                    for (uint64_t i = preview; i < n; i++)
                        gguf_read_value(r, elem_type, 0, depth + 1);
                    return;
                }
            }
            for (uint64_t i = 0; i < n; i++)
                gguf_read_value(r, elem_type, 0, depth + 1);
            return;
        }
        default:
            fprintf(stderr, "unknown metadata value type %u at pos %zu\n",
                    type, r->pos);
            exit(1);
    }
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


static uint64_t parse_and_print_metadata(gguf_reader *r, uint64_t kv_count) {
    uint64_t alignment = 32; /* GGUF default */

    printf("=== METADATA (%llu keys) ===\n", (unsigned long long)kv_count);
    for (uint64_t i = 0; i < kv_count; i++) {
        gguf_str key = rd_str(r);
        uint32_t vtype = rd_u32(r);

        printf("  %s (%s) = ", key.data, gguf_value_type_name(vtype));


        if (vtype == GGUF_TYPE_UINT32 && strcmp(key.data, "general.alignment") == 0) {
            size_t save = r->pos;
            uint32_t v = rd_u32(r);
            r->pos = save; /* rewind so gguf_read_value below reads it again */
            alignment = v;
        }

        gguf_read_value(r, vtype, 1, 0);
        printf("\n");
        free(key.data);
    }
    return alignment;
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

    uint64_t alignment = parse_and_print_metadata(&r, h.kv_count);

    close(fd);
    return 0;
}