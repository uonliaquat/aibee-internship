#include "bpe.h"
#include "byte_encoder.h"
#include "tokenizer_loader.h"
#include "pretokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char **chunk_to_symbols(const char *chunk, size_t *out_count) {
    if (!chunk) {
        *out_count = 0;
        return NULL;
    }

    size_t len = strlen(chunk);
    if (len == 0) {
        *out_count = 0;
        return NULL;
    }

    char **symbols = (char **)malloc(len * sizeof(char *));
    if (!symbols) {
        *out_count = 0;
        return NULL;
    }

    const unsigned char *bytes = (const unsigned char *)chunk;

    for (size_t i = 0; i < len; i++) {
        char buf[5];
        int n = byte_encode_single(bytes[i], buf);

        char *sym = (char *)malloc((size_t)n + 1);
        if (!sym) {
           
            for (size_t j = 0; j < i; j++) free(symbols[j]);
            free(symbols);
            *out_count = 0;
            return NULL;
        }
        memcpy(sym, buf, (size_t)n);
        sym[n] = '\0';
        symbols[i] = sym;
    }

    *out_count = len;
    return symbols;
}

void free_symbols(char **symbols, size_t count) {
    if (!symbols) return;
    for (size_t i = 0; i < count; i++) {
        free(symbols[i]);
    }
    free(symbols);
}


int find_best_pair(const TokenizerData *data, char **symbols, size_t count,
                    size_t *best_idx) {
    if (count < 2) return 0;

    int best_rank = 0;
    size_t chosen = 0;
    int found = 0;

    for (size_t i = 0; i + 1 < count; i++) {
        int rank = tokenizer_get_merge_rank(data, symbols[i], symbols[i + 1]);

#ifdef BPE_DEBUG
        fprintf(stderr, "  pair[%zu] = \"%s\" + \"%s\" -> rank %d\n",
                i, symbols[i], symbols[i + 1], rank);
#endif

        /* A negative rank means "t1 t2" is not a known merge. */
        if (rank < 0) continue;

        if (!found || rank < best_rank) {
            best_rank = rank;
            chosen = i;
            found = 1;
        }
    }

#ifdef BPE_DEBUG
    if (found) {
        fprintf(stderr, "  -> merging pair[%zu] = \"%s\" + \"%s\" (rank %d)\n\n",
                chosen, symbols[chosen], symbols[chosen + 1], best_rank);
    } else {
        fprintf(stderr, "  -> no mergeable pair found\n\n");
    }
#endif

    if (found) *best_idx = chosen;
    return found;
}

void merge_symbols(char ***symbols_ptr, size_t *count_ptr, size_t idx) {
    char **symbols = *symbols_ptr;
    size_t count = *count_ptr;

    if (idx + 1 >= count) return;

    size_t len1 = strlen(symbols[idx]);
    size_t len2 = strlen(symbols[idx + 1]);

    char *merged = (char *)malloc(len1 + len2 + 1);
    if (!merged) return; 

    memcpy(merged, symbols[idx], len1);
    memcpy(merged + len1, symbols[idx + 1], len2);
    merged[len1 + len2] = '\0';

    free(symbols[idx]);
    free(symbols[idx + 1]);
    symbols[idx] = merged;

    /* Shift everything after idx+1 left by one slot. */
    for (size_t i = idx + 1; i + 1 < count; i++) {
        symbols[i] = symbols[i + 1];
    }
    count--;

    if (count > 0) {
        char **shrunk = (char **)realloc(symbols, count * sizeof(char *));
        if (shrunk) symbols = shrunk;
    } else {
        free(symbols);
        symbols = NULL;
    }

    *symbols_ptr = symbols;
    *count_ptr = count;
}


int *bpe_encode_chunk(const TokenizerData *data, const char *chunk,
                       size_t *out_count) {
    if (!chunk || chunk[0] == '\0') {
        *out_count = 0;
        return NULL;
    }

    char *whole = byte_encode(chunk);
    if (whole) {
        int whole_id;
        if (tokenizer_lookup_id(data, whole, &whole_id)) {
            free(whole);
            int *ids = (int *)malloc(sizeof(int));
            if (!ids) {
                *out_count = 0;
                return NULL;
            }
            ids[0] = whole_id;
            *out_count = 1;
            return ids;
        }
        free(whole);
    }

    size_t count = 0;
    char **symbols = chunk_to_symbols(chunk, &count);

    if (count == 0) {
        *out_count = 0;
        return NULL;
    }

    while (count > 1) {
        size_t best_idx;
        if (!find_best_pair(data, symbols, count, &best_idx)) break;
        merge_symbols(&symbols, &count, best_idx);
    }

    int *ids = (int *)malloc(count * sizeof(int));
    if (!ids) {
        free_symbols(symbols, count);
        *out_count = 0;
        return NULL;
    }

    for (size_t i = 0; i < count; i++) {
        int id = -1;
        if (!tokenizer_lookup_id(data, symbols[i], &id)) {
            fprintf(stderr,
                    "bpe: warning - symbol \"%s\" not found in vocab\n",
                    symbols[i]);
            id = -1;
        }
        ids[i] = id;
    }

    free_symbols(symbols, count);
    *out_count = count;
    return ids;
}

int *encode_text(const TokenizerData *data, const char *text,
                  size_t *out_count) {
    *out_count = 0;

    if (!text) return NULL;

    SplitResult *result = pretokenize(text);
    if (!result) return NULL;

    size_t cap = 64;
    size_t total = 0;
    int *ids = (int *)malloc(cap * sizeof(int));
    if (!ids) {
        split_result_destroy(result);
        return NULL;
    }

    for (size_t c = 0; c < result->count; c++) {
        char *chunk = pretokenizer_get_chunk(text, &result->spans[c]);
        if (!chunk) continue;

        size_t n = 0;
        int *chunk_ids = bpe_encode_chunk(data, chunk, &n);
        free(chunk);

        if (n == 0) {
            free(chunk_ids);
            continue;
        }

        if (total + n > cap) {
            while (total + n > cap) cap *= 2;
            int *grown = (int *)realloc(ids, cap * sizeof(int));
            if (!grown) {
                free(chunk_ids);
                free(ids);
                split_result_destroy(result);
                *out_count = 0;
                return NULL;
            }
            ids = grown;
        }

        memcpy(ids + total, chunk_ids, n * sizeof(int));
        total += n;
        free(chunk_ids);
    }

    split_result_destroy(result);
    *out_count = total;
    return ids;
}