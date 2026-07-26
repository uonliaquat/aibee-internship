#include "byte_decoder.h"
#include "byte_encoder.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char *decode_text(const TokenizerData *data, const int *ids, size_t count) {
    if (!data || (!ids && count > 0)) return NULL;

    size_t cap = 64;
    size_t len = 0;
    char *buf = malloc(cap);
    if (!buf) return NULL;
    buf[0] = '\0';

    for (size_t i = 0; i < count; i++) {
        const char *piece = NULL;
        if (!tokenizer_lookup_token(data, ids[i], &piece) || !piece) {
            fprintf(stderr, "decode_text: id %d not found in vocab\n", ids[i]);
            free(buf);
            return NULL;
        }

        size_t piece_len = strlen(piece);
        if (len + piece_len + 1 > cap) {
            while (len + piece_len + 1 > cap) {
                cap *= 2;
            }
            char *grown = realloc(buf, cap);
            if (!grown) {
                free(buf);
                return NULL;
            }
            buf = grown;
        }

        memcpy(buf + len, piece, piece_len);
        len += piece_len;
        buf[len] = '\0';
    }

    char *result = byte_decode(buf);
    free(buf);
    return result;
}