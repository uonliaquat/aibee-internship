
#include "byte_encoder.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NUM_BYTES 256
#define MAX_CODEPOINT 324

static unsigned int byte_to_unicode[NUM_BYTES];
static int unicode_to_byte[MAX_CODEPOINT];
static int g_initialized = 0;


static int utf8_encode_codepoint(unsigned int cp, char *out) {
    if (cp < 0x80) {
        out[0] = (char)cp;
        return 1;
    } else if (cp < 0x800) {
        out[0] = (char)(0xC0 | (cp >> 6));
        out[1] = (char)(0x80 | (cp & 0x3F));
        return 2;
    } else if (cp < 0x10000) {
        out[0] = (char)(0xE0 | (cp >> 12));
        out[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        out[2] = (char)(0x80 | (cp & 0x3F));
        return 3;
    } else {
        out[0] = (char)(0xF0 | (cp >> 18));
        out[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
        out[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
        out[3] = (char)(0x80 | (cp & 0x3F));
        return 4;
    }
}

static int utf8_decode_codepoint(const unsigned char *s, unsigned int *cp_out) {
    if (s[0] == 0) return 0;

    if ((s[0] & 0x80) == 0x00) {
        *cp_out = s[0];
        return 1;
    }
    if ((s[0] & 0xE0) == 0xC0) {
        if (s[1] == 0 || (s[1] & 0xC0) != 0x80) return 0;
        *cp_out = ((unsigned int)(s[0] & 0x1F) << 6) |
                  ((unsigned int)(s[1] & 0x3F));
        return 2;
    }
    if ((s[0] & 0xF0) == 0xE0) {
        if (s[1] == 0 || s[2] == 0) return 0;
        if ((s[1] & 0xC0) != 0x80 || (s[2] & 0xC0) != 0x80) return 0;
        *cp_out = ((unsigned int)(s[0] & 0x0F) << 12) |
                  ((unsigned int)(s[1] & 0x3F) << 6) |
                  ((unsigned int)(s[2] & 0x3F));
        return 3;
    }
    if ((s[0] & 0xF8) == 0xF0) {
        if (s[1] == 0 || s[2] == 0 || s[3] == 0) return 0;
        if ((s[1] & 0xC0) != 0x80 || (s[2] & 0xC0) != 0x80 || (s[3] & 0xC0) != 0x80) return 0;
        *cp_out = ((unsigned int)(s[0] & 0x07) << 18) |
                  ((unsigned int)(s[1] & 0x3F) << 12) |
                  ((unsigned int)(s[2] & 0x3F) << 6) |
                  ((unsigned int)(s[3] & 0x3F));
        return 4;
    }

    /* Invalid leading byte (stray continuation byte, etc). */
    return 0;
}


void byte_encoder_init(void) {
    if (g_initialized) return;

    int is_printable[NUM_BYTES];
    memset(is_printable, 0, sizeof(is_printable));

    int bs[NUM_BYTES];
    int bs_count = 0;
    int b;

    /* range(ord('!'), ord('~')+1) -> 33..126 */
    for (b = '!'; b <= '~'; b++) {
        bs[bs_count++] = b;
        is_printable[b] = 1;
    }
    /* range(ord('¡'), ord('¬')+1) -> 161..172 (0xA1..0xAC) */
    for (b = 0xA1; b <= 0xAC; b++) {
        bs[bs_count++] = b;
        is_printable[b] = 1;
    }
    /* range(ord('®'), ord('ÿ')+1) -> 174..255 (0xAE..0xFF) */
    for (b = 0xAE; b <= 0xFF; b++) {
        bs[bs_count++] = b;
        is_printable[b] = 1;
    }

    for (int i = 0; i < MAX_CODEPOINT; i++) {
        unicode_to_byte[i] = -1;
    }

    /* Printable bytes map to themselves. */
    for (int i = 0; i < bs_count; i++) {
        byte_to_unicode[bs[i]] = (unsigned int)bs[i];
        unicode_to_byte[bs[i]] = bs[i];
    }

    int n = 0;
    for (b = 0; b < NUM_BYTES; b++) {
        if (!is_printable[b]) {
            unsigned int cp = 256u + (unsigned int)n;
            byte_to_unicode[b] = cp;
            unicode_to_byte[cp] = b;
            n++;
        }
    }

    g_initialized = 1;
}

int byte_encode_single(unsigned char byte, char *out) {
    if (!g_initialized) byte_encoder_init();
    unsigned int cp = byte_to_unicode[byte];
    return utf8_encode_codepoint(cp, out);
}

char *byte_encode(const char *text) {
    if (!g_initialized) byte_encoder_init();
    if (!text) return NULL;

    size_t len = strlen(text);
    char *out = (char *)malloc(len * 4 + 1);
    if (!out) return NULL;

    size_t pos = 0;
    const unsigned char *bytes = (const unsigned char *)text;

    for (size_t i = 0; i < len; i++) {
        char buf[5];
        int n = byte_encode_single(bytes[i], buf);
        memcpy(out + pos, buf, (size_t)n);
        pos += (size_t)n;
    }

    out[pos] = '\0';
    return out;
}

char *byte_decode(const char *text) {
    if (!g_initialized) byte_encoder_init();
    if (!text) return NULL;

    size_t len = strlen(text);
    /* Decoded output is always <= encoded input length (each decoded
     * byte comes from at least one encoded byte). */
    char *out = (char *)malloc(len + 1);
    if (!out) return NULL;

    size_t pos = 0;
    const unsigned char *s = (const unsigned char *)text;
    size_t i = 0;

    while (s[i] != 0) {
        unsigned int cp = 0;
        int consumed = utf8_decode_codepoint(s + i, &cp);

        if (consumed == 0) {
            /* Malformed UTF-8 in input; skip a byte to avoid an
             * infinite loop and keep decoding best-effort. */
            i++;
            continue;
        }

        int byte_val = -1;
        if (cp < (unsigned int)MAX_CODEPOINT) {
            byte_val = unicode_to_byte[cp];
        }

        if (byte_val >= 0) {
            out[pos++] = (char)byte_val;
        }
        /* If byte_val < 0 the codepoint wasn't produced by
         * byte_encode() (e.g. arbitrary unicode text was passed in
         * directly) - it is silently dropped rather than corrupting
         * output, since it has no corresponding raw byte. */

        i += (size_t)consumed;
    }

    out[pos] = '\0';
    return out;
}