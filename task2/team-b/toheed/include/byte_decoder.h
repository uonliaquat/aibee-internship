#ifndef BYTE_DECODER_H
#define BYTE_DECODER_H

#include "tokenizer_loader.h"
#include <stddef.h>

// Decodes an array of token IDs into a single byte-decoded text string.
// Note: The caller is responsible for freeing the returned memory buffer.
char *decode_text(const TokenizerData *data, const int *ids, size_t count);

#endif // BYTE_DECODER_H
