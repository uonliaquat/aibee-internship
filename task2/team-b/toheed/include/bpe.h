#ifndef BPE_H
#define BPE_H

#include <stddef.h>
#include "tokenizer_loader.h"

/*
 * bpe.h
 *
 * This file implements the GPT-2 Byte Pair Encoding (BPE) algorithm.
 * It converts text into token IDs by creating byte-level symbols,
 * merging them, and looking them up in the vocabulary.
 */

/* Convert a text chunk into GPT-2 byte-level symbols.
 * One symbol is created for each input byte.
 *
 * Returns an array of symbols and stores the count in out_count.
 */
char **chunk_to_symbols(const char *chunk, size_t *out_count);
/* Free the memory used by a symbol array. */
void free_symbols(char **symbols, size_t count);
/* Find the next pair of symbols that should be merged.
 *
 * Returns 1 if a mergeable pair is found, otherwise 0.
 */
int find_best_pair(const TokenizerData *data, char **symbols, size_t count,
                    size_t *best_idx);

/* Merge two adjacent symbols into one.
 * The symbol array size is reduced by one.
 */
void merge_symbols(char ***symbols_ptr, size_t *count_ptr, size_t idx);

/* Apply BPE to a single text chunk and convert it into token IDs.
 *
 * Returns an array of token IDs.
 */

int *bpe_encode_chunk(const TokenizerData *data, const char *chunk,
                       size_t *out_count);

/* Encode the complete input text into GPT-2 token IDs.
 *
 * Returns an array of token IDs.
 */
int *encode_text(const TokenizerData *data, const char *text,
                  size_t *out_count);

#endif 