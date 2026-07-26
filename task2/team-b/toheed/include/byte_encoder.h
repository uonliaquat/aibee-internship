#ifndef BYTE_ENCODER_H
#define BYTE_ENCODER_H

/*
 
 * This file contains functions used to convert bytes into the
 * special Unicode characters used by the GPT-2 tokenizer.
 * It also converts them back to the original bytes.
 */

/* Initialize the byte encoder.
 * Call this once before using the encoder functions.
 */

void byte_encoder_init(void);

/* Convert a normal string into GPT-2 byte-level Unicode format.
 *
 * Returns a newly allocated string.
 * Free the returned memory after use.
 */

char *byte_encode(const char *text);


/* Convert a GPT-2 byte-level Unicode string back to the original text.
 *
 * Returns a newly allocated string.
 * Free the returned memory after use.
 */

char *byte_decode(const char *text);

/* Convert a single byte into its GPT-2 Unicode representation.
 *
 * `out` should have space for at least 5 characters.
 * Returns the number of bytes written.
 */
 
int byte_encode_single(unsigned char byte, char *out);

#endif 