#ifndef PRETOKENIZER_H
#define PRETOKENIZER_H

#include <stddef.h>

typedef struct
{
   size_t start;
   size_t length;
}
TokenSpan;

typedef struct
{
  TokenSpan* spans;
  size_t count;
  size_t capacity;
}
SplitResult;

SplitResult* pretokenize(const char* text);
void split_result_destroy(SplitResult* result);
char* pretokenizer_get_chunk(const char* text,const TokenSpan* span);
void pretokenizer_print_chunks(const char* text,const SplitResult* result);

#endif