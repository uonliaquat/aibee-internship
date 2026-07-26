#include "../include/pretokenizer.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdbool.h>

typedef enum
{
    CLASS_SPACE,
    CLASS_LETTER,
    CLASS_DIGIT,
    CLASS_OTHER
}
CharClass;

static int decode_utf8(const char* text,size_t pos,size_t len,unsigned int* out_codepoint)
{
    unsigned char b0=(unsigned char)text[pos];
    if(b0<0x80) //0*80 is 128 // ascii 1 byte occupies 0 to 127
    {
        *out_codepoint =b0;
        return 1;
    }
    if((b0 & 0xE0) ==0xC0 && pos +1 <len)
    {
        unsigned char b1= (unsigned char)text[pos+1];
        *out_codepoint=((b0 & 0x1F) << 6) | (b1 & 0x3F);
        return 2;
    }
    if((b0 & 0xF0) == 0xE0 && pos + 2 < len) //
    {
        unsigned char b1 = (unsigned char)text[pos+1];
        unsigned char b2 = (unsigned char)text[pos+2];
        *out_codepoint = ((b0 & 0x0F) << 12) | ((b1 & 0x3F) << 6) | (b2 & 0x3F);
        return 3;
    }
    if((b0 & 0xF8) == 0xF0 && pos + 3 < len)
    {
        unsigned char b1 = (unsigned char)text[pos+1];
        unsigned char b2 = (unsigned char)text[pos+2];
        unsigned char b3 = (unsigned char)text[pos+3];
        *out_codepoint = ((b0 & 0x07) << 18) | ((b1 & 0x3F) << 12) | ((b2 & 0x3F) << 6) | (b3 & 0x3F);
        return 4;
    }
    // malformed byte
    *out_codepoint = b0;
    return 1;
}

static int is_letter_codepoint(unsigned int cp)
{
    //check english
    if((cp>='a' && cp<='z') || (cp>='A' && cp<='Z'))
    {
        return 1;
    }
    if(cp >= 0x00C0 && cp <= 0x00D6) // Latin-1 Supplement (À to Ö)
    {
        return 1;
    }
    if(cp >= 0x00D8 && cp <= 0x00F6) // Latin-1 Supplement (Ø to ö)
    {
        return 1;
    }
    if(cp >= 0x00F8 && cp <= 0x024F) // Latin Extended-A and part of Latin Extended-B (ø to ɏ)
    {
        return 1;
    }
    if(cp >= 0x0370 && cp <= 0x03FF) // Greek and Coptic block (Α, Β, Γ, α, β, γ.)
    {
        return 1;
    }
    if(cp >= 0x0400 && cp <= 0x04FF) // Cyrillic block (Русский и другие славянские языки)
    {
        return 1;
    }
    if(cp >= 0x3040 && cp <= 0x30FF) // Hiragana and Katakana (Японская письменность)
    {
        return 1;
    }

    if(cp >= 0x4E00 && cp <= 0x9FFF) // CJK Unified Ideographs (Китайские иероглифы)
    {
        return 1;
    }
    return 0; // dont support other language
}

static int is_digit_codepoint(unsigned int cp)
{
    return cp>='0' && cp<='9';
}

static int is_space_codepoint(unsigned int cp)
{
    return cp==' '||cp=='\t'||cp=='\n'||cp=='\r'||cp=='\f'||cp=='\v'||cp==0x00A0;
}

static CharClass classify_codepoint(unsigned int cp)
{
    if(is_space_codepoint(cp))
    {
        return CLASS_SPACE;
    }
    if(is_letter_codepoint(cp))
    {
        return CLASS_LETTER;
    }
    if(is_digit_codepoint( cp))
    {
        return CLASS_DIGIT;
    }
    return CLASS_OTHER;
}

static const char* CONTRACTIONS[] = {"'s", "'t","'re","'ve", "'m","'ll", "'d"};
static const int CONTRACTION_COUNT = 7;

static int match_contraction(const char* text,size_t pos,size_t len)
{
    if(text[pos] != '\'')
    {
        return 0;
    }
    for(int i=0;i < CONTRACTION_COUNT;i++)
    {
        size_t clen=strlen(CONTRACTIONS[i]);

        if(pos+ clen <=len && strncmp(text+pos,CONTRACTIONS[i],clen)==0)
        {
            return (int)clen;
        }
    }
    return 0;
}

static SplitResult* split_result_create(size_t initial_capacity)
{
    SplitResult* result=malloc(sizeof(SplitResult));
    if(!result)
    {
        fprintf(stderr,"ERROR: Failed to allocate split result\n");
        return NULL;
    }

    if (initial_capacity < 16)
    {
        initial_capacity = 16;
    }

    result->spans=malloc(initial_capacity*sizeof(TokenSpan));
    if(!result->spans)
    {
        free(result);
        fprintf(stderr,  "ERROR: Failed to allocate token spans\n");
        return NULL;
    }
    result->count=0;
    result->capacity=initial_capacity;
    return result;
}

static bool push_span(SplitResult* result,size_t start,size_t length)
{
    if(length == 0)
    {
        return true;
    }
    if(result->count >= result->capacity)
    {
        size_t new_capacity=result->capacity*2;
        TokenSpan* new_spans=realloc(result->spans,new_capacity * sizeof(TokenSpan));
        if(!new_spans)
        {
            fprintf(stderr, "ERROR: Failed to resize token spans\n");
            return false;
        }
        result->spans =new_spans;
        result->capacity = new_capacity;
    }
    result->spans[result->count].start =start;
    result->spans[result->count].length =length;
    result->count++;
    return true;
}

SplitResult* pretokenize(const char* text)
{
    if(!text)
    {
        fprintf(stderr,"ERROR: pretokenize called with NULL text\n");
        return NULL;
    }
    size_t len = strlen(text);
    SplitResult* result = split_result_create(len/3 + 8);
    if(!result)
    {
        return NULL;

    }

    size_t pos = 0;
    while(pos < len)
    {
        int contraction_len = match_contraction(text,pos,len);
        if(contraction_len > 0)
        {
            if(push_span(result,pos,(size_t)contraction_len))
            {
                pos += contraction_len;
                continue;
            }
            split_result_destroy(result);
            return NULL;
        }

        unsigned int cp;

        int cp_len = decode_utf8(text,pos,len,&cp);
        CharClass cls = classify_codepoint(cp);

        if(cls == CLASS_SPACE)
        {
            size_t run_start = pos;

            size_t peek = pos + cp_len;

            if(peek < len)
            {
                unsigned int next_cp;
                decode_utf8(text,peek,len,&next_cp);

                CharClass next_cls = classify_codepoint(next_cp);

                if(next_cls != CLASS_SPACE)
                {
                    pos = peek;
                    while(pos < len)
                    {
                        unsigned int run_cp;
                        int run_len = decode_utf8(text,pos,len,&run_cp);
                        if(classify_codepoint(run_cp) != next_cls)
                        {
                            break;
                        }
                        pos += run_len;
                    }
                    if(push_span(result, run_start, pos - run_start))
                    {
                        continue;
                    }
                    split_result_destroy(result);
                    return NULL;
                }
            }

            size_t last_space_pos = pos;
            while(pos < len)
            {
                unsigned int space_cp;

                int space_len = decode_utf8(text,pos,len,&space_cp);
                if(classify_codepoint(space_cp) != CLASS_SPACE)
                {
                    break;
                }
                last_space_pos = pos;
                pos += space_len;
            }

            if(pos < len)
            {
                pos = last_space_pos;
            }

            if(push_span(result,run_start,pos - run_start))
            {
                continue;
            }
            split_result_destroy(result);
            return NULL;
        }

        size_t chunk_start = pos;

        while(pos < len)
        {
            unsigned int run_cp;
            int run_len = decode_utf8(text,pos,len,&run_cp);

            if(classify_codepoint(run_cp) != cls)
            {
                break;
            }
            pos += run_len;
        }
        if(!push_span(result,chunk_start,pos - chunk_start))
        {
            split_result_destroy(result);
            return NULL;
        }
    }
    return result;
}

void split_result_destroy(SplitResult* result)
{
    if(!result)
    {
        return;
    }
    free(result->spans);

    free(result);
}

char* pretokenizer_get_chunk(const char* text,const TokenSpan* span)
{
    if(!text || !span)
    {
        return NULL;
    }
    char* chunk = malloc(span->length + 1);
    if(!chunk)
    {
        fprintf(stderr,"ERROR: Failed to allocate chunk copy\n");
        return NULL;
    }
    memcpy(chunk,text + span->start,span->length);
    chunk[span->length] = '\0';
    return chunk;
}

void pretokenizer_print_chunks(const char* text,const SplitResult* result)
{
    if(!result)
    {
        printf("Split result: NULL\n");
        return;
    }
    printf("=== Pretokenizer Chunks ===\n");
    printf("Total chunks: %zu\n",result->count);
    for(size_t i = 0;i < result->count;i++)
    {
        char* chunk = pretokenizer_get_chunk(text,&result->spans[i]);
        if(chunk)
        {
            printf("[%zu] \"%s\" (start=%zu, len=%zu)\n",i,chunk,result->spans[i].start,result->spans[i].length);
            free(chunk);
        }
    }
    printf("===========================\n");
}