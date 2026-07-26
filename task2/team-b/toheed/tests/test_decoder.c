#include "byte_decoder.h"
#include "byte_encoder.h"
#include "tokenizer_loader.h"
#include <stdio.h>
#include <stdlib.h>

static void test_decoder(const TokenizerData *data, const int *ids, size_t count) {
    char *result = decode_text(data, ids, count);
    if (!result) {
        fprintf(stderr, "decode_text failed\n");
        return;
    }
    printf("%s\n", result);
    free(result);
}

int main(void) {
    byte_encoder_init();

    TokenizerData *data = tokenizer_init("data/vocab.json", "data/merges.txt");
    if (!data) {
        fprintf(stderr, "test_decoder: failed to load vocab.json / merges.txt\n");
        return 1;
    }

    int ids[] = { 47249, 222, 47249, 224, 8582, 97, 96};
    test_decoder(data, ids, sizeof(ids) / sizeof(ids[0]));

    int ids2[] = {13409, 3435, 1332, 25, 5299, 0, 41573, 3, 61, 5, 9, 3419, 62, 10, 63, 12, 34758, 92, 91, 21737, 59, 25, 8172, 27, 29, 21747, 19571};
    test_decoder(data, ids2, sizeof(ids2) / sizeof(ids2[0]));

    tokenizer_free(data);
    return 0;
}
