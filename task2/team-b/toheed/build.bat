@echo off
echo Building GPT Tokenizer...

:: Create obj and bin directories if they don't exist
if not exist obj mkdir obj
if not exist bin mkdir bin

:: Compile object files
gcc -Wall -Wextra -O2 -g -Iinclude -c src/hashmap.c -o obj/hashmap.o
gcc -Wall -Wextra -O2 -g -Iinclude -c src/merge_table.c -o obj/merge_table.o
gcc -Wall -Wextra -O2 -g -Iinclude -c src/tokenizer_loader.c -o obj/tokenizer_loader.o
gcc -Wall -Wextra -O2 -g -Iinclude -c src/pretokenizer.c -o obj/pretokenizer.o
gcc -Wall -Wextra -O2 -g -Iinclude -c src/bpe.c -o obj/bpe.o
gcc -Wall -Wextra -O2 -g -Iinclude -c src/byte_encoder.c -o obj/byte_encoder.o

:: Compile tokenizer loader test
gcc -Wall -Wextra -O2 -g -Iinclude ^
obj/hashmap.o ^
obj/merge_table.o ^
obj/tokenizer_loader.o ^
tests/test.c ^
-o bin/test_loader -lm

:: Compile pretokenizer test
gcc -Wall -Wextra -O2 -g -Iinclude ^
obj/pretokenizer.o ^
tests/test_pretokenizer.c ^
-o bin/test_pretokenizer

gcc -Wall -Wextra -O2 -g -Iinclude ^
obj/hashmap.o ^
obj/merge_table.o ^
obj/tokenizer_loader.o ^
obj/pretokenizer.o ^
obj/bpe.o ^
obj/byte_encoder.o ^
tests/test_bpe.c ^
-o bin/test_bpe -lm

:: Compile compare_outputs
gcc -Wall -Wextra -O2 -g -Iinclude ^
obj/hashmap.o ^
obj/merge_table.o ^
obj/tokenizer_loader.o ^
obj/pretokenizer.o ^
obj/bpe.o ^
obj/byte_encoder.o ^
tests/compare_outputs.c ^
-o bin/compare_outputs -lm

echo Build complete!
echo Binaries are in bin/ directory
pause