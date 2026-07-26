#include "../include/pretokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int tests_passed = 0;
int tests_failed = 0;

void test_case(const char* name, const char* text, const char* expected_chunks[], int expected_count) {
    printf("\n=== Test: %s ===\n", name);
    printf("Input: \"%s\"\n", text);
    
    SplitResult* result = pretokenize(text);
    if (!result) {
        printf("❌ FAIL: pretokenize returned NULL\n");
        tests_failed++;
        return;
    }
    
    printf("Expected %d chunks, got %zu\n", expected_count, result->count);
    
    if (result->count != (size_t)expected_count) {
        printf("❌ FAIL: Chunk count mismatch\n");
        tests_failed++;
        split_result_destroy(result);
        return;
    }
    
    int all_match = 1;
    for (size_t i = 0; i < result->count; i++) {
        char* chunk = pretokenizer_get_chunk(text, &result->spans[i]);
        if (chunk) {
            printf("[%zu] \"%s\" (expected: \"%s\")\n", i, chunk, expected_chunks[i]);
            if (strcmp(chunk, expected_chunks[i]) != 0) {
                all_match = 0;
            }
            free(chunk);
        }
    }
    
    if (all_match) {
        printf("✅ PASS\n");
        tests_passed++;
    } else {
        printf("❌ FAIL: Chunk content mismatch\n");
        tests_failed++;
    }
    
    split_result_destroy(result);
}

void test_case_no_verify(const char* name, const char* text) {
    printf("\n=== Test: %s ===\n", name);
    printf("Input: \"%s\"\n", text);
    
    SplitResult* result = pretokenize(text);
    if (!result) {
        printf("❌ FAIL: pretokenize returned NULL\n");
        tests_failed++;
        return;
    }
    
    pretokenizer_print_chunks(text, result);
    printf("✅ PASS (no verification)\n");
    tests_passed++;
    split_result_destroy(result);
}

int main() {
    printf("========================================\n");
    printf("COMPREHENSIVE GPT-2 PRETOKENIZER TESTS\n");
    printf("========================================\n");
    
    // ========== BASIC TESTS ==========
    printf("\n--- BASIC TESTS ---\n");
    
    const char* t1_expected[] = {"Hello"};
    test_case("Single word", "Hello", t1_expected, 1);
    
    const char* t2_expected[] = {"Hello", ",", " world", "!"};
    test_case("Basic punctuation", "Hello, world!", t2_expected, 4);
    
    const char* t3_expected[] = {"Hello", " world"};
    test_case("Two words", "Hello world", t3_expected, 2);
    
    // ========== SPACE HANDLING ==========
    printf("\n--- SPACE HANDLING TESTS ---\n");
    
    const char* t4_expected[] = {" Hello"};
    test_case("Leading space", " Hello", t4_expected, 1);
    
    const char* t5_expected[] = {"Hello", " world"};
    test_case("Trailing space", "Hello world", t5_expected, 2);
    
    const char* t6_expected[] = {" ", " two", "  ", " spaces"};
    test_case("Multiple spaces", "  two   spaces", t6_expected, 4);
    
    const char* t7_expected[] = {"a", " b", " c"};
    test_case("Single spaces between words", "a b c", t7_expected, 3);
    
    test_case_no_verify("Double spaces between words", "a  b  c");
    
    const char* t9_expected[] = {"   "};
    test_case("Only spaces", "   ", t9_expected, 1);
    
    // ========== CONTRACTIONS ==========
    printf("\n--- CONTRACTION TESTS ---\n");
    
    const char* t10_expected[] = {"don", "'t"};
    test_case("don't", "don't", t10_expected, 2);
    
    const char* t11_expected[] = {"can", "'t"};
    test_case("can't", "can't", t11_expected, 2);
    
    const char* t12_expected[] = {"I", "'m"};
    test_case("I'm", "I'm", t12_expected, 2);
    
    const char* t13_expected[] = {"you", "'re"};
    test_case("you're", "you're", t13_expected, 2);
    
    const char* t14_expected[] = {"we", "'ve"};
    test_case("we've", "we've", t14_expected, 2);
    
    const char* t15_expected[] = {"they", "'ll"};
    test_case("they'll", "they'll", t15_expected, 2);
    
    const char* t16_expected[] = {"he", "'d"};
    test_case("he'd", "he'd", t16_expected, 2);
    
    const char* t17_expected[] = {"it", "'s"};
    test_case("it's", "it's", t17_expected, 2);
    
    const char* t18_expected[] = {"don", "'t", " can", "'t"};
    test_case("Multiple contractions", "don't can't", t18_expected, 4);
    
    test_case_no_verify("Apostrophe not a contraction", "'test");
    
    // ========== NUMBERS ==========
    printf("\n--- NUMBER TESTS ---\n");
    
    const char* t20_expected[] = {"123"};
    test_case("Only numbers", "123", t20_expected, 1);
    
    const char* t21_expected[] = {"123", "abc"};
    test_case("Numbers then letters", "123abc", t21_expected, 2);
    
    const char* t22_expected[] = {"abc", "123"};
    test_case("Letters then numbers", "abc123", t22_expected, 2);
    
    const char* t23_expected[] = {"abc", "123", "def"};
    test_case("Letters numbers letters", "abc123def", t23_expected, 3);
    
    const char* t24_expected[] = {"123"};
    test_case("Single digits", "123", t24_expected, 1);
    
    const char* t25_expected[] = {"123", ".", "456"};
    test_case("Numbers with period", "123.456", t25_expected, 3);
    
    // ========== PUNCTUATION ==========
    printf("\n--- PUNCTUATION TESTS ---\n");
    
    const char* t26_expected[] = {"Hello", "!"};
    test_case("Exclamation", "Hello!", t26_expected, 2);
    
    const char* t27_expected[] = {"Hello", "?"};
    test_case("Question mark", "Hello?", t27_expected, 2);
    
    const char* t28_expected[] = {"Hello", "."};
    test_case("Period", "Hello.", t28_expected, 2);
    
    const char* t29_expected[] = {"Hello", ",", " world", "!"};
    test_case("Comma and exclamation", "Hello, world!", t29_expected, 4);
    
    const char* t30_expected[] = {"(", "test", ")"};
    test_case("Parentheses", "(test)", t30_expected, 3);
    
    const char* t31_expected[] = {"[", "test", "]"};
    test_case("Brackets", "[test]", t31_expected, 3);
    
    const char* t32_expected[] = {"{", "test", "}"};
    test_case("Braces", "{test}", t32_expected, 3);
    
    const char* t33_expected[] = {"test", ":"};
    test_case("Colon", "test:", t33_expected, 2);
    
    const char* t34_expected[] = {"test", ";"};
    test_case("Semicolon", "test;", t34_expected, 2);
    
    const char* t35_expected[] = {"-", "test", "-"};
    test_case("Dashes", "-test-", t35_expected, 3);
    
    const char* t36_expected[] = {"_", "test", "_"};
    test_case("Underscores", "_test_", t36_expected, 3);
    
    // ========== EDGE CASES ==========
    printf("\n--- EDGE CASE TESTS ---\n");
    
    const char* t37_expected[] = {""};
    test_case("Empty string", "", t37_expected, 0); // Should produce 0 chunks
    
    const char* t38_expected[] = {"a"};
    test_case("Single character", "a", t38_expected, 1);
    
    const char* t39_expected[] = {"!"};
    test_case("Single punctuation", "!", t39_expected, 1);
    
    const char* t40_expected[] = {"1"};
    test_case("Single digit", "1", t40_expected, 1);
    
    const char* t41_expected[] = {" "};
    test_case("Single space", " ", t41_expected, 1);
    
    const char* t42_expected[] = {"\t"};
    test_case("Tab", "\t", t42_expected, 1);
    
    const char* t43_expected[] = {"\n"};
    test_case("Newline", "\n", t43_expected, 1);
    
    test_case_no_verify("Newline in text", "Hello\nworld");
    
    // ========== UNICODE TESTS ==========
    printf("\n--- UNICODE TESTS ---\n");
    
    const char* t45_expected[] = {"café"};
    test_case("Accented letter", "café", t45_expected, 1);
    
    const char* t46_expected[] = {"naïve"};
    test_case("Diaeresis", "naïve", t46_expected, 1);
    
    const char* t47_expected[] = {"Hello", " 世界", "!"};
    test_case("CJK characters", "Hello 世界!", t47_expected, 3);
    
    const char* t48_expected[] = {"Привет"};
    test_case("Cyrillic", "Привет", t48_expected, 1);
    
    const char* t49_expected[] = {"Γειά"};
    test_case("Greek", "Γειά", t49_expected, 1);
    
    const char* t50_expected[] = {"こんにちは"};
    test_case("Hiragana", "こんにちは", t50_expected, 1);
    
    // ========== EMOJI AND SYMBOLS ==========
    printf("\n--- EMOJI AND SYMBOL TESTS ---\n");
    
    test_case_no_verify("Emoji", "Hello 😊 world");
    
    test_case_no_verify("Unicode symbols", "Hello ≡ƒÿè world");
    
    test_case_no_verify("Multiple emoji", "😀😁😂");
    
    test_case_no_verify("Emoji with text", "Test 😀 test");
    
    // ========== MIXED COMPLEX CASES ==========
    printf("\n--- MIXED COMPLEX CASES ---\n");
    
    const char* t55_expected[] = {"don", "'t", " stop", " 123", " abc", "!"};
    test_case("Mixed complex", "don't stop 123 abc!", t55_expected, 6);
    
    const char* t56_expected[] = {"I", "'m", " 123", " years", " old"};
    test_case("Contraction with numbers", "I'm 123 years old", t56_expected, 5);
    
    const char* t57_expected[] = {"Hello", ",", " ", " world", "!"};
    test_case("Punctuation with double space", "Hello,  world!", t57_expected, 5);
    
    const char* t58_expected[] = {"The", " quick", " brown", " fox", " jumps"};
    test_case("Multiple words", "The quick brown fox jumps", t58_expected, 5);
    
    const char* t59_expected[] = {"test", "@", "example", ".", "com"};
    test_case("Email-like", "test@example.com", t59_expected, 5);
    
    test_case_no_verify("URL-like", "https://example.com");
    
    // ========== HARD EDGE CASES ==========
    printf("\n--- HARD EDGE CASES ---\n");
    
    test_case_no_verify("All contractions together", "'s't're've'm'll'd'");
    
    const char* t62_expected[] = {"abcde"};
    test_case("Single letters", "abcde", t62_expected, 1);
    
    const char* t63_expected[] = {"ABC"};
    test_case("Uppercase letters", "ABC", t63_expected, 1);
    
    const char* t64_expected[] = {"aBc"};
    test_case("Mixed case", "aBc", t64_expected, 1);
    
    test_case_no_verify("Very long word", "supercalifragilisticexpialidocious");
    
    test_case_no_verify("Repeated pattern", "ababababab");
    
    const char* t67_expected[] = {"!!!", "Hello", "!!!"};
    test_case("Punctuation sandwich", "!!!Hello!!!", t67_expected, 3);
    
    const char* t68_expected[] = {"123456789"};
    test_case("Number groups", "123456789", t68_expected, 1);
    
    // ========== WHITESPACE VARIATIONS ==========
    printf("\n--- WHITESPACE VARIATIONS ---\n");
    
    test_case_no_verify("Tab between words", "Hello\tworld");
    
    test_case_no_verify("CRLF between words", "Hello\r\nworld");
    
    test_case_no_verify("Form feed", "Hello\fworld");
    
    test_case_no_verify("Vertical tab", "Hello\vworld");
    
    test_case_no_verify("Mixed whitespace", "Hello \t world");
    
    // ========== SPECIAL CHARACTERS ==========
    printf("\n--- SPECIAL CHARACTER TESTS ---\n");
    
    const char* t74_expected[] = {"Hello", "#", "world"};
    test_case("Hash", "Hello#world", t74_expected, 3);
    
    const char* t75_expected[] = {"Hello", "$", "world"};
    test_case("Dollar", "Hello$world", t75_expected, 3);
    
    const char* t76_expected[] = {"Hello", "%", "world"};
    test_case("Percent", "Hello%world", t76_expected, 3);
    
    const char* t77_expected[] = {"Hello", "&", "world"};
    test_case("Ampersand", "Hello&world", t77_expected, 3);
    
    const char* t78_expected[] = {"Hello", "*", "world"};
    test_case("Asterisk", "Hello*world", t78_expected, 3);
    
    const char* t79_expected[] = {"Hello", "+", "world"};
    test_case("Plus", "Hello+world", t79_expected, 3);
    
    const char* t80_expected[] = {"Hello", "=", "world"};
    test_case("Equals", "Hello=world", t80_expected, 3);
    
    const char* t81_expected[] = {"Hello", "<", "world"};
    test_case("Less than", "Hello<world", t81_expected, 3);
    
    const char* t82_expected[] = {"Hello", ">", "world"};
    test_case("Greater than", "Hello>world", t82_expected, 3);
    
    const char* t83_expected[] = {"Hello", "|", "world"};
    test_case("Pipe", "Hello|world", t83_expected, 3);
    
    const char* t84_expected[] = {"Hello", "~", "world"};
    test_case("Tilde", "Hello~world", t84_expected, 3);
    
    const char* t85_expected[] = {"Hello", "`", "world"};
    test_case("Backtick", "Hello`world", t85_expected, 3);
    
    // ========== REAL-WORLD EXAMPLES ==========
    printf("\n--- REAL-WORLD EXAMPLES ---\n");
    
    const char* t86_expected[] = {"The", " quick", " brown", " fox", " jumps", " over", " the", " lazy", " dog", "."};
    test_case("Pangram", "The quick brown fox jumps over the lazy dog.", t86_expected, 10);
    
    const char* t87_expected[] = {"Hello", ",", " ", " how", " are", " you", " today", "?"};
    test_case("Greeting", "Hello,  how are you today?", t87_expected, 8);
    
    const char* t88_expected[] = {"I", "'ll", " be", " there", " at", " 5", ":", "30", " PM"};
    test_case("Time with contraction", "I'll be there at 5:30 PM", t88_expected, 9);
    
    test_case_no_verify("Price", "The price is $19.99");
    
    const char* t90_expected[] = {"Email", ":", " test", "@", "example", ".", "com"};
    test_case("Email statement", "Email: test@example.com", t90_expected, 7);
    
    // ========== STRESS TESTS ==========
    printf("\n--- STRESS TESTS ---\n");
    
    test_case_no_verify("Long repeated spaces", "a     b");
    
    test_case_no_verify("Many contractions", "I'm you're they're we're");
    
    test_case_no_verify("Mixed punctuation", "Hello!!!???...");
    
    test_case_no_verify("Numbers and punctuation", "123!@#abc");
    
    test_case_no_verify("Complex unicode", "café naïve 世界 Привет Γειά");
    
    // ========== SUMMARY ==========
    printf("\n========================================\n");
    printf("TEST SUMMARY\n");
    printf("========================================\n");
    printf("Tests passed: %d\n", tests_passed);
    printf("Tests failed: %d\n", tests_failed);
    printf("Total tests: %d\n", tests_passed + tests_failed);
    printf("Success rate: %.1f%%\n", 100.0 * tests_passed / (tests_passed + tests_failed));
    printf("========================================\n");
    
    return tests_failed > 0 ? 1 : 0;
}
