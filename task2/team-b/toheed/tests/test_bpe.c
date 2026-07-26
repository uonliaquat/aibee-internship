

#include <stdio.h>
#include <stdlib.h>

#include "byte_encoder.h"
#include "bpe.h"
#include "tokenizer_loader.h"

static void run_case(FILE *file, const TokenizerData *data,
                      const char *text) {
    
    size_t count = 0;
    int *ids = encode_text(data, text, &count);

    
    for (size_t i = 0; i < count; i++) {
        fprintf(file,"%d%s", ids[i], (i + 1 < count) ? " " : "");

    }
    fprintf(file,"\n");

    free(ids);
}

int main(void) {
    byte_encoder_init();

    TokenizerData *data = tokenizer_init("data/vocab.json", "data/merges.txt");
    if (!data) {
        fprintf(stderr, "test_bpe: failed to load vocab.json / merges.txt\n");
        return 1;
    }

    if (!tokenizer_verify(data)) {
        fprintf(stderr, "test_bpe: tokenizer_verify() failed\n");
        tokenizer_free(data);
        return 1;
    }
    FILE *fptr = fopen("output.txt", "w");
    if (fptr == NULL) {
        printf("Error opening the file!\n");
        tokenizer_free(data);
        return 1;
    }

  /* Zunair's Test Set: 100 Sentences */
    
    /* Group 1: Standard & Conversational English (1-20) */
    run_case(fptr, data, "Hello World");
    run_case(fptr, data, "The quick brown fox jumps over the lazy dog.");
    run_case(fptr, data, "Artificial intelligence is transforming how we process natural language.");
    run_case(fptr, data, "Tokenization is the very first step in training language models.");
    run_case(fptr, data, "Byte-Pair Encoding helps models handle out-of-vocabulary words smoothly.");
    run_case(fptr, data, "Python remains the most popular language for machine learning workflows.");
    run_case(fptr, data, "Hugging Face transformers make downloading pre-trained models effortless.");
    run_case(fptr, data, "OpenAI created tiktoken to make tokenization fast and efficient.");
    run_case(fptr, data, "Debugging code requires patience, logic, and plenty of coffee.");
    run_case(fptr, data, "Always double check your script dependencies before running large batch jobs.");
    run_case(fptr, data, "Data preprocessing takes up most of an engineer's actual work time.");
    run_case(fptr, data, "Large language models process text by converting words into numerical token IDs.");
    run_case(fptr, data, "Deep learning frameworks provide high-level abstractions for complex math.");
    run_case(fptr, data, "A good test dataset should cover normal text alongside severe edge cases.");
    run_case(fptr, data, "Neural networks rely on linear algebra and matrix multiplication under the hood.");
    run_case(fptr, data, "Understanding vocabulary size is crucial for setting embedding layer dimensions.");
    run_case(fptr, data, "Natural language processing has evolved rapidly over the past few years.");
    run_case(fptr, data, "Always validate model outputs against expected ground truth datasets.");
    run_case(fptr, data, "Clean code practices make collaboration much easier across tech teams.");
    run_case(fptr, data, "This sentence concludes the standard conversational English test segment.");

    /* Group 2: Numbers, Math & Financial Data (21-40) */
    run_case(fptr, data, "The answer to life, the universe, and everything is 42.");
    run_case(fptr, data, "In 2026, tech adoption reached an all-time peak of 99.8%.");
    run_case(fptr, data, "Account balance update: $1,234,567.89 transferred successfully.");
    run_case(fptr, data, "Order #84920-X was processed at 03:45:12 UTC on 12/05/2024.");
    run_case(fptr, data, "The value of Pi is approximately 3.14159265359.");
    run_case(fptr, data, "Coordinates logged: 37.7749° N, 122.4194° W.");
    run_case(fptr, data, "Call us at +1 (800) 555-0199 or extension x404.");
    run_case(fptr, data, "Temperature dropped by -15.5 degrees Celsius overnight.");
    run_case(fptr, data, "Fibonacci sequence: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144.");
    run_case(fptr, data, "Error code 0x80070005: Access denied during file execution.");
    run_case(fptr, data, "Invoice #9948 total: €845.50 (VAT included at 20%).");
    run_case(fptr, data, "Model loss decreased from 2.451 to 0.032 across 100 epochs.");
    run_case(fptr, data, "Item discount reduced price from $99.99 to $19.99 (80% OFF!).");
    run_case(fptr, data, "IPv4 address configured: 192.168.1.1 with subnet 255.255.255.0.");
    run_case(fptr, data, "The binary representation of 255 is 11111111.");
    run_case(fptr, data, "Transaction ID: TXN_9876543210_CONFIRMED.");
    run_case(fptr, data, "Flight AA-1042 departs gate B12 at 18:30 GMT.");
    run_case(fptr, data, "Speed limit enforced: 65 MPH (104.6 km/h).");
    run_case(fptr, data, "Serial key: A1B2-C3D4-E5F6-7890.");
    run_case(fptr, data, "Stock price jumped +14.2% following the quarterly earnings report.");

    /* Group 3: Dense Punctuation, Code & Symbols (41-60) */
    run_case(fptr, data, "Wait... is this real?! Are you sure???");
    run_case(fptr, data, "User email address: user.name+tag@subdomain.domain.co.uk");
    run_case(fptr, data, "Math formula: f(x) = (x^2 + 3x - 5) / (2x + 1)");
    run_case(fptr, data, "JSON key-value pair: {\"status\": 200, \"message\": \"OK\"}");
    run_case(fptr, data, "Special characters test: ~!@#$^&*()_+`-={}|[]\\:\";<>?,./");
    run_case(fptr, data, "Python syntax snippet: [x.strip() for x in raw_lines if x]");
    run_case(fptr, data, "SQL Query: SELECT * FROM users WHERE active = 1 AND role = 'admin';");
    run_case(fptr, data, "Terminal command: git commit -m \"Fix critical bug in tokenizer script\"");
    run_case(fptr, data, "HTML tag inside string: <div class=\"container\">Hello world</div>");
    run_case(fptr, data, "Path string: C:\\Users\\Zunair\\AppData\\Local\\Programs\\Python311");
    run_case(fptr, data, "URL with query params: https://example.com/search?q=gpt2+tokenizer&lang=en#results");
    run_case(fptr, data, "Quoted phrase: \"'Double' and 'single' quotes nested together.\"");
    run_case(fptr, data, "Semicolons; Colons: Dashes—Hyphens-Brackets [Parentheses] (Braces) {Angle} <Brackets>");
    run_case(fptr, data, "Is this a question? No, it's a statement! Or maybe... both?");
    run_case(fptr, data, "Math comparison: 5 > 3 and 10 <= 20 but 100 != 200.");
    run_case(fptr, data, "Regex pattern: ^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$");
    run_case(fptr, data, "Markdown syntax: **bold**, *italic*, `code`, and [links](http://example.com)");
    run_case(fptr, data, "Arrow indicators: ---> <-- ===> <=== <==> ==>");
    run_case(fptr, data, "Ellipsis variation: One... Two.... Three..... Done!");
    run_case(fptr, data, "Mixing operators: x += 1; y -= 2; z *= 3; w /= 4; k %= 5;");

    /* Group 4: Weird Spacing, Tabs, Newlines & Formatting (61-80) */
    run_case(fptr, data, "Too    many    spaces    between    these    words.");
    run_case(fptr, data, "    Leading and trailing spaces in this string.    ");
    run_case(fptr, data, "Sentence with a tab\tcharacter separating words.");
    run_case(fptr, data, "Multiple\ttabs\t\tin\t\t\ta\tsingle\tline.");
    run_case(fptr, data, "Line one.\nLine two.\nLine three inside a single string.");
    run_case(fptr, data, "Carriage returns:\r\nWindows style line endings included.");
    run_case(fptr, data, "Space, space, space   comma , space before punctuation !");
    run_case(fptr, data, "Nospacebetweenwordsatalljustonecontinuousstring.");
    run_case(fptr, data, "A   B   C   D   E   F   G   H   I   J   K   L");
    run_case(fptr, data, "  \t  Mixed tabs and spaces everywhere  \t  ");
    run_case(fptr, data, "Word\n\tIndented with newline and tab together.");
    run_case(fptr, data, "Capitalization test: tHiS iS a MExIeD cAsE sTrInG.");
    run_case(fptr, data, "Single-letter-words-separated-by-hyphens.");
    run_case(fptr, data, "S  P  A  C  E  D     O  U  T     L  E  T  T  E  R  S");
    run_case(fptr, data, "Repeated dots: . . . . . . . . . . . . . . .");
    run_case(fptr, data, "Sentence ending with trailing spaces      ");
    run_case(fptr, data, "Sentence starting with multiple newlines\n\n\nStarting here now.");
    run_case(fptr, data, "Mix of newlines \n and tabs \t and double spaces  together.");
    run_case(fptr, data, "CamelCase, PascalCase, snake_case, and kebab-case identifiers.");
    run_case(fptr, data, "  Off-center centered text alignment experiment  ");

    /* Group 5: Emojis, Unicode & Multilingual Snippets (81-100) */
    run_case(fptr, data, "Good morning! ☀️ Have a wonderful day! 🚀");
    run_case(fptr, data, "Testing emoji sequences: 😂 🔥 💯 🎉 🤖 ✨ 🐍 💻");
    run_case(fptr, data, "Thumbs up 👍, heart ❤️, rocket 🚀, and skull 💀.");
    run_case(fptr, data, "Combined emojis: 👨‍💻 (man technologist) and 👩‍🔬 (woman scientist).");
    run_case(fptr, data, "Emoji math: 🍕 + 🥤 = 😋");
    run_case(fptr, data, "Global greetings: Hello! Bonjour! ¡Hola! Ciao! Hallo! Olá!");
    run_case(fptr, data, "Urdu text test: پاکستان ایک خوبصورت ملک ہے");
    run_case(fptr, data, "Japanese Hiragana & Kanji: こんにちは世界 (Hello World)");
    run_case(fptr, data, "Arabic script: مرحباً بك في اختبار الترميز");
    run_case(fptr, data, "Chinese characters: 深度学习 and 自然语言处理");
    run_case(fptr, data, "Russian Cyrillic: Привет мир, это тестовое предложение.");
    run_case(fptr, data, "Hindi script: नमस्ते दुनिया! आप कैसे हैं?");
    run_case(fptr, data, "Greek text: Η γνώση είναι δύναμη.");
    run_case(fptr, data, "Currency symbols from around the world: $ USD, € EUR, £ GBP, ¥ JPY, ₨ PKR, ₹ INR.");
    run_case(fptr, data, "Accented European characters: François, Müller, Peña, Björn, Łukasz, Beyoncé.");
    run_case(fptr, data, "Zalgo / Glitch text test: H̵e̶l̷l̸o̴ ̶W̵o̷r̶l̷d̸");
    run_case(fptr, data, "Flags: 🇵🇰 🇺🇸 🇬🇧 🇯🇵 🇩🇪 🇫🇷 🇨🇦 🇦🇺");
    run_case(fptr, data, "Multi-emoji story: 🌧️ ➡️ 🌈 ➡️ ☀️ = 😄");
    run_case(fptr, data, "Unicode shapes: ★ ☆ ♠ ♣ ♥ ♦ ⬛ ⬜ 🔴 🔵 🔺 🔻");
    run_case(fptr, data, "Zunair's 100th test sentence complete! 🎉 💯 🚀");

    fclose(fptr);
    tokenizer_free(data);
    return 0;
}