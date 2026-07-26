import json
from transformers import GPT2Tokenizer

sentences = [
    # --- Group 1: Standard & Conversational English (1-20) ---
    "Hello World",
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is transforming how we process natural language.",
    "Tokenization is the very first step in training language models.",
    "Byte-Pair Encoding helps models handle out-of-vocabulary words smoothly.",
    "Python remains the most popular language for machine learning workflows.",
    "Hugging Face transformers make downloading pre-trained models effortless.",
    "OpenAI created tiktoken to make tokenization fast and efficient.",
    "Debugging code requires patience, logic, and plenty of coffee.",
    "Always double check your script dependencies before running large batch jobs.",
    "Data preprocessing takes up most of an engineer's actual work time.",
    "Large language models process text by converting words into numerical token IDs.",
    "Deep learning frameworks provide high-level abstractions for complex math.",
    "A good test dataset should cover normal text alongside severe edge cases.",
    "Neural networks rely on linear algebra and matrix multiplication under the hood.",
    "Understanding vocabulary size is crucial for setting embedding layer dimensions.",
    "Natural language processing has evolved rapidly over the past few years.",
    "Always validate model outputs against expected ground truth datasets.",
    "Clean code practices make collaboration much easier across tech teams.",
    "This sentence concludes the standard conversational English test segment.",

    # --- Group 2: Numbers, Math & Financial Data (21-40) ---
    "The answer to life, the universe, and everything is 42.",
    "In 2026, tech adoption reached an all-time peak of 99.8%.",
    "Account balance update: $1,234,567.89 transferred successfully.",
    "Order #84920-X was processed at 03:45:12 UTC on 12/05/2024.",
    "The value of Pi is approximately 3.14159265359.",
    "Coordinates logged: 37.7749° N, 122.4194° W.",
    "Call us at +1 (800) 555-0199 or extension x404.",
    "Temperature dropped by -15.5 degrees Celsius overnight.",
    #Temperature dropped by -15.5 degrees Celsius overnight.
    "Fibonacci sequence: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144.",
    "Error code 0x80070005: Access denied during file execution.",
    "Invoice #9948 total: €845.50 (VAT included at 20%).",
    "Model loss decreased from 2.451 to 0.032 across 100 epochs.",
    "Item discount reduced price from $99.99 to $19.99 (80% OFF!).",
    "IPv4 address configured: 192.168.1.1 with subnet 255.255.255.0.",
    "The binary representation of 255 is 11111111.",
    "Transaction ID: TXN_9876543210_CONFIRMED.",
    "Flight AA-1042 departs gate B12 at 18:30 GMT.",
    "Speed limit enforced: 65 MPH (104.6 km/h).",
    "Serial key: A1B2-C3D4-E5F6-7890.",
    "Stock price jumped +14.2% following the quarterly earnings report.",

    # --- Group 3: Dense Punctuation, Code & Symbols (41-60) ---
    "Wait... is this real?! Are you sure???",
    "User email address: user.name+tag@subdomain.domain.co.uk",
    "Math formula: f(x) = (x^2 + 3x - 5) / (2x + 1)",
    "JSON key-value pair: {\"status\": 200, \"message\": \"OK\"}",
    "Special characters test: ~!@#$^&*()_+`-={}|[]\\:\";<>?,./", 
    #Special characters test: ~!@#$^&*()_+`-={}|[]\\:\";<>?,./"
    #Special characters test: ~!@#$^&*()_+`-={}|[]\:";<>?,./
    "Python syntax snippet: [x.strip() for x in raw_lines if x]",
    "SQL Query: SELECT * FROM users WHERE active = 1 AND role = 'admin';",
    "Terminal command: git commit -m \"Fix critical bug in tokenizer script\"",
    "HTML tag inside string: <div class=\"container\">Hello world</div>",
    "Path string: C:\\Users\\Zunair\\AppData\\Local\\Programs\\Python311",
    "URL with query params: https://example.com/search?q=gpt2+tokenizer&lang=en#results",
    "Quoted phrase: \"'Double' and 'single' quotes nested together.\"",
    "Semicolons; Colons: Dashes—Hyphens-Brackets [Parentheses] (Braces) {Angle} <Brackets>",
    "Is this a question? No, it's a statement! Or maybe... both?",
    "Math comparison: 5 > 3 and 10 <= 20 but 100 != 200.",
    "Regex pattern: ^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$",
    "Markdown syntax: **bold**, *italic*, `code`, and [links](http://example.com)",
    "Arrow indicators: ---> <-- ===> <=== <==> ==>",
    "Ellipsis variation: One... Two.... Three..... Done!",
    "Mixing operators: x += 1; y -= 2; z *= 3; w /= 4; k %= 5;",

    # --- Group 4: Weird Spacing, Tabs, Newlines & Formatting (61-80) ---
    "Too    many    spaces    between    these    words.",
    "    Leading and trailing spaces in this string.    ",
    "Sentence with a tab\tcharacter separating words.",
    "Multiple\ttabs\t\tin\t\t\ta\tsingle\tline.",
    "Line one.\nLine two.\nLine three inside a single string.",
    "Carriage returns:\r\nWindows style line endings included.",
    "Space, space, space   comma , space before punctuation !",
    "Nospacebetweenwordsatalljustonecontinuousstring.",
    "A   B   C   D   E   F   G   H   I   J   K   L",
    "  \t  Mixed tabs and spaces everywhere  \t  ",
    "Word\n\tIndented with newline and tab together.",
    "Capitalization test: tHiS iS a MExIeD cAsE sTrInG.",
    "Single-letter-words-separated-by-hyphens.",
    "S  P  A  C  E  D     O  U  T     L  E  T  T  E  R  S",
    "Repeated dots: . . . . . . . . . . . . . . .",
    "Sentence ending with trailing spaces      ",
    "Sentence starting with multiple newlines\n\n\nStarting here now.",
    "Mix of newlines \n and tabs \t and double spaces  together.",
    "CamelCase, PascalCase, snake_case, and kebab-case identifiers.",
    "  Off-center centered text alignment experiment  ",

    # --- Group 5: Emojis, Unicode & Multilingual Snippets (81-100) ---
    "Good morning! ☀️ Have a wonderful day! 🚀",
    "Testing emoji sequences: 😂 🔥 💯 🎉 🤖 ✨ 🐍 💻",
    "Thumbs up 👍, heart ❤️, rocket 🚀, and skull 💀.",
    "Combined emojis: 👨‍💻 (man technologist) and 👩‍🔬 (woman scientist).",
    "Emoji math: 🍕 + 🥤 = 😋",
    "Global greetings: Hello! Bonjour! ¡Hola! Ciao! Hallo! Olá!",
    "Urdu text test: پاکستان ایک خوبصورت ملک ہے",
    "Japanese Hiragana & Kanji: こんにちは世界 (Hello World)",
    "Arabic script: مرحباً بك في اختبار الترميز",
    "Chinese characters: 深度学习 and 自然语言处理",
    "Russian Cyrillic: Привет мир, это тестовое предложение.",
    "Hindi script: नमस्ते दुनिया! आप कैसे हैं?",
    "Greek text: Η γνώση είναι δύναμη.",
    "Currency symbols from around the world: $ USD, € EUR, £ GBP, ¥ JPY, ₨ PKR, ₹ INR.",
    "Accented European characters: François, Müller, Peña, Björn, Łukasz, Beyoncé.",
    "Zalgo / Glitch text test: H̵e̶l̷l̸o̴ ̶W̵o̷r̶l̷d̸",
    "Flags: 🇵🇰 🇺🇸 🇬🇧 🇯🇵 🇩🇪 🇫🇷 🇨🇦 🇦🇺",
    "Multi-emoji story: 🌧️ ➡️ 🌈 ➡️ ☀️ = 😄",
    "Unicode shapes: ★ ☆ ♠ ♣ ♥ ♦ ⬛ ⬜ 🔴 🔵 🔺 🔻",
    "Zunair's 100th test sentence complete! 🎉 💯 🚀"
]

print(f"Total sentences created: {len(sentences)}")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenized_data = []


with open("hf_tokens_normalized.txt", "w", encoding="utf-8") as txt_file:
    for idx, sentence in enumerate(sentences):
        # Tokenize
        token_ids = tokenizer.encode(sentence)
        
        # Build JSON entry
        tokenized_data.append({
            "sentence_id": idx + 1,
            "text": sentence,
            "token_ids": token_ids
        })

        token_str = " ".join(map(str, token_ids))
        txt_file.write(token_str + "\n")
with open("gpt2_ids_huggingface.json", "w", encoding="utf-8") as f:
    json.dump(tokenized_data, f, indent=2)    
 