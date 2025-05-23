from transformers import MarianMTModel, MarianTokenizer
from contextlib import redirect_stdout

# ✅ rule based machine transmation（RBMT）：Use English-Chinese dictionary
def rbmt_translate(text):
    dictionary = {
        "hello": "你好",
        "world": "世界",
        "machine": "机器",
        "translation": "翻译",
        "is": "是",
        "awesome": "棒极了"
    }

    words = text.lower().split()
    translated = [dictionary.get(word, word) for word in words]
    return " ".join(translated)

# ✅ based on statistics translation (SMT）：Use phrase matching translation
def smt_translate(text):
    phrase_table = {
        "machine translation": "机器翻译",
        "is awesome": "非常棒",
        "hello": "你好"
    }

    text = text.lower()
    for phrase in sorted(phrase_table, key=len, reverse=True):
        if phrase in text:
            text = text.replace(phrase, phrase_table[phrase])
    return text

# ✅ Based on the neural network use Hugging Face' Marian model to translate from English to Chinese
def nmt_translate(text, src="en", tgt="zh"):
    model_name = f"Helsinki-NLP/opus-mt-{src}-{tgt}"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    encoded = tokenizer([text], return_tensors="pt", padding=True)
    generated = model.generate(**encoded)
    return tokenizer.decode(generated[0], skip_special_tokens=True)

# ✅ output three method's outcome
def translation_demo():
    with open("./output_results/translate_result.txt", "w", encoding="utf-8") as f:
        with redirect_stdout(f):
            print("RBMT:", rbmt_translate("Machine translation is awesome"))
            print("SMT:", smt_translate("Machine translation is awesome"))
            print("NMT (English → Chinese):", nmt_translate("Machine translation is awesome"))
