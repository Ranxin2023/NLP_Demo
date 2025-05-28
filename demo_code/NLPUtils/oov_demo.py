from transformers import BertTokenizer
from nltk.corpus import wordnet as wn
from contextlib import redirect_stdout
def character_level_tokenize(text):
    return list(text)

def character_level_models():
    print("caracter leve models")
    oov_word = "xenobot"
    print("Character-level tokens:", character_level_tokenize(oov_word))

def subword_tokenization():
    print("subword tokenization")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokens = tokenizer.tokenize("xenobot is a new AI-driven lifeform.")
    print("Subword tokens:", tokens)

def unknown_token():
    print("unknown token......")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokens = tokenizer.tokenize("This sentence includes blorptastic.")  # fake word

    # Find where [UNK] appears
    ids = tokenizer.convert_tokens_to_ids(tokens)
    unk_token = tokenizer.unk_token
    unk_id = tokenizer.convert_tokens_to_ids([unk_token])[0]

    print("Tokens:", tokens)
    print("Contains [UNK]?", unk_id in ids)

def lookup_word(word):
    synsets = wn.synsets(word)
    if synsets:
        return synsets[0].definition()
    else:
        return "OOV word. No definition found."
    
def external_knowledge():
    print("external knowledge......")
    print("Meaning of 'neuralink':", lookup_word("neuralink"))
    print("Meaning of 'run':", lookup_word("run"))
# Pseudocode / Simulated fine-tuning loop
from transformers import BertTokenizer, BertForMaskedLM
import torch

def fine_tuning():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForMaskedLM.from_pretrained("bert-base-uncased")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForMaskedLM.from_pretrained("bert-base-uncased")

    # Fine-tune on a new sentence that includes OOV term
    sentence = "Neuralink is developing a brain-computer interface."
    inputs = tokenizer(sentence, return_tensors="pt")
    labels = inputs.input_ids.clone()

    # Simulate training step
    outputs = model(**inputs, labels=labels)
    loss = outputs.loss
    print("Simulated fine-tuning loss:", loss.item())

def oov_demo():
    with open("./output_results/oov_demo.txt", "w", encoding="utf-8") as f:
        with redirect_stdout(f):
            character_level_models()
            external_knowledge()
            fine_tuning()
