from collections import Counter

class Tokenizer:
    def __init__(self, max_vocab_size=20000):
        self.max_vocab_size = max_vocab_size
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}

    def build_vocab(self, texts):
        words = []
        for text in texts:
            words.extend(text.split())

        counts = Counter(words).most_common(self.max_vocab_size)
        for idx, (word, _) in enumerate(counts, start=2):
            self.word2idx[word] = idx

    def encode(self, text):
        return [self.word2idx.get(w, 1) for w in text.split()]
