import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 4
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        # YOUR CODE HERE
        all_words = []
        for words in texts:
            for t in words.split():
                all_words.append(t.lower())
        all_words.sort()
        for t in all_words:
            if t not in self.word_to_id: 
                    self.word_to_id[t] = self.vocab_size
                    self.id_to_word[self.vocab_size] = t
                    self.vocab_size += 1
                
        self.word_to_id["<PAD>"] = 0
        self.word_to_id["<UNK>"] = 1
        self.word_to_id["<BOS>"] = 2
        self.word_to_id["<EOS>"] = 3
                
        self.id_to_word[0] = "<PAD>"
        self.id_to_word[1] = "<UNK>"
        self.id_to_word[2] = "<BOS>"
        self.id_to_word[3] = "<EOS>"
                
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        encoding = []
        for t in text.split():
            if t.lower() in self.word_to_id:
                encoding.append(self.word_to_id[t.lower()])
            else:
                encoding.append(1)
        
        return encoding 
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        strs = []
        for i in ids:
            if i in self.id_to_word:
                strs.append(self.id_to_word[i])
            elif i == 0:
                strs.append("<PAD>")
            elif i == 2:
                strs.append("<BOS>")
            elif i == 3:
                strs.append("<EOS>")
            else:
                strs.append("<UNK>")
        return " ".join(strs)
