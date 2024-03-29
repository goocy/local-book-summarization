from src import utils
from src import sentence_splitter

class TokenizedText:
    def __init__(self, text='', tokenizer=None, sentences=[], token_lists=[]):
        # Five options to initialize this class:
        # 1. A string-based text and a tokenizer
        # 2. A list of sentences and a tokenizer
        # 3. A string-based text and list of tokens
        # 4. A list of sentences and list of tokens
        # 5. No input (results in a valid empty object)

        self.text = text
        self.tokenizer = tokenizer
        self.sentences = sentences
        self.token_lists = token_lists

        # fill empty fields if necessary
        if len(text) > 0:
            if len(sentences) == 0:
                self.sentences = sentence_splitter.split(text)
            if len(token_lists) == 0:
                assert tokenizer is not None, 'Initialization with text requires a tokenizer.'
                self.token_lists = [self.tokenize(sentence) for sentence in self.sentences]
        if len(sentences) > 0:
            if len(token_lists) == 0:
                assert tokenizer is not None, 'Initialization with sentences requires a tokenizer.'
                self.token_lists = [self.tokenize(sentence) for sentence in sentences]
            if len(text) == 0:
                self.text = ' '.join(sentences)
        assert len(self.token_lists) == len(self.sentences), 'Mismatch between sentences and token lists.'

    def tokenize(self, text):
        return self.tokenizer.encode(text).ids

    def split_sentences_at(self, index):
        first_part = self.sentences[:index]
        remaining_part = self.sentences[index:]
        return first_part, remaining_part

    def split_tokens_at(self, index):
        first_part = self.token_lists[:index]
        remaining_part = self.token_lists[index:]
        return first_part, remaining_part

    def split_at(self, index):
        first_sentences, remaining_sentences = self.split_sentences_at(index)
        first_tokens, remaining_tokens = self.split_tokens_at(index)
        first_part = TokenizedText('', None, sentences=first_sentences, token_lists=first_tokens)
        remaining_part = TokenizedText('', None, sentences=remaining_sentences, token_lists=remaining_tokens)
        return first_part, remaining_part

    def measure_token_length(self):
        return sum(len(tokens) for tokens in self.token_lists)

    def measure_character_length(self):
        return sum(len(sentence) for sentence in self.sentences)

    def slice_tokens(self, start, end):
        return self.token_lists[start:end]

    def slice_sentences(self, start, end):
        return self.sentences[start:end]

    def extract_sentence_slice(self, start=None, end=None):
        if start is None:
            start = 0
        if end is None:
            end = len(self.sentences)
        sentence_selection = self.sentences[start:end]
        return ' '.join(sentence_selection)

    def extract_token_slice(self, start=None, end=None):
        # this requires a re-tokenization of the extracted slice - avoid if possible
        text = self.extract_sentence_slice(start, end)
        return self.tokenize(text)

    def __len__(self):
        return len(self.token_lists)

    def __iter__(self):
        return zip(self.sentences, self.token_lists)