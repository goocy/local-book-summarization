import sentence_splitter
from .utils import *
import tokenizers


class TokenManager:
    def __init__(self, config):
        tokenizer_name = config['tokenizer_name']
        self.tokenizer = tokenizers.Tokenizer.from_pretrained(tokenizer_name)

    def measure_tokens(self, text):
        tokens = self.tokenizer.encode(text)
        return len(tokens.ids)

    def _split_long_sentences(self, sentences, token_limit):
        # Contingency: if a sentence is above the token limit (rare but it can happen), we split it in equal parts
        potential_long_sentences = False
        for sentence in sentences:
            if len(sentence) > token_limit:  # cheap lower estimate
                potential_long_sentences = True
                break
        if potential_long_sentences:
            for i, sentence in enumerate(sentences):
                sentence_length = self.measure_tokens(sentence)  # more expensive precise measurement
                if sentence_length > token_limit:
                    target_part_count = int(ceil(sentence_length / token_limit))
                    sentence_parts = split_sentence(sentence, target_part_count)
                    sentences.pop(i)
                    sentences.extend(sentence_parts)
        return sentences

    def _find_best_split_point(self, sentences, token_limit):
        low = 0
        high = len(sentences)
        best_split_point = 0
        while low < high:
            mid = (low + high) // 2
            first_part_test = ' '.join(sentences[:mid])
            test_length = self.measure_tokens(first_part_test)
            if test_length < token_limit:
                best_split_point = mid  # this split is valid, might be the best one so far
                low = mid + 1  # look for a split that allows more text in the first part
            else:
                high = mid  # look for a split that puts less text in the first part
        return best_split_point

    def split_text(self, text, token_limit):
        if len(text) == 0:
            return '', ''

        # Check if the entire text happens to be shorter than the token limit
        text_length = self.measure_tokens(text)
        if text_length < token_limit:
            return text, ''

        # Avoid splitting the text in the middle of a sentence
        sentences = sentence_splitter.split(text)

        # Contingency: if we're not more than twice over the token limit, we should split the text in half
        if token_limit * 1.9 > text_length > token_limit:
            first_half, second_half = chunk_in_half(sentences)
            return first_half, second_half

        # Run a binary search to find the split point that fits the maximum amount of text into the token limit
        sentence_split_point = self._find_best_split_point(sentences, token_limit)

        # check if we failed to split the text
        if sentence_split_point == 0 or sentence_split_point == len(sentences):
            # Contingency: if a sentence is above the token limit (rare but it can happen), we split it in equal parts
            sentences = self._split_long_sentences(sentences, token_limit)
            # try splitting again
            sentence_split_point = self._find_best_split_point(sentences, token_limit)

        # recombine the two sentence lists
        remaining_part = ''
        first_part = ' '.join(sentences[:sentence_split_point])
        if sentence_split_point < len(sentences):
            remaining_part = ' '.join(sentences[sentence_split_point:])
        return first_part, remaining_part
