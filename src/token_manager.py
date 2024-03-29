import tokenizers
from src import utils
from src.tokenized_text import TokenizedText

class TokenManager:
    def __init__(self, config):
        tokenizer_name = config['tokenizer_name']
        self.tokenizer = tokenizers.Tokenizer.from_pretrained(tokenizer_name)

    def measure_tokens(self, text):
        return len(self.tokenizer.encode(text).ids)

    def _split_long_sentences(self, tokenized_text, token_limit):
        # Contingency: if a sentence is above the token limit (rare but it can happen), we split it into equal parts
        new_sentences = []
        new_token_lists = []
        for idx, (sentence, token_list) in enumerate(tokenized_text):
            if len(token_list) > token_limit:
                separator = utils.find_most_common_separator(sentence)
                split_count = utils.ceil(len(token_list) / token_limit)
                sentence_parts = utils.split_sentence(sentence, split_count, separator)
                for new_sentence in sentence_parts:
                    new_token_list = self.tokenizer.encode(new_sentence).ids
                    new_sentences.append(new_sentence)
                    new_token_lists.append(new_token_list)
            else:
                new_sentences.append(sentence)
                new_token_lists.append(token_list)
        tokenized_text = TokenizedText('', None, sentences=new_sentences, token_lists=new_token_lists)
        return tokenized_text

    def _find_best_split_point(self, tokenized_text, token_limit):
        low = 0
        high = len(tokenized_text)
        best_split_point = 0
        while low < high:
            mid = (low + high) // 2
            first_part_tokens = tokenized_text.slice_tokens(0, mid)
            test_length = sum(len(tokens) for tokens in first_part_tokens)
            if test_length < token_limit:
                best_split_point = mid  # this split is valid, might be the best one so far
                low = mid + 1  # look for a split that allows more text in the first part
            else:
                high = mid  # look for a split that puts less text in the first part
        return best_split_point

    def split_text(self, tokenized_text, token_limit):
        # Check if the incoming text is empty
        empty_tokenized_text = TokenizedText()
        if tokenized_text.measure_token_length() == 0:
            return tokenized_text, empty_tokenized_text

        # Check if the entire text happens to be shorter than the token limit
        token_count = tokenized_text.measure_token_length()
        if token_count < token_limit:
            return tokenized_text, empty_tokenized_text

        # Contingency: if we're not more than twice over the token limit, we should split the text in half
        sentence_count = len(tokenized_text)
        if token_limit * 1.9 > token_count > token_limit:
            first_half, second_half = tokenized_text.split_at(sentence_count // 2)
            return first_half, second_half

        # Run a binary search to find the split point that fits the maximum amount of text into the token limit
        split_point = self._find_best_split_point(tokenized_text, token_limit)

        # check if we failed to split the text
        if split_point == 0 or split_point == sentence_count:
            # Contingency: if a sentence is above the token limit (rare but it can happen), we split it in equal parts
            tokenized_text = self._split_long_sentences(tokenized_text, token_limit)
            # try splitting again
            split_point = self._find_best_split_point(tokenized_text, token_limit)

        # split the tokenized text
        return tokenized_text.split_at(split_point)
