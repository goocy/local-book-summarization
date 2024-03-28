
def measure_tokens(text, tokenizer):
    tokens = tokenizer.encode(text)
    return len(tokens.ids)

def find_most_common_separator(text):
    # find the most common non-alphanumeric character in this sentence
    all_chars = set(text)
    separators = [c for c in all_chars if not c.isalnum()]
    if len(separators) == 0:
        return None
    separator_counts = {c: text.count(c) for c in separators}
    most_common_separator = max(separator_counts, key=separator_counts.get)
    return most_common_separator

def split_sentence(sentence, target_part_count, separator=None):
    if separator is None:
        separator = find_most_common_separator(sentence)
    if separator is None:
        # fuck it, we tried. split it into equal parts
        return list(chunk(sentence, target_part_count))
    sentence_atoms = sentence.split(separator)
    sentence_sections = chunk(sentence_atoms, target_part_count, separator)
    return list(sentence_sections)

def chunk(in_string, num_chunks, separator=''):
    # https://stackoverflow.com/questions/22571259/split-a-string-into-n-equal-parts
    chunk_size = len(in_string) // num_chunks
    if len(in_string) % num_chunks: chunk_size += 1
    iterator = iter(in_string)
    for _ in range(num_chunks):
        accumulator = list()
        for _ in range(chunk_size):
            try:
                accumulator.append(next(iterator))
            except StopIteration:
                break
        yield separator.join(accumulator)

def chunk_in_half(sentences):
    if len(sentences) == 1:
        sentences = split_sentence(sentences[0], 2)
    half = len(sentences) // 2
    first_half = ' '.join(sentences[:half])
    second_half = ' '.join(sentences[half:])
    return first_half, second_half

def ceil(n):
    return int(-1 * n // 1 * -1)