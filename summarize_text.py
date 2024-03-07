import sentence_splitter

from bs4 import BeautifulSoup
import ebooklib.epub
import tokenizers
import argparse
import ollama


def extract_text_from_epub(epub_path):
    book = ebooklib.epub.read_epub(epub_path)
    items = list(book.get_items_of_type(ebooklib.ITEM_DOCUMENT))
    full_text = ''
    for item in items:
        chapter = item.get_body_content()
        soup = BeautifulSoup(chapter, 'html.parser')
        section = ''
        for element in soup.find_all('p'):
            paragraph = element.get_text()
            section += paragraph + '\n'
        full_text += section
    return full_text


def measure_tokens(text, tokenizer):
    tokens = tokenizer.encode(text)
    return len(tokens.ids)


def extract_next_chunk(text, token_limit, tokenizer):
    # model_identifier refers to last part of the Huggingface URL, for example "NousResearch/Yarn-Llama-2-7b-128k"
    # no this can not be done in a deterministic way, because the amount of backstory grows with each iteration
    trim_whitespace = True

    if len(text) == 0:
        return '', None

    sentences = sentence_splitter.split(text)

    # test if we can fit the entire text within one chunk
    if measure_tokens(text, tokenizer) < token_limit:
        return text, None

    # simulate filling the text chunk with sentences until the token limit is reached
    chunk_length = 0
    remainder_length = len(text)
    combined_token_length = 0
    for sentence_index, sentence in enumerate(sentences):
        # check if this new sentence would exceed the token limit
        token_length = measure_tokens(sentence, tokenizer)
        if combined_token_length + token_length > token_limit:
            break

        # update the running tallies
        combined_token_length += token_length
        chunk_length += len(sentence)
        remainder_length -= len(sentence)

        # contingency case: if the remaining text threatens to become shorter than the current chunk, stop early
        if chunk_length > remainder_length:
            break

    # assemble the text chunk and the remainder
    text_chunk = ' '.join(sentences[:sentence_index])
    text_remainder = ' '.join(sentences[sentence_index:])

    return text_chunk, text_remainder


def remove_duplicate_lines(text):
    if len(text) == 0:
        return text
    lines = text.split('\n')
    duplicate_groups = []
    for i, current_line in enumerate(lines):
        if len(current_line) < 20: # skip short lines
            continue
        if current_line.endswith(':'): # skip headers
            continue

        # look for duplicates
        duplicate_group = set()
        for j, potential_duplicate in enumerate(lines):
            if i == j:
                continue
            if len(potential_duplicate) < 20:
                continue
            if potential_duplicate.endswith(':'): # skip headers
                continue
            if current_line == potential_duplicate:
                duplicate_group.add(j)
        if len(duplicate_group) > 0:
            duplicate_groups.append(duplicate_group)

    # go through the duplicate groups and select all but the first occurrence
    delete_indices = set()
    for group in duplicate_groups:
        group = list(group)
        group.sort()
        for group_index in group[1:]:
            delete_indices.add(group_index)

    # delete selected duplicate lines
    delete_indices = list(delete_indices)
    delete_indices.sort(reverse=True)
    for i in delete_indices:
        del lines[i]

    delete_indices = []
    # select any empty headers
    for i, line in enumerate(lines[:-1]):
        if line.endswith(':'):
            next_line = lines[i+1]
            if len(next_line) == 0 or next_line.endswith(':'):
                delete_indices.append(i)

    # select any empty section markers
    last_section_marker = 0
    section_is_empty = True
    for i, line in enumerate(lines):
        if line.startswith('---') and line.endswith('---'):
            if section_is_empty:
                last_section_range = range(last_section_marker, i)
                delete_indices += list(last_section_range)
            last_section_marker = i
            section_is_empty = True
        else:
            if line.isalnum():
                section_is_empty = False

    # actually delete the selected lines
    delete_indices.sort(reverse=True)
    for i in delete_indices:
        del lines[i]

    # assemble the remaining lines into a single text
    text = '\n'.join(lines)
    return text


# prompt crafting
prompt_template = """

--- Start full text of part {section_index:d} ---
{full_text}
--- End full text of part {section_index:d} ---

Summary focus: most relevant plot events, time and locations, overall mood
Summary format: english, 3-5 sentences, part numbers are irrelevant
Summary of part {section_index:d}:

"""

# filename handling
arg_parser = argparse.ArgumentParser(description='Summarize a text file using a transformer model')
arg_parser.add_argument('input_filename', help='The name of the input file to summarize')
args = arg_parser.parse_args()
input_filename = args.input_filename
base = input_filename.split('.')[0]
detailed_output_filename = f'{base}-detailed.txt'
short_output_filename = f'{base}-short.txt'
model_name = 'mistral'
model_identifier = 'mistralai/Mistral-7B-v0.1'
model_token_limit = 4*1024
reasonable_text_length = 2000
ollama_options = {'temperature': 0.0, 'num_ctx': model_token_limit, 'num_predict': 512, 'top_k': 20, 'top_p': 0.5}
# more settings here https://github.com/ollama/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values

# split the book into chunks
input_text = extract_text_from_epub(input_filename)
tokenizer = tokenizers.Tokenizer.from_pretrained(model_identifier)
input_token_length = measure_tokens(input_text, tokenizer)

# summarize chunks iteratively
run_index = 1
while True: # run iterative condensation rounds until the output is short enough
    input_text_length = len(input_text)
    print(f'Trying to condense {input_text_length:d} characters ({input_token_length:d} tokens)...')
    section_index = 1
    responses = []
    condensed_text = ''
    while True: # process the incoming text in chunks
        # calculate the number of input sentences we can fit into this round's token limit
        condensed_token_length = measure_tokens(condensed_text, tokenizer)
        if condensed_token_length > model_token_limit:
            print('This book is too long to summarize with this model. Sorry.')
            exit()
        placeholder_main_prompt = prompt_template.format(section_index=section_index, full_text='')
        placeholder_main_token_length = measure_tokens(placeholder_main_prompt, tokenizer)
        token_limit = model_token_limit - (condensed_token_length + placeholder_main_token_length)
        text_chunk, text_remainder = extract_next_chunk(input_text, token_limit, tokenizer)

        # assemble the prompt
        main_prompt = prompt_template.format(section_index=section_index, full_text=text_chunk)
        prompt = condensed_text + main_prompt

        # condense the new text chunk by running it through ollama
        prompt_token_length = measure_tokens(prompt, tokenizer)
        print(f'\nRound {run_index:d}-{section_index:d}: running a prompt with {prompt_token_length} tokens through {model_name}...')
        output = ollama.generate(model=model_name, prompt=prompt, options=ollama_options)
        response = output['response']
        processing_time = output['total_duration'] / 1000000000
        responses.append(response)
        print(f'...processed {len(text_chunk):d} characters in {processing_time:.0f} seconds.')

        # assemble the responses into a condensed text block
        condensed_text = ''
        for response_index, response in enumerate(responses):
            condensed_text += f'---\n\n{response}\n\n'
        #condensed_text = remove_duplicate_lines(condensed_text)

        # check if there's still unprocessed text left
        if text_remainder is None:
            break

        # yup, there is still unprocessed text, so prepare for the next iteration
        print(f'Remaining text length: {len(text_remainder):d} characters')
        input_text = text_remainder
        section_index += 1

    # write a detailed summary file at the end of the first round
    if run_index == 1:
        with open(detailed_output_filename, 'w', encoding='utf-8') as f:
            f.write(condensed_text)
            print(f'Wrote a detailed summary to {detailed_output_filename}.')

    condensation_factor = 1 - len(condensed_text) / input_text_length
    print(f'\nRound {run_index:d} finished! Condensed the text by {condensation_factor:.0%}.')

    # check if the condensed output has a reasonable length
    if len(condensed_text) < reasonable_text_length:
        break

    # output is still too long, let's run one more round
    input_text = condensed_text
    run_index += 1

with open(short_output_filename, 'w', encoding='utf-8') as f:
    f.write(condensed_text)
print(f'Processing complete! Wrote a brief summary to {short_output_filename}.')