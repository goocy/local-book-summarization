import pickle

import sentence_splitter
from bs4 import BeautifulSoup
import ebooklib.epub
import tokenizers
import argparse
import ollama
import json
import os

class Summarizer:
    def __init__(self, backstory_strength):
        self.model_name = 'mistral'
        self.tokenizer = tokenizers.Tokenizer.from_pretrained('mistralai/Mistral-7B-v0.1')
        self.reasonable_text_length = 2000
        self.model_token_limit = 4*1024 # anyone who tells you that Mistral can process 8k tokens is lying
        self.context_ratio = 0.5  # context is allowed to take up this percentage of the context window space before we start compressing context
        self.backstory_strength = backstory_strength
        self.ollama_options = {
            'num_ctx': self.model_token_limit,
            'temperature': 0.0,
            'num_predict': 512,
            'top_k': 20,
            'top_p': 0.5
        }
        self.summary_template = """
        --- Start of backstory ---
        {backstory}
        --- End of backstory ---

        """
        self.prompt_template = """
        --- Start of text section ---
        {full_text}
        --- End of text section ---

        Summary format: english, 3-5 sentences
        Summary of text section:
        
        """

    def load(self, input_filepath):
        filepath_parts = os.path.split(input_filepath)
        root_path = filepath_parts[0]
        base, ext = os.path.splitext(filepath_parts[-1])
        detailed_filename = f'{base}-detailed.txt'
        short_filename = f'{base}-short.txt'
        self.detailed_output_filepath = os.path.join(root_path, detailed_filename)
        self.short_output_filepath = os.path.join(root_path, short_filename)
        if input_filepath.endswith('.epub'):
            return self.load_epub(input_filepath)
        elif input_filepath.endswith('.json'):
            return self.load_json(input_filepath)
        else:
            return self.load_txt(input_filepath)

    def load_epub(self, epub_path):
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

    def load_txt(self, txt_path):
        with open(txt_path, 'r', encoding='utf-8') as f:
            return f.read()

    def load_json(self, json_path):
        # this is specific to the Discord chatlog format
        content = None
        with open(json_path, 'r', encoding='utf-8') as f:
            content = json.load(f)
        if content is None:
            raise ValueError('The JSON file could not be read.')
        messages = ''
        for message in content['messages']:
            author = message['author']['name']
            content = message['content']
            messages += f'{author}: {content}\n'
        return messages

    def measure_tokens(self, text):
        tokens = self.tokenizer.encode(text)
        return len(tokens.ids)

    def split_text(self, text, token_limit, from_the_end=False, contingency=True):
        if len(text) == 0:
            return '', ''

        # Check if the entire text happens to be shorter than the token limit
        text_length = self.measure_tokens(text)
        if text_length < token_limit:
            return text, ''

        # Avoid splitting the text in the middle of a sentence
        sentences = sentence_splitter.split(text)
        if from_the_end:
            sentences.reverse()

        # Contingency: if we're not more than twice over the token limit, we should split the text in half
        if contingency:
            if text_length < token_limit * 2:
                token_limit = text_length // 2

        # Run a binary search to find the split point that fits the maximum amount of text into the token limit
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
        sentence_split_point = best_split_point

        if from_the_end:
            sentences.reverse()
            last_part = ' '.join(sentences[-sentence_split_point:])
            if sentence_split_point == len(sentences):
                remaining_part = ''
            else:
                remaining_part = ' '.join(sentences[:-sentence_split_point])
            return last_part, remaining_part

        first_part = ' '.join(sentences[:sentence_split_point])
        if sentence_split_point == len(sentences):
            remaining_part = ''
        else:
            remaining_part = ' '.join(sentences[sentence_split_point:])
        return first_part, remaining_part

    def create_context(self, round_index, section_index):
        responses = self.responses[round_index]
        context = '\n\n'.join(responses)

        # check if we take up too much room in the context window
        context_token_size = self.measure_tokens(context)
        context_token_limit = self.model_token_limit * self.context_ratio
        if context_token_size < context_token_limit:
            return context

        # weak backstory means that we forget the first part of the context once our context space is exceeded
        if self.backstory_strength == 'weak' and section_index is not None:
            text_chunk, remaining_chunk = self.split_text(context, context_token_limit, from_the_end=True, contingency=False)
            discard_ratio = len(remaining_chunk) / len(context)
            print(f'Context limit exceeded, omitting the first {discard_ratio:.0%} of context...')
            return text_chunk

        # strong backstory means that we condense the previous condensed context even further
        input_context = context
        condensed_context = ''
        input_context_length = self.measure_tokens(input_context)
        print(f'\nCondensing {input_context_length:d} tokens of context before continuing...')
        # this needs to be done every time a new text chunk comes in, so it takes more time overall
        while True:
            # process the text in chunks small enough to fit into the LLM's context window
            condensate, remaining_context = self.condense_text(input_context, condensed_context)
            condensed_context += condensate

            # check if we can wrap up this round because the entire input text has been condensed
            if len(remaining_context) == 0:
                break

            # not done? aw. prepare for the next round
            input_context = remaining_context

        print('...finished.')
        return condensed_context

    def condense_text(self, input_text, condensed_text):
        # check if the invariate parts of the prompt exceed our token limit
        if len(condensed_text) > 0:
            summary_prompt = self.summary_template.format(backstory=condensed_text)
            summary_length = self.measure_tokens(summary_prompt)
        else:
            summary_prompt = ''
            summary_length = 0
        template_length = self.measure_tokens(self.prompt_template)
        if summary_length + template_length > self.model_token_limit:
            raise ValueError('This text is too long to summarize, at least with the default network. Try using one with a longer context window!')

        # try to squeeze as much input text as close as possible into the prompt
        chunk_limit = self.model_token_limit - (summary_length + template_length)
        text_chunk, remaining_chunk = self.split_text(input_text, chunk_limit)
        main_prompt = self.prompt_template.format(full_text=text_chunk)
        prompt = summary_prompt + main_prompt
        prompt_length = self.measure_tokens(prompt)

        # ask an LLM to condense the full text for us
        text_chunk_length = self.measure_tokens(text_chunk)
        overhead_ratio = (prompt_length - text_chunk_length) / self.model_token_limit
        print(f'New prompt: {text_chunk_length:d} tokens of text plus {overhead_ratio:.0%} overhead')
        print(f'Sending {prompt_length:d} tokens to the LLM...')
        output = ollama.generate(model=self.model_name, prompt=prompt, options=self.ollama_options)

        # gather the LLM output
        response = output['response']
        response_length = self.measure_tokens(response)
        processing_time = output['total_duration'] / 1000000000
        if len(text_chunk) is not None:
            remaining_text_length = self.measure_tokens(remaining_chunk)
            processing_speed = (prompt_length + response_length) / processing_time
            print(f'...finished in {processing_time:.0f} seconds ({processing_speed:.0f} t/s).')
            print(f'LLM summary: {response_length:d} tokens')
            print(f'Remaining input text: {remaining_text_length:d} tokens')
        return response, remaining_chunk

    def summarize(self, input_text):
        input_text_length = self.measure_tokens(input_text)
        print(f'Trying to condense a text with a length of {input_text_length:d} tokens...')

        run_index = 0
        self.responses = []
        while True:
            # run the compression stage, replacing the input text with the condensed text, until the output is short enough
            condensed_text = ''
            section_index = 0
            self.responses.append([])
            while True:
                # process the text in chunks small enough to fit into the LLM's context window
                print(f'\nRound {run_index+1:d}-{section_index+1:d}...')
                response, remaining_chunk = self.condense_text(input_text, condensed_text)

                # store the results
                self.responses[run_index].append(response)

                # check if we can wrap up this round because the entire input text has been condensed
                if len(remaining_chunk) == 0:
                    break

                # not done? aw. prepare for the next round
                section_index += 1
                if self.backstory_strength in ('weak', 'strong'):
                    condensed_text = self.create_context(run_index, section_index)
                else:
                    condensed_text = ''
                input_text = remaining_chunk

            # collect all the condensed text from this round
            summary_text = self.create_context(run_index, None)
            condensation_factor = 1 - self.measure_tokens(summary_text) / input_text_length
            print(f'...finished round {run_index+1:d}! Condensed the text by {condensation_factor:.2%}.')

            # write out the raw responses at the start of round 0
            if run_index == 0:
                with open(self.detailed_output_filepath, 'w', encoding='utf-8') as f:
                    merged_responses = '\n\n'.join(self.responses[0])
                    f.write(merged_responses)
                print(f'Raw responses written to {self.detailed_output_filepath}.')

            # if the summary has a reasonable length, we output it and wrap up
            if len(summary_text) < self.reasonable_text_length:
                with open(self.short_output_filepath, 'w', encoding='utf-8') as f:
                    f.write(summary_text)
                print(f'Processing complete!')
                break

            # nope not short enough. Try again, this time summarizing the condensed text
            run_index += 1
            input_text = condensed_text

        return condensed_text


def main():
    arg_parser = argparse.ArgumentParser(description='Summarize a large text file using a transformer model')
    arg_parser.add_argument('book_folderpath', help='The path to the folder that contains books')
    arg_parser.add_argument('--backstory_strength', default='strong', help='How strongly the the previous context should be tracked (none/weak/strong)')
    args = arg_parser.parse_args()
    book_folderpath = args.book_folderpath
    backstory_strength = args.backstory_strength
    valid_extensions = '.epub', '.txt', '.json'

    book_summarizer = Summarizer(backstory_strength)
    for filename in os.listdir(book_folderpath):
        base, ext = os.path.splitext(filename)
        if ext not in valid_extensions:
            continue
        if ext == '.txt':
            if base.endswith('-detailed') or base.endswith('-short'):
                continue
        input_filepath = os.path.join(book_folderpath, filename)
        input_text = book_summarizer.load(input_filepath)
        if os.path.exists(book_summarizer.detailed_output_filepath) or os.path.exists(book_summarizer.short_output_filepath):
            print(f'Summary already exists for {filename}')
            continue
        print(f'Summarizing {filename}...')
        condensed_text = book_summarizer.summarize(input_text)

if __name__ == '__main__':
    main()