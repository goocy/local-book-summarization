import sentence_splitter
from bs4 import BeautifulSoup
import ebooklib.epub
import tokenizers
import argparse
import ollama


class BookSummarizer:
    def __init__(self, input_filename):
        base = input_filename.split('.')[0]
        self.detailed_output_filename = f'{base}-detailed.txt'
        self.short_output_filename = f'{base}-short.txt'
        self.model_name = 'mistral'
        self.tokenizer = tokenizers.Tokenizer.from_pretrained('mistralai/Mistral-7B-v0.1')
        self.reasonable_text_length = 2000
        self.model_token_limit = 4*1024 # whoever tells you that Mistral
        self.context_ratio = 0.5  # context is allowed to take up this much context window space
        self.ollama_options = {
            'num_ctx': self.model_token_limit,
            'temperature': 0.0,
            'num_predict': 512,
            'top_k': 20,
            'top_p': 0.5
        }
        self.prompt_template = """
        --- Start of backstory ---
        {backstory}
        --- End of backstory ---

        --- Start new section ---
        {full_text}
        --- End new section ---

        Summary focus: most relevant plot events, time and locations, overall mood
        Summary format: english, 3-5 sentences, chapter numbers are irrelevant
        Summary of new section:

        """

    def extract_text_from_epub(self, epub_path):
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

    def measure_tokens(self, text):
        tokens = self.tokenizer.encode(text)
        return len(tokens.ids)

    def calculate_text_split(self, text, token_limit):
        if len(text) == 0:
            return '', None

        # Check if the entire text is short enough to summarize it in one go
        if self.measure_tokens(text) < token_limit:
            return None

        # Avoid splitting the text in the middle of a sentence
        sentences = sentence_splitter.split(text)

        # Do a virtual run through all sentences to see how many we can fit into the token limit
        chunk_length = 0
        remainder_length = len(text)
        combined_token_length = 0
        for sentence_index, sentence in enumerate(sentences):
            token_length = self.measure_tokens(sentence)
            # If the next sentence would exceed the token limit, that means that the current sentence_index is the limit
            if combined_token_length + token_length > token_limit:
                break
            # update the running tallies
            combined_token_length += token_length
            chunk_length += len(sentence)
            remainder_length -= len(sentence)
            # Contingency case: before the remainder gets too short, we'd rather not try to fill the token limit
            if chunk_length > remainder_length:
                break

        # convert the sentence index into a character index
        text_index = sum(len(sentence) for sentence in sentences[:sentence_index])
        return text_index

    def create_context(self, round_index, section_index):
        # if section_index is None, we concatenate all the responses associated with this round
        if section_index is None:
            responses = self.responses[round_index]
            context = ''.join(responses)
            return context

        # no? ok this means we're not finished with this round yet
        # but we can still try to do the same thing: concatenate everything we have *so far* from this round
        responses = self.responses[round_index]
        context = ''.join(responses)
        # except this time we need to be careful to not take up too much room in the context window
        context_token_size = self.measure_tokens(context)
        if context_token_size < self.model_token_limit * self.context_ratio:
            # yay the simple approach worked
            return context

        # oof we need to summarize the context before we can continue
        # I know this could be fully recursive but I'm scared to debug a recursive method
        input_context = context
        condensed_context = ''
        print(f'\nCondensing previous context of round {round_index+1:d} before continuing...')
        while True:
            # process the text in chunks small enough to fit into the LLM's context window
            condensate, context_split_index = self.condense_text(input_context, condensed_context)

            # check if we can wrap up this round because the entire input text has been condensed
            if context_split_index is None:
                break

            # not done? aw. prepare for the next round
            condensed_context += condensate
            input_context = input_context[context_split_index:]

        print('...finished.')
        return condensed_context


    def condense_text(self, input_text, condensed_text):
        # measure the structural (invariant) parts of the prompt
        placeholder_prompt = self.prompt_template.format(full_text='', backstory=condensed_text)
        placeholder_token_length = self.measure_tokens(placeholder_prompt)
        if placeholder_token_length > self.model_token_limit:
            raise ValueError('This book is too long to summarize, at least for now. Sorry. Please check back in a few weeks.')

        # try to squeeze as much input text as close as possible into the prompt
        token_limit = self.model_token_limit - placeholder_token_length
        split_index = self.calculate_text_split(input_text, token_limit)
        text_chunk = input_text[:split_index]
        main_prompt = self.prompt_template.format(full_text=text_chunk, backstory=condensed_text)
        prompt = condensed_text + main_prompt

        # ask an LLM to condense the full text for us
        output = ollama.generate(model=self.model_name, prompt=prompt, options=self.ollama_options)

        # gather the LLM output
        response = output['response']
        processing_time = output['total_duration'] / 1000000000
        if split_index is not None:
            print(f'...processed {split_index:d} characters in {processing_time:.0f} seconds.')
            print(f'Remaining text length: {len(input_text)-split_index:d} characters')
            processing_speed = split_index / processing_time
            print(f'{processing_speed:.0f} chars per second')
        return response, split_index

    def summarize_book(self, input_text):
        input_token_length = self.measure_tokens(input_text)
        input_text_length = len(input_text)
        print(f'Trying to condense {input_text_length:d} characters ({input_token_length:d} tokens)...')

        run_index = 0
        self.responses = []
        while True:
            # run the compression stage, replacing the input text with the condensed text, until the output is short enough
            condensed_text = ''
            section_index = 0
            self.responses.append([])
            while True:
                # process the text in chunks small enough to fit into the LLM's context window
                print(f'\nRound {run_index+1:d}-{section_index+1:d} starting...')
                response, text_split_index = self.condense_text(input_text, condensed_text)

                # store the results
                self.responses[run_index].append(response)

                # check if we can wrap up this round because the entire input text has been condensed
                if text_split_index is None:
                    break

                # not done? aw. prepare for the next round
                section_index += 1
                condensed_text = self.create_context(run_index, section_index)
                input_text = input_text[text_split_index:]

            condensation_factor = 1 - len(condensed_text) / input_text_length
            print(f'...finished round {run_index+1:d}! Condensed the text by {condensation_factor:.2%}.')

            # check if our condensed responses are short enough to be considered a summary
            condensed_text = self.create_context(run_index, None)
            if len(condensed_text) < self.reasonable_text_length:
                with open(self.short_output_filename, 'w', encoding='utf-8') as f:
                    f.write(condensed_text)
                print(f'Processing complete! Wrote a brief summary to {self.short_output_filename}.')
                break
            else:
                if run_index == 0:
                    with open(self.detailed_output_filename, 'w', encoding='utf-8') as f:
                        f.write(condensed_text)
                        print(f'Wrote a detailed summary to {self.detailed_output_filename}.')

            # nope not short enough. Try again, this time summarizing the condensed text
            run_index += 1
            input_text = condensed_text

        return condensed_text


def main():
    arg_parser = argparse.ArgumentParser(description='Summarize a text file using a transformer model')
    arg_parser.add_argument('input_filename', help='The name of the input file to summarize')
    args = arg_parser.parse_args()
    input_filename = args.input_filename

    book_summarizer = BookSummarizer(input_filename)
    input_text = book_summarizer.extract_text_from_epub(input_filename)
    condensed_text = book_summarizer.summarize_book(input_text)

if __name__ == '__main__':
    main()