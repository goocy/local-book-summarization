import ollama
from .token_manager import TokenizedText

class LLMConnector:
    def __init__(self, config):
        self.model_name = config['model_name']
        self.model_token_limit = config['model_token_limit']
        self.ollama_options = config['ollama_options']
        self.processing_speed = None

    def _process_prompt(self, prompt):
        output = ollama.generate(model=self.model_name, prompt=prompt, options=self.ollama_options)
        response = output['response']
        read_time = output['prompt_eval_duration'] / 1000000000 if 'prompt_eval_duration' in output else -1
        write_time = output['eval_duration'] / 1000000000 if 'eval_duration' in output else -1
        return response, read_time, write_time

    def condense_text(self, input_chunk: TokenizedText, backstory='', token_manager=None, prompt_builder=None, speed_test=False):
        if token_manager is None or prompt_builder is None:
            raise ValueError("token_manager and prompt_builder must be provided.")

        # Check if the invariate parts of the prompt exceed the token limit
        summary_prompt = prompt_builder.build_summary_prompt(backstory)
        summary_length = token_manager.measure_tokens(summary_prompt)
        template_length = token_manager.measure_tokens(prompt_builder.prompt_template)
        if summary_length + template_length > self.model_token_limit:
            raise ValueError(
                'This text is too long to summarize, at least with the default network. '
                'Try using one with a longer context window!'
            )

        # Try to squeeze as much input text as close as possible into the prompt
        chunk_limit = self.model_token_limit - (summary_length + template_length)
        section_chunk, remaining_chunk = token_manager.split_text(input_chunk, chunk_limit)
        section_text = section_chunk.extract_sentence_slice()
        prompt = prompt_builder.build_prompt(section_text, backstory)
        prompt_length = token_manager.measure_tokens(prompt)

        # Ask the LLM to condense the full text
        section_chunk_length = token_manager.measure_tokens(section_text)
        overhead_ratio = (prompt_length - section_chunk_length) / self.model_token_limit
        if not speed_test:
            print(f'Processing {section_chunk_length:d} tokens of text plus {overhead_ratio:.0%} overhead...')
        response, read_time, write_time = self._process_prompt(prompt)

        if response is None:
            raise ValueError("Failed to generate a response from the LLM.")

        # Gather the LLM output
        response_length = token_manager.measure_tokens(response)
        if read_time > 0 and write_time > 0:
            if speed_test: # this is a better speed estimate when using very small prompts
                processing_speed = prompt_length / read_time
            else:
                processing_time = read_time + write_time
                processing_speed = (prompt_length + response_length) / processing_time
                print(f'...done in {processing_time:.0f} seconds ({processing_speed:.0f} t/s).')
            self.processing_speed = processing_speed

        else:
            print('...done.')
        if len(remaining_chunk) > 0:
            remaining_text_length = remaining_chunk.measure_token_length()
            print(f'Remaining input text: {remaining_text_length:d} tokens')

        return response, remaining_chunk