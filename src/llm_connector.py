import ollama

class LLMConnector:
    def __init__(self, config):
        self.model_name = config['model_name']
        self.model_token_limit = config['model_token_limit']
        self.ollama_options = config['ollama_options']

    def _process_prompt(self, prompt):
        output = ollama.generate(model=self.model_name, prompt=prompt, options=self.ollama_options)
        response = output['response']
        processing_time = output['total_duration'] / 1000000000 if 'total_duration' in output else None
        return response, processing_time

    def condense_text(self, input_text, backstory='', token_manager=None, prompt_builder=None):
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
        text_chunk, remaining_chunk = token_manager.split_text(input_text, chunk_limit)
        prompt = prompt_builder.build_prompt(text_chunk, backstory)
        prompt_length = token_manager.measure_tokens(prompt)

        # Ask the LLM to condense the full text
        text_chunk_length = token_manager.measure_tokens(text_chunk)
        overhead_ratio = (prompt_length - text_chunk_length) / self.model_token_limit
        print(f'Processing {text_chunk_length:d} tokens of text plus {overhead_ratio:.0%} overhead...')
        response, processing_time = self._process_prompt(prompt)

        if response is None:
            raise ValueError("Failed to generate a response from the LLM.")

        # Gather the LLM output
        response_length = token_manager.measure_tokens(response)
        if processing_time is not None:
            processing_speed = (prompt_length + response_length) / processing_time
            print(f'...done in {processing_time:.0f} seconds ({processing_speed:.0f} t/s).')
        else:
            print('...done.')
        if len(remaining_chunk) > 0:
            remaining_text_length = token_manager.measure_tokens(remaining_chunk)
            print(f'Remaining input text: {remaining_text_length:d} tokens')

        return response, remaining_chunk