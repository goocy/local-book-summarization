from typing import List, Tuple
from .file_loader import FileLoader
from .token_manager import TokenManager
from .prompt_builder import PromptBuilder
from .llm_connector import LLMConnector


class Summarizer:
    def __init__(self, config: dict):
        self.config = config
        self.reasonable_text_length = config['reasonable_text_length']
        self.context_ratio = config['context_ratio']
        self.backstory_strength = config['backstory_strength']
        response_separator = config['response_separator']
        if '\\n' in response_separator:
            response_separator = response_separator.replace('\\n', '\n')
        self.response_separator = response_separator

        self.file_loader = FileLoader(config)
        self.token_manager = TokenManager(config)
        self.prompt_builder = PromptBuilder(config)
        self.llm_connector = LLMConnector(config)

    def _create_weak_context(self, context: str) -> str:
        token_manager = TokenManager(self.config['tokenizer_name'])
        context_token_limit = self.config['model_token_limit'] * self.context_ratio
        text_chunk, remaining_chunk = token_manager.split_text(context, context_token_limit)
        discard_ratio = len(remaining_chunk) / len(context)
        print(f'Context limit exceeded, omitting the first {discard_ratio:.0%} of context...')
        return text_chunk

    def _create_strong_context(self, context: str) -> str:
        input_context = context
        condensed_context = ''
        sub_round_letter = 'a'
        while True:
            print(f'\nSub-round {sub_round_letter}...')
            condensate, remaining_context = self.llm_connector.condense_text(
                input_context,
                token_manager=self.token_manager,
                prompt_builder=self.prompt_builder,
                backstory=condensed_context
            )
            condensed_context += condensate

            if len(remaining_context) == 0:
                break

            input_context = remaining_context
            sub_round_letter = chr(ord(sub_round_letter) + 1)

        print('...finished.')
        return condensed_context

    def create_context(self, responses: List[str]) -> str:
        context = self.response_separator.join(responses)
        context_token_size = self.token_manager.measure_tokens(context)
        context_token_limit = self.config['model_token_limit'] * self.context_ratio

        # check if we can just return all entire previous responses
        if context_token_size < context_token_limit:
            return context

        if self.backstory_strength == 'weak':
            return self._create_weak_context(context)
        elif self.backstory_strength == 'strong':
            return self._create_strong_context(context)

    def _condense_text(self, input_text, responses, run_index, input_text_length):
        condensed_text = ''
        section_index = 0
        while True:
            print(f'\nRound {run_index + 1:d}-{section_index + 1:d}...')
            response, remaining_chunk = self.llm_connector.condense_text(
                input_text,
                token_manager=self.token_manager,
                prompt_builder=self.prompt_builder,
                backstory=condensed_text
            )

            # save the raw LLM response for later
            responses[run_index].append(response)

            # check if we can wrap up this round
            if len(remaining_chunk) == 0:
                break

            # still more to do? aw. prepare for the next round
            section_index += 1
            condensed_text = self.create_context(responses[run_index])
            input_text = remaining_chunk

        summary_text = self.create_context(responses[run_index])
        condensation_factor = 1 - self.token_manager.measure_tokens(summary_text) / input_text_length
        print(f'...finished round {run_index + 1:d}! Condensed the text by {condensation_factor:.2%}.')

        return summary_text, responses

    def summarize(self, input_filepath: str) -> Tuple[List[List[str]], str]:
        input_text = self.file_loader.load(input_filepath)
        input_text_length = self.token_manager.measure_tokens(input_text)
        print(f'Trying to condense a text with a length of {input_text_length:d} tokens...')

        run_index = 0
        responses = []
        while True:
            if len(responses) <= run_index:
                responses.append([])
            summary_text, responses = self._condense_text(input_text, responses, run_index, input_text_length)

            if len(summary_text) < self.reasonable_text_length:
                print(f'Processing complete!')
                break

            run_index += 1
            input_text = summary_text

        detailed_text = self.response_separator.join(responses[0])
        return detailed_text, summary_text
