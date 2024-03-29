from typing import List, Tuple
from .file_loader import FileLoader
from .token_manager import TokenManager
from .tokenized_text import TokenizedText
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

    def _create_weak_context(self, context: TokenizedText) -> TokenizedText:
        context_token_limit = self.config['model_token_limit'] * self.context_ratio
        text_chunk, remaining_chunk = self.token_manager.split_text(context, context_token_limit)
        context_ratio = text_chunk.measure_token_length() / context.measure_token_length()
        print(f'Context limit exceeded, only keeping the last {context_ratio:.0%} of context...')
        return text_chunk.extract_sentence_slice()

    def _create_strong_context(self, input_chunk: TokenizedText) -> str:
        condensed_context = ''
        sub_round_letter = 'a'
        while True:
            sub_round_text = self.round_text + sub_round_letter
            print(f'\n{sub_round_text}...')
            condensed_response, remaining_chunk = self.llm_connector.condense_text(
                input_chunk,
                token_manager=self.token_manager,
                prompt_builder=self.prompt_builder,
                backstory=condensed_context
            )
            condensed_context += condensed_response

            if len(remaining_chunk) == 0:
                break

            input_chunk = remaining_chunk
            sub_round_letter = chr(ord(sub_round_letter) + 1)

        print('...finished the sub-round.')
        return condensed_context

    def create_context(self, responses: List[str]) -> str:
        text = self.response_separator.join(responses)
        context = TokenizedText(text=text, tokenizer=self.token_manager.tokenizer)
        context_token_size = context.measure_token_length()
        context_token_limit = self.config['model_token_limit'] * self.context_ratio

        # check if we can just return all entire previous responses
        if context_token_size < context_token_limit:
            return context.extract_sentence_slice()

        summary = ''
        if self.backstory_strength == 'weak':
            summary = self._create_weak_context(context)
        elif self.backstory_strength == 'strong':
            summary = self._create_strong_context(context)
        return summary

    def _condense_text(self, input_chunk: TokenizedText, responses: List[List[str]], run_index: int, input_text_length: int) -> Tuple[str, str]:
        condensed_text = ''
        section_index = 0
        while True:
            round_text = f'Round {run_index + 1:d}-{section_index + 1:d}'
            print(f'\n{round_text}...')
            self.round_text = round_text
            response_text, remaining_chunk = self.llm_connector.condense_text(
                input_chunk,
                token_manager=self.token_manager,
                prompt_builder=self.prompt_builder,
                backstory=condensed_text
            )

            # save the raw LLM response for later
            responses[run_index].append(response_text)

            # check if we can wrap up this round
            if len(remaining_chunk) == 0:
                break

            # still more to do? aw. prepare for the next round
            section_index += 1
            condensed_text = self.create_context(responses[run_index])
            input_chunk = remaining_chunk

        summary_text = self.create_context(responses[run_index])
        condensation_factor = 1 - self.token_manager.measure_tokens(summary_text) / input_text_length
        print(f'...finished round {run_index + 1:d}! Condensed the text by {condensation_factor:.2%}.')

        return summary_text, responses

    def summarize(self, input_filepath: str) -> Tuple[str, str]:
        input_text = self.file_loader.load(input_filepath)
        tokenized_input_text = TokenizedText(text=input_text, tokenizer=self.token_manager.tokenizer)
        input_text_length = tokenized_input_text.measure_token_length()
        print(f'Trying to condense a text with a length of {input_text_length:d} tokens...')

        run_index = 0
        responses = []
        while True:
            if len(responses) <= run_index:
                responses.append([])
            summary_text, responses = self._condense_text(tokenized_input_text, responses, run_index, input_text_length)

            if len(summary_text) < self.reasonable_text_length:
                print(f'Processing complete!')
                break

            run_index += 1
            tokenized_input_text = TokenizedText(text=summary_text, tokenizer=self.token_manager.tokenizer)

        detailed_text = self.response_separator.join(responses[0])
        return detailed_text, summary_text

    def speed_test(self):
        print('Running a speed test...')
        test_text = 'We do what we must because we can, for the good of all of us, except the ones who are dead.'
        tokenized_test_text = TokenizedText(text=test_text, tokenizer=self.token_manager.tokenizer)
        self.llm_connector.condense_text(
            tokenized_test_text,
            token_manager=self.token_manager,
            prompt_builder=self.prompt_builder,
            speed_test=True
        )
        speed = self.llm_connector.processing_speed
        if speed is not None:
            estimated_time = self.config['model_token_limit'] / speed
            print(f'...estimated time per round: {estimated_time:.0f} seconds.')
