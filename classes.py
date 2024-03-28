import sentence_splitter
from bs4 import BeautifulSoup
import ebooklib.epub
import tokenizers
import ollama
import fitz
import json
import os

class Summarizer:
	def __init__(self, backstory_strength):
		self.model_name = 'mistral'
		self.tokenizer = tokenizers.Tokenizer.from_pretrained('mistralai/Mistral-7B-v0.1')
		self.reasonable_text_length = 2000
		self.model_token_limit = 4 * 1024  # anyone who tells you that Mistral can process 8k tokens is lying
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

        Summary format: english, 3-5 sentences, ignoring section titles and backstory
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
		elif input_filepath.endswith('.pdf'):
			with fitz.open(input_filepath) as doc:
				text = ""
				for page in doc:
					text += page.get_text()
			return (text)
		else:
			with open(input_filepath, 'r', encoding='utf-8') as f:
				return f.read()

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

	def split_text(self, text, token_limit):

		def ceil(n):
			return int(-1 * n // 1 * -1)

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

		def find_most_common_separator(sentence):
			# find the most common non-alphanumeric character in this sentence
			all_chars = set(sentence)
			separators = [c for c in all_chars if not c.isalnum()]
			if len(separators) == 0:
				return None
			separator_counts = {c: sentence.count(c) for c in separators}
			most_common_separator = max(separator_counts, key=separator_counts.get)
			return most_common_separator

		def split_sentence(sentence, target_part_count):
			separator = find_most_common_separator(sentence)
			if separator is None:
				# fuck it, we tried. split it into equal parts
				return chunk(sentence, target_part_count)
			sentence_atoms = sentence.split(separator)
			sentence_sections = chunk(sentence_atoms, target_part_count, separator)
			return list(sentence_sections)

		def split_long_sentences(sentences, token_limit):
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

		def find_best_split_point(sentences, token_limit):
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

		def chunk_in_half(sentences):
			if len(sentences) == 1:
				sentences = split_sentence(sentences[0], 2)
			half = len(sentences) // 2
			first_half = ' '.join(sentences[:half])
			second_half = ' '.join(sentences[half:])
			return first_half, second_half

		# TODO: avoid splitting and merging sentences every time

		if len(text) == 0:
			return '', ''

		# Check if the entire text happens to be shorter than the token limit
		text_length = self.measure_tokens(text)
		if text_length < token_limit:
			return text, ''

		# Avoid splitting the text in the middle of a sentence
		sentences = sentence_splitter.split(text)

		# Contingency: if we're not more than twice over the token limit, we should split the text in half
		if text_length < token_limit * 1.9 and text_length > token_limit:
			first_half, second_half = chunk_in_half(sentences)
			return first_half, second_half

		# Run a binary search to find the split point that fits the maximum amount of text into the token limit
		sentence_split_point = find_best_split_point(sentences, token_limit)

		# check if we failed to split the text
		if sentence_split_point == 0 or sentence_split_point == len(sentences):
			# Contingency: if a sentence is above the token limit (rare but it can happen), we split it in equal parts
			sentences = split_long_sentences(sentences, token_limit)
			# try splitting again
			sentence_split_point = find_best_split_point(sentences, token_limit)

		# recombine the two sentence lists
		remaining_part = ''
		first_part = ' '.join(sentences[:sentence_split_point])
		if sentence_split_point < len(sentences):
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
			text_chunk, remaining_chunk = self.split_text(context, context_token_limit, from_the_end=True,
														  contingency=False)
			discard_ratio = len(remaining_chunk) / len(context)
			print(f'Context limit exceeded, omitting the first {discard_ratio:.0%} of context...')
			return text_chunk

		# strong backstory means that we condense the previous condensed context even further
		input_context = context
		condensed_context = ''
		sub_round_letter = 'a'
		# this needs to be done every time a new text chunk comes in, so it takes more time overall
		while True:
			# process the text in chunks small enough to fit into the LLM's context window
			print(f'\nSub-round {sub_round_letter}...')
			condensate, remaining_context = self.condense_text(input_context, condensed_context)
			condensed_context += condensate

			# check if we can wrap up this round because the entire input text has been condensed
			if len(remaining_context) == 0:
				break

			# not done? aw. prepare for the next round
			input_context = remaining_context
			sub_round_letter = chr(ord(sub_round_letter) + 1)

		print('...finished.')
		return condensed_context

	def condense_text(self, input_text, backstory=''):
		# check if the invariate parts of the prompt exceed our token limit
		if len(backstory) > 0:
			summary_prompt = self.summary_template.format(backstory=backstory)
			summary_length = self.measure_tokens(summary_prompt)
		else:
			summary_prompt = ''
			summary_length = 0
		template_length = self.measure_tokens(self.prompt_template)
		if summary_length + template_length > self.model_token_limit:
			raise ValueError(
				'This text is too long to summarize, at least with the default network. Try using one with a longer context window!')

		# try to squeeze as much input text as close as possible into the prompt
		chunk_limit = self.model_token_limit - (summary_length + template_length)
		text_chunk, remaining_chunk = self.split_text(input_text, chunk_limit)
		main_prompt = self.prompt_template.format(full_text=text_chunk)
		prompt = summary_prompt + main_prompt
		prompt_length = self.measure_tokens(prompt)

		# ask an LLM to condense the full text for us
		text_chunk_length = self.measure_tokens(text_chunk)
		overhead_ratio = (prompt_length - text_chunk_length) / self.model_token_limit
		print(f'Processing {text_chunk_length:d} tokens of text plus {overhead_ratio:.0%} overhead...')
		# print(f'Sending {prompt_length:d} tokens to the LLM...')
		output = ollama.generate(model=self.model_name, prompt=prompt, options=self.ollama_options)

		# gather the LLM output
		response = output['response']
		response_length = self.measure_tokens(response)
		# ollama api defines total_duration as omitted if empty, defaulting to a value of 1.
		processing_time = (output['total_duration'] / 1000000000) if 'total_duration' in output else 1
		if len(text_chunk) is not None:
			remaining_text_length = self.measure_tokens(remaining_chunk)
			processing_speed = (prompt_length + response_length) / processing_time
			print(f'...finished in {processing_time:.0f} seconds ({processing_speed:.0f} t/s).')
			# print(f'LLM summary: {response_length:d} tokens')
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
			if len(self.responses) <= run_index:
				self.responses.append([])
			while True:
				# process the text in chunks small enough to fit into the LLM's context window
				print(f'\nRound {run_index + 1:d}-{section_index + 1:d}...')
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
			print(f'...finished round {run_index + 1:d}! Condensed the text by {condensation_factor:.2%}.')

			# write out the raw responses at the end of round 0
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

