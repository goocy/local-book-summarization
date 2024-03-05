from semantic_text_splitter import CharacterTextSplitter
from bs4 import BeautifulSoup
import ebooklib.epub
import argparse
import ollama
import tqdm


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


# prompt crafting
backstory_template = """ 
--- backstory plot start ---
{backstory}
--- backstory plot end ---
"""
condense_template = """
--- text fragment start ---
{fulltext}
--- text fragment end ---

Analysis items: most relevant plot events, time and locations, overall mood
Short itemized analysis of this text fragment (in english):
-
"""
synopsis_template = """
--- Itemized list of plot points start ---
{plotpoints}
--- Itemized list of plot points end ---

Summary fields: 1) One-line summary, 2) Setting (time and location), 3) Brief character sheet, 4) Mood, 5) Extended plot synopsis
Summary of plot points:
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
model_token_limit = 8*1024
ollama_options = {'temperature': 0, 'num_ctx': model_token_limit, 'num_predict': 512, 'top_k': 20}
# more settings here https://github.com/ollama/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values

# split the book into chunks
chunk_size = int(model_token_limit * 1.6)
input_text = extract_text_from_epub(input_filename)
splitter = CharacterTextSplitter(trim_chunks=True)
text_chunks = splitter.chunks(input_text, chunk_size)

# summarize chunks iteratively
run_index = 0
while len(text_chunks) > 1:
    text_chunk_length = sum([len(chunk) for chunk in text_chunks])
    print(f'Round {run_index + 1} of text summarization, trying to condense a text of length {text_chunk_length:d}...')
    summarized_text = ''
    backstory_prompt = ''
    for chunk_index, text_chunk in enumerate(text_chunks):
        print(f'{chunk_index+1}/{len(text_chunks)}')
        if len(summarized_text) > 0:
            backstory_prompt = backstory_template.format(backstory=summarized_text)
        prompt_template = backstory_prompt + condense_template
        prompt = prompt_template.format(fulltext=text_chunk)
        output = ollama.generate(model=model_name, prompt=prompt, options=ollama_options)
        response = output['response']
        print(response)
        summarized_text += f'---\n\n{response}\n\n'
    if run_index == 0:
        with open(detailed_output_filename, 'w', encoding='utf-8') as f:
            f.write(summarized_text)
    print(f'...condensed the text to a length of {len(summarized_text):d}.')
    text_chunks = splitter.chunks(summarized_text, chunk_size)
    run_index += 1

print('Building a synopsis from the findings...')
prompt = synopsis_template.format(plotpoints=summarized_text)
output = output = ollama.generate(model=model_name, prompt=prompt, options=ollama_options)
response = output['response']

with open(short_output_filename, 'w', encoding='utf-8') as f:
    f.write(response)