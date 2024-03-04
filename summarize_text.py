from semantic_text_splitter import HuggingFaceTextSplitter
from tokenizers import Tokenizer
from bs4 import BeautifulSoup
import ebooklib.epub
import argparse
import requests
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


def ask_oobabooga(prompt, context_limit):
    api_url = "http://127.0.0.1:5000/v1/completions"  # Adjust as necessary
    headers = {"Content-Type": "application/json"}
    data = {
        "prompt": prompt,
        "max_tokens": context_limit,  # Adjust based on your needs
        "use_samplers": False
    }
    response = requests.post(api_url, headers=headers, json=data)
    if response.status_code == 200:
        summary = response.json()["choices"][0]["text"]
        return summary
    else:
        return "Error: Unable to summarize text."

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
Itemized analysis of this text fragment (in english):
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
arg_parser.add_argument('--context-limit', type=int, help='The maximum token size for the currently loaded model')
arg_parser.add_argument('--model-name', type=str, help='The name of the model to use for summarization')
args = arg_parser.parse_args()
input_filename = args.input_filename
context_limit = args.context_limit
model_name = args.model_name
base = input_filename.split('.')[0]
detailed_output_filename = f'{base}-detailed.txt'
short_output_filename = f'{base}-short.txt'

# split the book into chunks
input_text = extract_text_from_epub(input_filename)
tokenizer = Tokenizer.from_pretrained(model_name)
splitter = HuggingFaceTextSplitter(tokenizer, trim_chunks=False)
text_chunks = splitter.chunks(input_text, context_limit)

# summarize chunks iteratively
run_index = 0
while len(text_chunks) > 1:
    text_chunk_length = sum([len(chunk) for chunk in text_chunks])
    print(f'Round {run_index + 1} of text summarization, trying to condense a text of length {text_chunk_length}...')
    summarized_text = ''
    backstory_prompt = ''
    for chunk_index, text_chunk in tqdm.tqdm(enumerate(text_chunks), total=len(text_chunks)):
        if len(summarized_text) > 0:
            backstory_prompt = backstory_template.format(backstory=summarized_text)
        prompt_template = backstory_prompt + condense_template
        prompt = prompt_template.format(fulltext=text_chunk)
        print(f"Sending a prompt of length {len(prompt):d} to the model for summarization...")
        output = ask_oobabooga(prompt, context_limit)
        print(f"...received a response of length {len(output):d} from the model.")
        summarized_text += f'---\n\n{output}\n\n'
    if run_index == 0:
        with open(detailed_output_filename, 'w') as f:
            f.write(summarized_text)
    text_chunks = splitter.chunks(input_text, context_limit)
    run_index += 1

print('Building a synopsis from the findings...')
prompt = synopsis_template.format(plotpoints=summarized_text)
output = ask_oobabooga(prompt, context_limit)

with open(short_output_filename, 'w') as f:
    f.write(output)
print('...finished.')