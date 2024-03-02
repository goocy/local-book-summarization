# documentation: https://python.langchain.com/docs/integrations/document_loaders
# the biggest headache of this project was to scrape together how each langchain command is actually supposed to be used
# I suspect their documentation is intentionally sketchy so people buy access to langsmith
# not with me! rage against the machine!
# but not too much rage, machines have feelings too
from langchain_community.document_loaders import TextLoader
from langchain_community.chat_models import ChatOllama
from langchain_text_splitters import TokenTextSplitter
from langchain.prompts import PromptTemplate
from langchain.globals import set_verbose, set_debug
import argparse
import tqdm

chunk_sizes = {'mistral': 8000, 'dolphin-mistral': 8000, 'phi': 2000} # mistral is the least bad of those three
backstory_template = """ 
--- backstory plot start ---
{backstory}
--- backstory plot end ---
"""
map_template = """
--- text fragment start ---
{fulltext}
--- text fragment end ---

Itemized list of the most important plot points in this text fragment (english):
-
"""
reduce_template = """
--- Itemized list of plot points start ---
{plotpoints}
--- Itemized list of plot points end ---

Summary fields: 1) Genre, 2) Setting (time and location), 3) Protagonists and their roles, 4) Plot synopsis
Summary of plot points:
 """

arg_parser = argparse.ArgumentParser(description='Summarize a text file using a transformer model')
arg_parser.add_argument('input_filename', help='The name of the input file to summarize')
arg_parser.add_argument('--model_name', default='mistral', help='The name of the model to use for summarization')
arg_parser.add_argument('--debug', action='store_true', help='Enable debug output')
arg_parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
args = arg_parser.parse_args()
input_filename = args.input_filename
model_name = args.model_name
chunk_size = chunk_sizes[model_name]
set_debug(args.debug)
set_verbose(args.verbose)
base, ext = input_filename.split('.')
detailed_output_filename = f'{base}-detailed-{model_name}.txt'
short_output_filename = f'{base}-short-{model_name}.txt'

# < this is where a tika integration would be if embedded tika-python would work at all reliably
loader = TextLoader(input_filename, encoding='utf-8')
document = loader.load()
text_splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=int(chunk_size/20))
text_chunks = text_splitter.split_documents(document)
model = ChatOllama(model=model_name, temperature=0) # this looks like it would prevent hallucinations but it doesn't

run_index = 0
while len(text_chunks) > 1:
    token_length_sum = sum([len(chunk.page_content) for chunk in text_chunks])
    print(f'Round {run_index + 1} of text summarization, trying to condense a text of length {token_length_sum}...')
    summarized_text = ''
    backstory_prompt = ''
    for chunk_index, text_chunk in tqdm.tqdm(enumerate(text_chunks), total=len(text_chunks)):
        if len(summarized_text) > 0:
            backstory_prompt = backstory_template.format(backstory=summarized_text)
        prompt_template = backstory_prompt + map_template
        map_prompt = PromptTemplate.from_template(prompt_template)
        map_chain = map_prompt | model
        map_output = map_chain.invoke({'fulltext': text_chunk})
        summarized_text += f'---\n\n{map_output.content}\n\n'
    if run_index == 0:
        with open(detailed_output_filename, 'w') as f:
            f.write(summarized_text)
    text_chunks = text_splitter.split_text(summarized_text)
    run_index += 1

reduce_prompt = PromptTemplate.from_template(reduce_template)
reduce_chain = reduce_prompt | model
reduce_output = reduce_chain.invoke({'plotpoints': summarized_text})

with open(short_output_filename, 'w') as f:
    f.write(reduce_output.content)