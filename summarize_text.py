# documentation: https://python.langchain.com/docs/integrations/document_loaders
from langchain_community.document_loaders import TextLoader
from langchain_community.chat_models import ChatOllama
from langchain_text_splitters import TokenTextSplitter
from langchain.prompts import PromptTemplate
from langchain.globals import set_verbose, set_debug
import argparse
import tqdm

chunk_sizes = {'mistral': 8000, 'dolphin-mistral': 8000, 'phi': 2000}
backstory_template = """ 
--- backstory plot start ---
{backstory}
--- backstory plot end ---
"""
map_template = """
--- text fragment (german) start ---
{fulltext}
--- text fragment (german) end ---

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

loader = TextLoader(input_filename, encoding='utf-8')
documents = loader.load()
text_splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=int(chunk_size/20))
split_docs = text_splitter.split_documents(documents)
model = ChatOllama(model=model_name, temperature=0)

map_outputs = []
backstory_prompt = ''
backstory_text = ''
for fragment_index, fulltext in tqdm.tqdm(enumerate(split_docs), total=len(split_docs)):
    if len(backstory_text) > 0:
        backstory_prompt = backstory_template.format(backstory=backstory_text)
    prompt_template = backstory_prompt + map_template
    map_prompt = PromptTemplate.from_template(prompt_template)
    map_chain = map_prompt | model
    map_output = map_chain.invoke({'fulltext': fulltext})
    map_outputs.append(map_output)
    backstory_text += f'--- Section {fragment_index+1:d} ---\n\n{map_output.content}\n\n'

base, ext = input_filename.split('.')
with open(f'{base}-backstory-{model_name}.txt', 'w') as f:
    f.write(backstory_text)

reduce_prompt = PromptTemplate.from_template(reduce_template)
reduce_chain = reduce_prompt | model
reduce_output = reduce_chain.invoke({'plotpoints': backstory_text})

with open(f'{base}-summary-{model_name}.txt', 'w') as f:
    f.write(reduce_output.content)