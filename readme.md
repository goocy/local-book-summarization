# Installation

## LLM stuff

install ollama

download your model of choice (I added support for mistral, gemma, mistral-dolphin, gemma:2b, and phi)
> ollama pull mistral
don't use instruct models, they suck for this


## Python stuff

install Python 3.10

make a new virtual environment
> python -m venv venv

activate virtual environment
Windows: > venv\scripts\activate
Linux: > source venv/scripts/activate (I think)

install dependencies
> pip install -r requirements.txt


# Run

> python summarizeTextSplit.py your_text_file.txt

if it fails to find a module, I forgot to put it into requirements.txt, please install manually and ping me

if you don't have a text file, do this to convert your document:
> pip install tika-python
> tika-python parse text your_document.pdf
rename your_document.pdf_meta.json to your_document.txt