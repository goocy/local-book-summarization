# Installation

## LLM stuff

- install ollama

- download your model of choice (I added support for mistral, mistral-dolphin, and phi)

``` ollama pull mistral ```

- don't use *-instruct* models, they suck for this


## Python stuff

- install Python 3.10

- make a new virtual environment

``` python -m venv venv ```

- activate virtual environment

    Windows: ``` venv\scripts\activate ```

    Linux: you already know this better than I do

- install dependencies

``` pip install -r requirements.txt ```


# Run

``` python summarize_text.py your_text_file.txt ```

if it fails to find a module, I forgot to put it into requirements.txt, please install manually and ping me

if you don't have a text file, do this to convert your document:

```pip install tika-python ```

``` tika-python parse text your_document.pdf ```
 
- rename your_document.pdf_meta.json to your_document.txt

# Core idea

So you have a book or several million and you don't live long enough to read them all but maybe you kinda want to know what happens in one of them. Or you're listening to an audiobook and want to see if it's safe to skip ahead or you keep losing the plot and want to fill up the gaps with a little cheat sheet.
This is what this project is for. Point to a .txt file, and it spits out a summary of the plot. One detailed, one short.

The thing is, most consumer LLMs nowadays have a context window between 2k and 8k tokens, which is ridiculously low compared to the length of an entire book (easily 100k tokens). I'm splitting up the book into bite sized chunks and asking the LLM to summarize each chunk into plot points. 
Those plot points then get injected into the start of every new text chunk, so the AI doesn't have to guess wildly what happened before. And at the end I'm taking all the summarized plot points and running an iterative chain of summaries over *that*. Buuuut I'm kinda relying on the sliding context window in Mistral to let older plot points gradually dissolve into the virtual oblivion of embedded space. Which means that this approach will just straight-up break anything that doesn't handle context windows as gracefully as Mistral. Would not recommend to try with gemma, for example.

I *would* recommend trying a model with a much longer context window, like LongLlama and its glorious 256k tokens. But that's not in Ollama yet, so I can't.