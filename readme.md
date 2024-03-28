# Installation

## LLM stuff

- install ollama (https://ollama.com/)

- open a command line window and download mistral

```ollama pull mistral```


## Python stuff

- install Python 3.10

- open a command line window, change to the local-book-summary directory

- make a new virtual environment

``` python -m venv venv ```

- activate virtual environment

    Windows: ``` venv\scripts\activate ```

    Linux: you already know this better than I do

- install dependencies

``` pip install -r requirements.txt ```


# Run

``` python summarize.py folder_with_books```

# Configuration

If you know what you're doing, feel free to edit config.yaml.

Especially for Discord-formatted JSON files, it's highly recommended to set backstory_strength to "weak".

# Core idea

So you have a book or several million and you don't live long enough to read them all but maybe you kinda want to know what happens in one of them. Or you're listening to an audiobook and want to see if it's safe to skip ahead or you keep losing the plot and want to fill up the gaps with a little cheat sheet.
This is what this project is for. Point it to a folder that contains ebooks, and it spits out a summary of each book. One detailed, one short.