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

Should probably be run on a machine with a decent GPU.

CPU-only does technically work, but it's going to be slow. Like, really slow.

Successfully tested for books up to 360k tokens. Long books will take disproportionally longer to process.


Typical status messages:
```
Summarizing Duel Without End.epub...
Trying to condense a text with a length of 230311 tokens...

Round 1-1...
Processing 3891 tokens of text plus 1% overhead...
done in 6 seconds (647 t/s).
Remaining input text: 226323 tokens

Round 1-2...
Processing 3791 tokens of text plus 5% overhead...
done in 6 seconds (718 t/s).
Remaining input text: 222473 tokens

Round 1-3...
Processing 3670 tokens of text plus 7% overhead...
done in 6 seconds (660 t/s).
Remaining input text: 218716 tokens
```

Typical output (from "Duel Without End-short.txt":
```
The text delves into the historical development of theories on infectious diseases,
featuring key figures like Alexandre Yersin and his discovery of the plague bacterium.
It discusses various ways microbes harm humans, the immune system's role,
and infamous epidemics such as the Black Death and smallpox.
These diseases significantly impacted history, causing societal disruption and
cultural shifts. Modern medical treatments can weaken the immune system,
making individuals more susceptible to infections from opportunistic microbes
like Aspergillus and Candida. The text also explores zoonotic viruses such as
Hendra, Nipah, SARS, MERS, Lassa fever, West Nile virus, and Zika virus,
emphasizing international cooperation for prevention and control.
Historical figures like Edward Jenner, Louis Pasteur, Alexander Fleming,
and Paul Ehrlich are highlighted for their contributions to vaccination and
antibiotic therapy. The text also discusses ongoing challenges in vaccine development
and fair global distribution.
```


# Configuration

If you know what you're doing, feel free to edit config.yaml.

For Discord-formatted JSON files, it's highly recommended to set backstory_strength to "weak".

# Core idea

So you have a book or several million and you don't live long enough to read them all but maybe you kinda want to know what happens in one of them. Or you're listening to an audiobook and want to see if it's safe to skip ahead or you keep losing the plot and want to fill up the gaps with a little cheat sheet.
This is what this project is for. Point it to a folder that contains ebooks, and it spits out a summary of each book. One detailed, one short.