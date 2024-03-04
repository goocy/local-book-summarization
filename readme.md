# Setup

## Oobabooga

- install and run text-generation-webui (https://github.com/oobabooga/text-generation-webui)

- once running, open http://127.0.0.1:7860/ in a browser and the visit the sessions tab

- enable the checkmark *openai* to enable remote controlled generation

- click the button "apply flags/extensions and restart"

- once the restart has finished, reload the browser tab

- pick a language model on huggingface.co (I recommend Mistral-7B, or Llama-2-7B-32k)

- in text-generation-webui, visit the "model" tab and enter the identifier for the model you picked

- click Download

- don't message me if any of this fails, I can barely use text-generation-webui myself


## Python

- install Python 3.10

- make a new virtual environment

``` python -m venv venv ```

- activate virtual environment

    Windows: ``` venv\scripts\activate ```

    Linux: you already know this better than I do

- install dependencies

``` pip install -r requirements.txt ```


# Run

- Run text-generation-webui with the starter script ("start-windows.bat" or similar)

- once running, open http://127.0.0.1:7860/ in a browser and the visit the *models* tab

- select the model you downloaded in the upper left dropdown menu

- assign most of your free GPU memory, a few gigabytes of CPU memory for safety

- if you have them, enable use_flash_attention_2 and load_in_8_bit in the model settings

- click the "Load" button (top left of the page)

- open a second terminal window and navigate to the folder containing the project

``` python summarize_text.py <your_book.epub> ```

The output should look like this:
 


# Core idea

So you have a book or several million and you don't live long enough to read them all but maybe you kinda want to know what happens in one of them. Or you're listening to an audiobook and want to see if it's safe to skip ahead or you keep losing the plot and want to fill up the gaps with a little cheat sheet.
This is what this project is for. Point to a .txt file, and it spits out a summary of the plot. One detailed, one short.