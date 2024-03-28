import argparse
import classes
import os

def main():
    arg_parser = argparse.ArgumentParser(description='Summarize a large text file using a transformer model')
    arg_parser.add_argument('--book_folderpath', default='books', help='The path to the folder that contains books')
    arg_parser.add_argument('--backstory_strength', default='strong', help='How strongly the the previous context should be tracked (none/weak/strong)')
    args = arg_parser.parse_args()
    book_folderpath = args.book_folderpath
    backstory_strength = args.backstory_strength
    valid_extensions = '.epub', '.txt', '.json', '.pdf'

    book_summarizer = classes.Summarizer(backstory_strength)
    for filename in os.listdir(book_folderpath):
        base, ext = os.path.splitext(filename)
        if ext not in valid_extensions:
            continue
        if ext == '.txt':
            if base.endswith('-detailed') or base.endswith('-short'):
                continue
        input_filepath = os.path.join(book_folderpath, filename)
        input_text = book_summarizer.load(input_filepath)
        if os.path.exists(book_summarizer.detailed_output_filepath) or os.path.exists(book_summarizer.short_output_filepath):
            print(f'Summary already exists for {filename}')
            continue
        print(f'Summarizing {filename}...')
        condensed_text = book_summarizer.summarize(input_text)

if __name__ == '__main__':
    main()