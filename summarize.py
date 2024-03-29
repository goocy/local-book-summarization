import argparse
import yaml
import os

from src.summarizer import Summarizer

def write_summary(filepath: str, content: str):
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

def main():
    # Parse command-line arguments
    arg_parser = argparse.ArgumentParser(description='Summarize a large text file using a transformer model')
    arg_parser.add_argument('book_folderpath', help='The path to the folder that contains books')
    args = arg_parser.parse_args()
    book_folderpath = args.book_folderpath

    # Load configuration from YAML file
    with open('config.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)

    # Initialize Summarizer instance
    summarizer = Summarizer(config)

    # run a very small speed test
    summarizer.speed_test()

    # Process each book in the folder
    for filename in os.listdir(book_folderpath):

        # Check if this is a valid book
        base, ext = os.path.splitext(filename)
        if ext not in summarizer.file_loader.valid_extensions:
            continue
        if ext == '.txt':
            if base.endswith('-detailed') or base.endswith('-short'):
                continue
        input_filepath = os.path.join(book_folderpath, filename)

        # Check if summary already exists
        detailed_output_filename = config['output_filenames']['detailed'].format(base=base)
        short_output_filename = config['output_filenames']['short'].format(base=base)
        detailed_output_filepath = os.path.join(book_folderpath, detailed_output_filename)
        short_output_filepath = os.path.join(book_folderpath, short_output_filename)
        if os.path.exists(detailed_output_filepath) or os.path.exists(short_output_filepath):
            print(f'\nSummary already exists for {filename}, skipping...')
            continue

        # Summarize the book
        print(f'\nSummarizing {filename}...')
        detailed_text, short_text = summarizer.summarize(input_filepath)

        # Write the detailed and short summaries to files
        write_summary(detailed_output_filepath, detailed_text)
        write_summary(short_output_filepath, short_text)

if __name__ == '__main__':
    main()