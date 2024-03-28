from bs4 import BeautifulSoup
import ebooklib.epub
import fitz
import json
import os

class FileLoader:
    def __init__(self, config):
        self.valid_extensions = ['.epub', '.json', '.pdf', '.txt']

    def load(self, file_path):
        _, ext = os.path.splitext(file_path)
        if ext not in self.valid_extensions:
            raise ValueError(f"Unsupported file format: {ext}")

        if ext == '.epub':
            return self._load_epub(file_path)
        elif ext == '.json':
            return self._load_json(file_path)
        elif ext == '.pdf':
            return self._load_pdf(file_path)
        elif ext == '.txt':
            return self._load_text(file_path)

    def _load_epub(self, file_path):
        book = ebooklib.epub.read_epub(file_path)
        items = list(book.get_items_of_type(ebooklib.ITEM_DOCUMENT))
        text = ""
        for item in items:
            chapter = item.get_body_content().decode('utf-8')
            soup = BeautifulSoup(chapter, 'html.parser')
            item_text = soup.get_text()
            text += item_text
        return text

    def _load_json(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        messages = [f"{m['author']['name']}: {m['content']}" for m in data['messages']]
        return "\n".join(messages)

    def _load_pdf(self, file_path):
        with fitz.open(file_path) as doc:
            text = ""
            for page in doc:
                text += page.get_text()
        return text

    def _load_text(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
