
class PromptBuilder:
    def __init__(self, config):
        self.summary_template = config['summary_template']
        self.prompt_template = config['prompt_template']

    def build_summary_prompt(self, backstory):
        if len(backstory) == 0:
            return ''
        return self.summary_template.format(backstory=backstory)

    def build_main_prompt(self, text_chunk):
        return self.prompt_template.format(full_text=text_chunk)

    def build_prompt(self, text_chunk, backstory=''):
        summary_prompt = ''
        if backstory:
            summary_prompt = self.build_summary_prompt(backstory)

        main_prompt = self.build_main_prompt(text_chunk)
        return summary_prompt + main_prompt
