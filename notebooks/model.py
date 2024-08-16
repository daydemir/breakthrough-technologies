from enum import Enum
from typing import Any

import ollama
from pyairtable import Api

token = 'patphUJScafduJmpI.d2549138065af741d38c396bdc189948b1a18503df821a0cd78765fc1d576dcf'
api = Api(api_key=token)
table = api.table('appOHJEmEMgO9CV6X', 'tblPEUr3QOpPlxIYY')


systemPrompt = 'You are an expert in technology, economics, and social well-being, and are an editor for the MIT Technology Review. You are critical and discerning, with an eye for scientific accuracy. Make no reference to the prompt when providing your responses. Provide your responses directly without commentary.'

summarySystemPrompt = f'''{systemPrompt}
Every prompt will be an article about a technology written by the MIT Technology Review and why that technology will be a breakthrough. Provide a short summary of the article that describes the technology and why the MIT Technology Review picked it as a breakthrough technology in they year they did. Use this format:

"SUMMARY: [technology name] is a technology that does X, Y, and Z. The following industries are relevant to the tehnology: A, B, and C. 
IMPACT: [technology name] could improve human well-being by A, B, or C. It could impact the economy by A, B, or C. The expected timeline for this technology to have a large economic or social impact is 0-5 years / 5-15 years / 15-30 / 30+ years.
AUTHOR: [author name]
OPINION: [technology name] has had/not had the expected impact on social well-being or the economy since the year [year]. The impact has/has not been restricted to research and not widely available. For example, A, B, or C."
            
More info about the above sections:
SUMMARY should be 30 words and describe the technology, its potential uses, and relevant industries. All this should be derived directly from the article.
IMPACT should be 50 words, and about the expected impact of the technology and a specific timeline of expected impact, and any other relevant factors for why this technology is expected to be a breakthrough. This should all be derived from the article.
OPINION should be 50 words, and should be your opinion about whether the technology has had anything like the expected impact on social well-being or the economy since the year [year], provide a specific example or two to justify your claim. Be critical, differentiate between simply impact at a research level and impact on a wide-reaching social or economic level.

Do not provide any commentary or give any introduction to the summary, do not start with anything like "Here is a short summary of the article", just give the summary of the article in the above format. Be terse and direct with your sentences in order to fit more information into the summary. Ensure the end result is less than 150 words.
'''

class Model(Enum):
    phi3 = 'phi3:14b'
    gemma2 = 'gemma2:27b'
    # mistral = 'mistral-large:latest' #too slow
    llama31 = 'llama3.1:70b'
    qwen2 = 'qwen2:72b'
    deepseek = 'deepseek-llm:67b'

    def create(self):
        modelFile = f'''
            FROM {self.value}
            SYSTEM """{summarySystemPrompt}"""
            PARAMETER temperature 0.3
            PARAMETER num_ctx 8192
            PARAMETER seed 42
        '''
        print(modelFile)

        return ollama.create(model=f'MITTR_{self.value}', modelfile=modelFile, stream=False)

    def run(self, prompt: str) -> str:
        return ollama.generate(model=f'MITTR_{self.value}', prompt=prompt)["response"].strip()

class Record:
    def __init__(self, record):
        self.id: str = record['id']
        self.technology = Technology(record['fields'])
        self.createdTime = record['createdTime']

    def __repr__(self):
        return f"{self.fields}"




class Technology: 
    def __init__(self, fields: dict[str, Any]):
        name = fields.get('name')
        if name is not None:
            self.name = fields['name']
            self.year = fields['year']
            # self.link = fields['link']
            # self.specific_link = fields['specific_link']
            self.blurb = fields['blurb']
            self.summary = fields.get('summary')
            self.impact = fields.get('impact')
            self.author = fields.get('author')
            self.opinion = fields.get('opinion')


    
    def summarize(self, model: Model) -> str:
        summary = model.run(f'''Here is the article to summarize about {self.name} from the year {self.year}: "{self.blurb}"''')
        return summary



def summarize(technology: Technology, model: Model) -> tuple[str, str, str, str]:
        extractions = technology.summarize(model)
        summary = ollama.generate(model=Model.llama31.value, prompt=f'Do not add any text of your own, and ignore the "IMPACT:", "AUTHOR:" and "OPINION:" sections. Just return the "SUMMARY:" text verbatim from following text: "{extractions}"')["response"].strip()
        impact = ollama.generate(model=Model.llama31.value, prompt=f'Do not add any text of your own, and ignore the "SUMMARY:", "AUTHOR:" and "OPINION:" sections. Just return the "IMPACT:" text verbatim from following text: "{extractions}"')["response"].strip()
        author = ollama.generate(model=Model.llama31.value, prompt=f'Do not add any text of your own, and ignore the "SUMMARY:", "IMPACT:" and "OPINION:" sections. Just return the "AUTHOR:" text verbatim from following text: "{extractions}"')["response"].strip()
        opinion = ollama.generate(model=Model.llama31.value, prompt=f'Do not add any text of your own, and ignore the "SUMMARY:", "IMPACT:" and "AUTHOR:" sections. Just return the "OPINION:" text verbatim from following text: "{extractions}"')["response"].strip()
        return (summary, impact, author, opinion)

records = list(map(lambda record: Record(record), table.all()))


llama31 = Model.llama31.create()

for record in records:
    tech = record.technology
    if tech.author is None:
        (summary, impact, author, opinion) = summarize(tech, Model.llama31)
        response = table.update(record.id, {'summary': summary, 'impact': impact, 'author': author, 'opinion': opinion})
        print(response)