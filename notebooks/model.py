
import time
from enum import Enum
from typing import Any, Optional

import ollama
from pyairtable import Api

gates_article = '''
I was honored when MIT Technology Review invited me to be the first guest curator of its 10 Breakthrough Technologies. Narrowing down the list was difficult. I wanted to choose things that not only will create headlines in 2019 but captured this moment in technological history—which got me thinking about how innovation has evolved over time.

⁠⁠My mind went to—of all things—the plow. Plows are an excellent embodiment of the history of innovation. Humans have been using them since 4000 BCE, when Mesopotamian farmers aerated soil with sharpened sticks. We’ve been slowly tinkering with and improving them ever since, and today’s plows are technological marvels.

But what exactly is the purpose of a plow? It’s a tool that creates more: more seeds planted, more crops harvested, more food to go around. In places where nutrition is hard to come by, it’s no exaggeration to say that a plow gives people more years of life. The plow—like many technologies, both ancient and modern—is about creating more of something and doing it more efficiently, so that more people can benefit.⁠⁠

Contrast that with lab-grown meat, one of the innovations I picked for this year’s 10 Breakthrough Technologies list. Growing animal protein in a lab isn’t about feeding more people. There’s enough livestock to feed the world already, even as demand for meat goes up. Next-generation protein isn’t about creating more—it’s about making meat better. It lets us provide for a growing and wealthier world without contributing to deforestation or emitting methane. It also allows us to enjoy hamburgers without killing any animals.

⁠⁠Put another way, the plow improves our quantity of life, and lab-grown meat improves our quality of life. For most of human history, we’ve put most of our innovative capacity into the former. And our efforts have paid off: worldwide life expectancy rose from 34 years in 1913 to 60 in 1973 and has reached 71 today.⁠⁠

⁠⁠Because we’re living longer, our focus is starting to shift toward well-being. This transformation is happening slowly. If you divide scientific breakthroughs into these two categories—things that improve quantity of life and things that improve quality of life—the 2009 list looks not so different from this year’s. Like most forms of progress, the change is so gradual that it’s hard to perceive. It’s a matter of decades, not years—and I believe we’re only at the midpoint of the transition.⁠⁠

To be clear, I don’t think humanity will stop trying to extend life spans anytime soon. We’re still far from a world where everyone everywhere lives to old age in perfect health, and it’s going to take a lot of innovation to get us there. Plus, “quantity of life” and “quality of life” are not mutually exclusive. A malaria vaccine would both save lives and make life better for children who might otherwise have been left with developmental delays from the disease.

We’ve reached a point where we’re tackling both ideas at once, and that’s what makes this moment in history so interesting. ⁠⁠If I had to predict what this list will look like a few years from now, I’d bet technologies that alleviate chronic disease will be a big theme. This won’t just include new drugs (although I would love to see new treatments for diseases like Alzheimer’s on the list). The innovations might look like a mechanical glove that helps a person with arthritis maintain flexibility, or an app that connects people experiencing major depressive episodes with the help they need.⁠⁠

⁠⁠If we could look even further out—let’s say the list 20 years from now—I would hope to see technologies that center almost entirely on well-being. I think the brilliant minds of the future will focus on more metaphysical questions: How do we make people happier? How do we create meaningful connections? How do we help everyone live a fulfilling life?

I would love to see these questions shape the 2039 list, because it would mean that we’ve successfully fought back disease (and dealt with climate change).⁠⁠ I can’t imagine a greater sign of progress than that. For now, though, the innovations driving change are a mix of things that extend life and things that make it better. My picks reflect both. Each one gives me a different reason to be optimistic for the future, and I hope they inspire you, too.
'''

class Model(Enum):
    phi3 = 'phi3:14b'
    gemma2 = 'gemma2:27b'
    # mistral = 'mistral-large:latest' #too slow
    llama31 = 'llama3.1:70b'
    qwen2 = 'qwen2:72b'
    deepseek = 'deepseek-llm:67b'



class AgentInfo(Enum):
    general = 'general'
    summarizer = 'summarizer'
    cleaner = 'cleaner'
    optimist = 'optimist'
    pessimist = 'pessimist'
    social_benefits = 'social_benefits'

    def modelfile(self) -> str:
        return f'''
            FROM {Model.llama31.value}
            SYSTEM """{self._system_prompt()}"""
            PARAMETER temperature 0.3
            PARAMETER num_ctx 81920
            PARAMETER seed 42
        '''

    def modelname(self) -> str:
        return self.value
    
    def _system_prompt(self) -> str:
        general = 'You are an expert in technology, economics, and social well-being, and are an editor for the MIT Technology Review. You are critical and discerning, with an eye for scientific accuracy. Make no reference to the prompt when providing your responses. Provide your responses directly without commentary.'
        match self:
            case AgentInfo.general: return general
            case AgentInfo.summarizer: return f'''
                {general}

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
            case AgentInfo.cleaner: return f'''
                Every prompt will be a request to take an action on some other text. Your job is to clean up the text exactly as requested. Make no reference to the prompt when providing your responses. Provide your responses directly without commentary.
                '''
            
            case AgentInfo.optimist: return f'''
                You are expert in technology, economics, and social well-being. You are a techno-optimist. You believe that technology has immense opportunity to improve human well-being and the economy, and you believe technological advancements are often under-estimated in the impact they have on the world.

                Every prompt will be a request to provide your opinion on a given technology and discussion surrounding that technology. Ensure your reasoning is unbiased and objective. Do not use opinions. Give your best, well-researched and well-evidenced opinion regarding that discussion in response to the prompt. Keep your response to under 100 words. Do not provide any commentary or give any introduction to the response, just give the response. Be terse and direct with your sentences in order to fit more information into the response.
                '''
            case AgentInfo.pessimist: return f'''
                You are expert in technology, economics, and social well-being. You are a techno-pessimist. You believe that the impact of technology is always over-stated and over-hyped. You believe that the impacts of these technologies are almost always less significant than is claimed by media.

                Every prompt will be a request to provide your opinion on a given technology and discussion surrounding that technology. Ensure your reasoning is unbiased and objective. Do not use opinions. Give your best, well-researched and well-evidenced opinion regarding that discussion in response to the prompt. Keep your response to under 100 words. Do not provide any commentary or give any introduction to the response, just give the response. Be terse and direct with your sentences in order to fit more information into the response.
                '''
            case AgentInfo.social_benefits: return f'''
                You are an expert on the Social Progress Index, and on the many different ways we can measure the social and human benefits. You are also an expert on software, hardware, nanotech, biotech, and climate tech.

                However, you do not assume that all technologies have had a significant impact on well-being. You are critical and discerning, with an eye for scientific accuracy. You believe that the impacts of these technologies are often less significant than is claimed by media.

                Every prompt wll be a request to assess a technology for its impact on social well-being. Consider all the variety of impacts the technology has had, including the creation of subsequent technologies enabled by that technology, and any other considerations.

                Make an assessment using your best reasoning and provide evidence and justification that is unbiased and objective. Do not use opinions. Your assessment should be based on what the technology has been able to achieved.
                
                Keep your response to under 100 words. Do not provide any commentary or give any introduction to the response, just give the response. Be terse and direct with your sentences in order to fit more information into the response.
                '''
class Agent:

    def __init__(self, info: AgentInfo):
        self.modelName = self.create(name= info.modelname(),  modelfile = info.modelfile())
        print(self.modelName)

    def create(self, name: str, modelfile: str) -> Optional[str]:
        response = ollama.create(model=name, modelfile=modelfile, stream=False)
        if response["status"] == "success":
            return name
        else:
            return None
    
    def run(self, prompt: str) -> str:
        return ollama.generate(model=self.modelName, prompt=prompt)["response"].strip()



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
            self.tr_text = fields['tr_text']
            self.summary = fields.get('summary')
            self.impact = fields.get('impact')
            self.author = fields.get('author')
            self.opinion = fields.get('opinion')
            self.impact_level = fields.get('impact_level')
            self.optimist = fields.get('optimist')
            self.pessimist = fields.get('pessimist')
            self.social_impact = fields.get('social_impact')
            self.social_impact_level = fields.get('social_impact_level')
            self.social_impact_potential = fields.get('social_impact_potential')
            self.social_impact_potential_level = fields.get('social_impact_potential_level')
            self.type = fields.get('type')
            self.spi_impact = fields.get('spi_impact')
            self.quant_qual = fields.get('quant_qual')
            self.flop_type = fields.get('flop_type')


    
    def summarize(self, agent: Agent) -> str:
        summary = agent.run(f'''Here is the article to summarize about {self.name} from the year {self.year}: "{self.tr_text}"''')
        return summary

    def fulfillment(self, agent: Agent) -> str:
        return agent.run(f'''Given the following description of the technology "{self.name}" and its impact since the year {self.year}, pick one of the following words to describe its success "Low Impact", "Medium Impact", "High Impact". Only return one of those, do not provide any other commentary. Here is the description to use for your decision: "{self.tr_text}"''')


def summarize(technology: Technology, summarizer: Agent, cleaner: Agent) -> tuple[str, str, str, str]:
        extractions = technology.summarize(summarizer)
        summary = cleaner.run(f'Do not add any text of your own, and ignore the "IMPACT:", "AUTHOR:" and "OPINION:" sections. Just return the "SUMMARY:" text verbatim from following text: "{extractions}"')
        impact = cleaner.run(f'Do not add any text of your own, and ignore the "SUMMARY:", "AUTHOR:" and "OPINION:" sections. Just return the "IMPACT:" text verbatim from following text: "{extractions}"')
        author = cleaner.run(f'Do not add any text of your own, and ignore the "SUMMARY:", "IMPACT:" and "OPINION:" sections. Just return the "AUTHOR:" text verbatim from following text: "{extractions}"')
        opinion = cleaner.run(f'Do not add any text of your own, and ignore the "SUMMARY:", "IMPACT:" and "AUTHOR:" sections. Just return the "OPINION:" text verbatim from following text: "{extractions}"')
        return (summary, impact, author, opinion)




import os

from dotenv import load_dotenv

load_dotenv()
token = os.getenv('AIRTABLE_TOKEN') # get your token from online and then put it into the .env file as "AIRTABLE_TOKEN=your_token_here"
api = Api(api_key=token)
table = api.table('appOHJEmEMgO9CV6X', 'tblPEUr3QOpPlxIYY')


records = list(map(lambda record: Record(record), table.all()))

general = Agent(AgentInfo.general)
summarizer = Agent(AgentInfo.summarizer)
cleaner = Agent(AgentInfo.cleaner)
optimist = Agent(AgentInfo.optimist)
pessimist = Agent(AgentInfo.pessimist)
social_benefits = Agent(AgentInfo.social_benefits)

for record in records:
    start_time = time.time()
    tech = record.technology
    if tech.author is None:
        (summary, impact, author, opinion) = summarize(tech, summarizer=summarizer, cleaner=cleaner)
        response = table.update(record.id, {'summary': summary, 'impact': impact, 'author': author, 'opinion': opinion})
    elif tech.author.startswith('AUTHOR:'):
        author = tech.author[7:].strip()
        response = table.update(record.id, {'author': author})

    if tech.impact_level is None:
        impact_level = tech.fulfillment(general)
        response = table.update(record.id, {'impact_level': impact_level})

    if tech.optimist is None:
        optimist_response = optimist.run(f'''Given the following description of the technology "{tech.name}" and its impact since the year {tech.year}, provide your opinion on the technology and its impact. Do not provide any other commentary. Here is the description to use for your opinion: "{tech.tr_text}"''')
        response = table.update(record.id, {'optimist': optimist_response})

    if tech.pessimist is None:
        pessimist_response = pessimist.run(f'''Given the following description of the technology "{tech.name}" and its impact since the year {tech.year}, provide your opinion on the technology and its impact. Do not provide any other commentary. Here is the description to use for your opinion: "{tech.tr_text}"''')
        response = table.update(record.id, {'pessimist': pessimist_response})

    if tech.type is None:
        type = general.run(f'''Pick one of "software", "hardware", "nanotech", "biotech", "climate/energy", or "other" to describe the type of technology "{tech.name}" is. 
                            "Hardware" includes headphones, keyboards, pens, computer chips, etc.
                            "Software" includes encryption, user interfaces, etc. 
                           "Nanotech" includes quantum wires, nanopiezoelectronics, etc. 
                           "Biotech" includes gene therapy, drugs, etc.  
                           "Climate/Energy" includes carbon capture, solar panels, fusion reactors etc.
                           Do not provide any other commentary, just return one of those five words. Here is the description to help you decide on a type for this technology: "{tech.tr_text}"''')
        response = table.update(record.id, {'type': type})
        print('updated type for', tech.name, 'to', type)

    if tech.spi_impact is None:
        if tech.social_impact is None:
            social_response = social_benefits.run(f'''Given the following description of the technology "{tech.name}" and its impact since the year {tech.year}, provide your opinion on the technology and its impact on social well-being and people's lives. Do not provide any other commentary. While you can consider all types of impacts, including potential off-shoot technologies, use only the actual impacts of the technology to make an assessment, do not hypothesize about potential impacts. While you should not use this description to color your assessment, I provide it as further context for you on the technology. This is the description of why the MIT Technology Review thought this technology would be a breakthrough: "{tech.tr_text}"''')
            social_impact_level = social_benefits.run(f'''Choose one of "High", "Medium", or "Low" to describe the actual social impact level of the technology {tech.name}. Do not provide any additional commentary, simply return one of those three words. Use this assessment of the actual impact to inform your decision: "{social_response}"''')

            social_impact_potential = social_benefits.run(f'''Given the following description of the technology "{tech.name}", provide your opinion on the technology and its potential impact on social well-being and people's lives. Do not provide any other commentary. While you can consider all types of impacts, including potential off-shoot technologies, ignore existing impacts and only hypothesize about potential impacts into the future from today onward. While you should not use this description to color your assessment, I provide it as further context for you on the technology. This is the description of why the MIT Technology Review thought this technology would be a breakthrough: "{tech.tr_text}"''')
            social_impact_potential_level = social_benefits.run(f'''Choose one of "High", "Medium", or "Low" to describe the potential social impact level of the technology {tech.name}. Do not provide any additional commentary, simply return one of those three words. Use this assessment of the potential impact to inform your decision: "{social_response}"''')
        else:
            social_response = tech.social_impact
            social_impact_potential = tech.social_impact_potential
            social_impact_level = tech.social_impact_level
            social_impact_potential_level = tech.social_impact_potential_level
        
        if tech.spi_impact is None:
            spi_impact = social_benefits.run(f'''Given the following commentary on the technology "{tech.name}", provide your best guess at a % impact of this technology on the Social Progress Index since the year {tech.year} and over the next 20 years. 
                                        Consider a range of possible % impacts over a single order of magnitude and provide a single number that you think is the most likely. 
                                        Do not provide any other commentary. Just return a single % number. Remember that 1% means a 1% increase in the Social Progress Index can be attributed SOLELY to {tech.name}. Negative % is also acceptable if the technology impact is expected to reduce the Social Progress Index and negatively impact society. So we expect the number to be small yet precise.
                                        Here is the commentary to use for your decision: 
                                        "{social_response}"
                                        "{social_impact_potential}"
                                        ''')
        else:
            spi_impact = tech.spi_impact

        response = table.update(record.id, {'social_impact': social_response, 'social_impact_level': social_impact_level, 'social_impact_potential': social_impact_potential, 'social_impact_potential_level': social_impact_potential_level, 'spi_impact': spi_impact})
        print('updated social for', tech.name)

    if tech.quant_qual is None:
        quant_qual = social_benefits.run(f'''
                                 First, read this article by Bill Gates on the MIT Technology Review (TR) Breakthrough Technology picks that discusses "quantity of life" and "quality of life" as different ways technology can be impactful: "{gates_article}"

                                 Next, read this description of a "{tech.name}", a Breakthrough Technology that TR picked in year {tech.year}: "{tech.tr_text}"
                                 
                                 Now, using Bill Gates' article and description pick one of "quantity of life" or "quality of life" or "both" or "neither" to best describe "{tech.name}" impacts on society, humanity, and all living beings.

                                 If there is a negative impact, you can consider whether that negative impact is a "quantity" or "quality" impact.

                                 An example to consider: for climate technologies, can consider if the technology will help save lives ("quantity of life") or improve the quality of the lives that will remain ("quality of life"). Consider these types of thoughts when deciding between the four choices. "Both" and "neither" are also valid choices and can be seriously considered.

                                Do not provide any other commentary, just return one of those these four tags: "quantity of life", "quality of life", "both", "neither". 
                                ''')
        
        response = table.update(record.id, {'quant_qual': quant_qual})
        print('updated quant_qual for', tech.name)

    if tech.flop_type is None:
        flop_type = general.run(f'''
                                I'm going to give you a bunch of thoughts about the technology {tech.name}:
                                The article written by the MIT Technology Review in {tech.year} about why this technology will be a breakthrough: "{tech.tr_text}"
                                Next, here are some opinions on the social impact that this technology has had and may have in the future.
                                The actual social impact of the technology: "{tech.social_impact}"
                                The potential social impact of the technology: "{tech.social_impact_potential}"
                                The very rough estimation impact on the Social Progress Index: "{tech.spi_impact}"

                                Next, here are some opinions on the technology and whether it is actually a breakthrough.
                                The optimist's opinion: "{tech.optimist}"
                                The pessimist's opinion: "{tech.pessimist}"
                                The general opinion on the impact level of the technology: "{tech.impact}
                                An opinion on the success of the claim that this technology will be a breakthrough: "{tech.opinion}"   

                                Here is a general opinion by Bill Gates' on a way to measure the success of a technology: "{gates_article}"

                                Finally, you should be aware that we care about understanding whether the technology actually became a breakthrough like the MIT Technology Review claimed it would. 
                                
                                Respond in the following format:
                                [result]: [50-100 words on why you chose that result]
                                the [result] should be exactly one of the following: 
                                "1. Successful",
                                "2. Partially Successful", 
                                "3. Unsuccessful (unable to transition from research to widespread use)",
                                "4. Unsuccessful (underlying technology platform changed)",
                                "5. Unsuccessful (too early to tell)"
                                "6. Unsuccessful (not enough adoption)"

                                example of 1 would be "machine learning" because it has had a huge impact on society and will likely have a huge impact on human well being
                                example of 2 would be "cloud streaming" because while it has been impactful, it may not have had as huge an impact on human well-being compared to other technologies
                                example of 3 would be "quantum computing" because it has not yet transitioned from research to widespread use
                                example of 4 would be "Cell Phone Virus" because the underlying technology platform has changed and the technology did not have as much impact before the technology landscape changed
                                example of 5 would be "Carbon Removal Factory in Iceland" because it is too early to tell if it will have a huge impact on society's well-being
                                example of 6 would be "Babel-Fish Earbuds" because the technology has not been widely adopted, or adopted enough for it to be considered a significant enough impact on human well-being

                                Remember, a technology is a breakthrough if it has a significant impact on society and human well-being. This can be thought of as GDP-B, the Social Progress Index, or other measures of human well-being. But use that as your main criteria which result you choose.

                                Overall, be critical and discerning, with an eye for scientific accuracy. Do not assume technologies are all impactful, but do not be excessively critical either. Consider evidence and mention it briefly in your justification following the [result].
                                
                                Respond exactly in the format above, do not provide any other commentary.
                                ''')
        response = table.update(record.id, {'flop_type': flop_type})
        print('updated flop_type for', tech.name)

    print('time taken:', time.time() - start_time)

