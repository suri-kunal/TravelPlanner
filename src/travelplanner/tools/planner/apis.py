from functools import partial
import sys
import os
from travelplanner.agents.prompts import planner_agent_prompt, cot_planner_agent_prompt, react_planner_agent_prompt, reflect_prompt, react_reflect_planner_agent_prompt, REFLECTION_HEADER, PromptTemplateFromScratch
from travelplanner.tools.planner.env import ReactEnv,ReactReflectEnv
import tiktoken
import re
import openai
import time
from enum import Enum
from typing import List

def catch_openai_api_error():
    error = sys.exc_info()[0]
    if error == openai.error.APIConnectionError:
        print("APIConnectionError")
    elif error == openai.error.RateLimitError:
        print("RateLimitError")
        time.sleep(60)
    elif error == openai.error.APIError:
        print("APIError")
    elif error == openai.error.AuthenticationError:
        print("AuthenticationError")
    else:
        print("API error:", error)

class ReflexionStrategy(Enum):
    """
    REFLEXION: Apply reflexion to the next reasoning trace 
    """
    REFLEXION = 'reflexion'

class Planner:
    def __init__(self,
                 # args,
                 agent_prompt: PromptTemplateFromScratch = planner_agent_prompt,
                 model_name: str = 'gpt-4o-mini-2024-07-18',
                 ) -> None:
        
        if "gemini" in model_name.lower():
            self.model_name = model_name
            self.parameters = {"temperature":1,"max_completion_tokens":4000}
        elif ("gpt" in model_name.lower()) or model_name.lower().startswith("o"):
            self.model_name = model_name
            self.parameters = {"temperature":1,"max_completion_tokens":4000}
        else:
            raise Exception(f"Currently we support Gemini and OpenAI models only.")
        self.client = openai.OpenAI(
            # Should be for Gemini or OpenAI. We rely on user to provide correct API - we don't do any checks
            api_key=os.environ.get("API_KEY"),
            base_url=os.environ.get("BASE_URL")
        )
        self.llm = partial(self._get_completion,model_name=self.model_name,**self.parameters)
        self.agent_prompt = agent_prompt
        self.scratchpad: str = ''
        self.enc = tiktoken.encoding_for_model("gpt-3.5-turbo")

        print(f"PlannerAgent {model_name} loaded.")

    def run(self, text, query, log_file=None) -> str:
        if log_file:
            log_file.write('\n---------------Planner\n'+self._build_agent_prompt(text, query))
        if len(self.enc.encode(self._build_agent_prompt(text, query))) > 12000:
            return 'Max Token Length Exceeded.'
        else:
            result = self.llm(content=self._build_agent_prompt(text, query))
            if len(result.choices) == 0:
                raise Exception("result has no response")
            if len(result.choices) > 1:
                raise Exception("result has multiple responses. Let's check it")
            return result.choices[0].message.content

    def _build_agent_prompt(self, text, query) -> str:
        return self.agent_prompt(
            text = text,
            query = query
        )
    
    def _get_completion(self,model_name,content,**params):
        completion = self.client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": content,
                },
            ],
            **params
        )
        return completion

class ReactPlanner:
    """
    A question answering ReAct Agent.
    """
    def __init__(self,
                 agent_prompt: PromptTemplateFromScratch = react_planner_agent_prompt,
                 model_name: str = 'gpt-3.5-turbo-1106',
                 ) -> None:
        
        if "gemini" in model_name.lower():
            self.model_name = model_name
            self.parameters = {"temperature":1,"max_completion_tokens":4000,"stop":["Action","Thought","Observation"]}
        elif ("gpt" in model_name.lower()) or model_name.lower().startswith("o"):
            self.model_name = model_name
            self.parameters = {"temperature":1,"max_completion_tokens":4000,"stop":["Action","Thought","Observation"]}
        else:
            raise Exception(f"Currently we support Gemini and OpenAI models only.")
        self.client = openai.OpenAI(
            # Should be for Gemini or OpenAI. We rely on user to provide correct API - we don't do any checks
            api_key=os.environ.get("API_KEY"),
            base_url=os.environ.get("BASE_URL")
        )

        self.agent_prompt = agent_prompt
        self.react_llm = \
        partial(
            self._get_completion,
            model_name=self.model_name,
            **self.parameters
        )
        self.env = ReactEnv()
        self.query = None
        self.max_steps = 30
        self.reset()
        self.finished = False
        self.answer = ''
        self.enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
    
    def _get_completion(self,model_name,content,**params):
        completion = self.client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": content,
                },
            ],
            **params
        )
        return completion

    def run(self, text, query, reset = True) -> None:

        self.query = query
        self.text = text

        if reset:
            self.reset()
        

        while not (self.is_halted() or self.is_finished()):
            self.step()
        
        return self.answer, self.scratchpad

    
    def step(self) -> None:
        # Think
        self.scratchpad += f'\nThought {self.curr_step}:'
        self.scratchpad += ' ' + self.prompt_agent()
        print(self.scratchpad.split('\n')[-1])

        # Act
        self.scratchpad += f'\nAction {self.curr_step}:'
        action = self.prompt_agent()
        self.scratchpad += ' ' + action
        print(self.scratchpad.split('\n')[-1])

        # Observe
        self.scratchpad += f'\nObservation {self.curr_step}: '

        action_type, action_arg = parse_action(action)

        if action_type == 'CostEnquiry':
            try:
                input_arg = eval(action_arg)
                if type(input_arg) != dict:
                    raise ValueError('The sub plan can not be parsed into json format, please check. Only one day plan is supported.')
                observation = f'Cost: {self.env.run(input_arg)}'
            except SyntaxError:
                observation = f'The sub plan can not be parsed into json format, please check.'
            except ValueError as e:
                observation = str(e)
        
        elif action_type == 'Finish':
            self.finished = True
            observation = f'The plan is finished.'
            self.answer = action_arg
        
        else:
            observation = f'Action {action_type} is not supported.'
        
        self.curr_step += 1

        self.scratchpad += observation
        print(self.scratchpad.split('\n')[-1])

    def prompt_agent(self) -> str:
        while True:
            try:
                return format_step(self.react_llm(content=self._build_agent_prompt()).choices[0].message.content)
            except:
                catch_openai_api_error()
                print(self._build_agent_prompt())
                print(len(self.enc.encode(self._build_agent_prompt())))
                time.sleep(5)
    
    def _build_agent_prompt(self) -> str:
        return self.agent_prompt(
            query = self.query,
            text = self.text,
            scratchpad = self.scratchpad
        )
    
    def is_finished(self) -> bool:
        return self.finished

    def is_halted(self) -> bool:
        agent_prompt = self._build_agent_prompt()
        print(agent_prompt)
        return ((self.curr_step > self.max_steps) or (
                    len(self.enc.encode(agent_prompt)) > 14000)) and not self.finished

    def reset(self) -> None:
        self.scratchpad = ''
        self.answer = ''
        self.curr_step = 1
        self.finished = False

class ReactReflectPlanner:
    """
    A question answering Self-Reflecting React Agent.
    """
    def __init__(
            self,
            agent_prompt: PromptTemplateFromScratch = react_reflect_planner_agent_prompt,
            reflect_prompt: PromptTemplateFromScratch = reflect_prompt,
            model_name: str = 'gpt-3.5-turbo-1106',
    ) -> None:
        
        if "gemini" in model_name.lower():
            self.model_name = model_name
            self.parameters = {"temperature":1,"max_completion_tokens":4000,"stop":["Action","Thought","Observation"]}
        elif ("gpt" in model_name.lower()) or model_name.lower().startswith("o"):
            self.model_name = model_name
            self.parameters = {"temperature":1,"max_completion_tokens":4000,"stop":["Action","Thought","Observation"]}
        else:
            raise Exception(f"Currently we support Gemini and OpenAI models only.")
        self.client = openai.OpenAI(
            # Should be for Gemini or OpenAI. We rely on user to provide correct API - we don't do any checks
            api_key=os.environ.get("API_KEY"),
            base_url=os.environ.get("BASE_URL")
        )

        self.agent_prompt = agent_prompt
        self.reflect_prompt = reflect_prompt
        self.react_llm = \
        partial(
            self._get_completion,
            model_name=self.model_name,
            **self.parameters
        )
        self.reflect_llm = \
        partial(
            self._get_completion,
            model_name=self.model_name,
            **self.parameters
        )
        self.model_name = model_name
        self.env = ReactReflectEnv()
        self.query = None
        self.max_steps = 30
        self.reset()
        self.finished = False
        self.answer = ''
        self.reflections: List[str] = []
        self.reflections_str: str = ''
        self.enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
    
    def _get_completion(self,model_name,content,**params):
        completion = self.client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": content,
                },
            ],
            **params
        )
        return completion

    def run(self, text, query, reset = True) -> None:

        self.query = query
        self.text = text

        if reset:
            self.reset()
        

        while not (self.is_halted() or self.is_finished()):
            self.step()
            if self.env.is_terminated and not self.finished:
                self.reflect(ReflexionStrategy.REFLEXION)

        
        return self.answer, self.scratchpad

    
    def step(self) -> None:
        # Think
        self.scratchpad += f'\nThought {self.curr_step}:'
        self.scratchpad += ' ' + self.prompt_agent()
        print(self.scratchpad.split('\n')[-1])

        # Act
        self.scratchpad += f'\nAction {self.curr_step}:'
        action = self.prompt_agent()
        self.scratchpad += ' ' + action
        print(self.scratchpad.split('\n')[-1])

        # Observe
        self.scratchpad += f'\nObservation {self.curr_step}: '

        action_type, action_arg = parse_action(action)

        if action_type == 'CostEnquiry':
            try:
                input_arg = eval(action_arg)
                if type(input_arg) != dict:
                    raise ValueError('The sub plan can not be parsed into json format, please check. Only one day plan is supported.')
                observation = f'Cost: {self.env.run(input_arg)}'
            except SyntaxError:
                observation = f'The sub plan can not be parsed into json format, please check.'
            except ValueError as e:
                observation = str(e)
        
        elif action_type == 'Finish':
            self.finished = True
            observation = f'The plan is finished.'
            self.answer = action_arg
        
        else:
            observation = f'Action {action_type} is not supported.'
        
        self.curr_step += 1

        self.scratchpad += observation
        print(self.scratchpad.split('\n')[-1])

    def reflect(self, strategy: ReflexionStrategy) -> None:
        print('Reflecting...')
        if strategy == ReflexionStrategy.REFLEXION: 
            self.reflections += [self.prompt_reflection()]
            self.reflections_str = format_reflections(self.reflections)
        else:
            raise NotImplementedError(f'Unknown reflection strategy: {strategy}')
        print(self.reflections_str)

    def prompt_agent(self) -> str:
        while True:
            try:
                return format_step(self.react_llm(content=self._build_agent_prompt()).choices[0].message.content)
            except:
                catch_openai_api_error()
                print(self._build_agent_prompt())
                print(len(self.enc.encode(self._build_agent_prompt())))
                time.sleep(5)
    
    def prompt_reflection(self) -> str:
        while True:
            try:
                return format_step(self.reflect_llm(content=self._build_reflection_prompt()).choices[0].message.content)
            except:
                catch_openai_api_error()
                print(self._build_reflection_prompt())
                print(len(self.enc.encode(self._build_reflection_prompt())))
                time.sleep(5)
    
    def _build_agent_prompt(self) -> str:
        return self.agent_prompt(
            query = self.query,
            text = self.text,
            scratchpad = self.scratchpad,
            reflections = self.reflections_str
        )
    
    def _build_reflection_prompt(self) -> str:
        return self.reflect_prompt(
            query = self.query,
            text = self.text,
            scratchpad = self.scratchpad
        )
    
    def is_finished(self) -> bool:
        return self.finished

    def is_halted(self) -> bool:
        return ((self.curr_step > self.max_steps) or (
                    len(self.enc.encode(self._build_agent_prompt())) > 14000)) and not self.finished

    def reset(self) -> None:
        self.scratchpad = ''
        self.answer = ''
        self.curr_step = 1
        self.finished = False
        self.reflections = []
        self.reflections_str = ''
        self.env.reset()

def format_step(step: str) -> str:
    return step.strip('\n').strip().replace('\n', '')

def parse_action(string):
    pattern = r'^(\w+)\[(.+)\]$'
    match = re.match(pattern, string)

    try:
        if match:
            action_type = match.group(1)
            action_arg = match.group(2)
            return action_type, action_arg
        else:
            return None, None
        
    except:
        return None, None

def format_reflections(reflections: List[str],
                        header: str = REFLECTION_HEADER) -> str:
    if reflections == []:
        return ''
    else:
        return header + 'Reflections:\n- ' + '\n- '.join([r.strip() for r in reflections])