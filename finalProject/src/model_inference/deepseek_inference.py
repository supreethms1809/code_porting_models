from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from datasets import Dataset
import logging
import json
import re
from tqdm import tqdm
import os
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class modelInference:
    def __init__(self, model_name="deepseek-r1:8b"):
        self.model_name = model_name
        self.llm = ChatOllama(model=model_name, temperature=0.6, num_predict=-1)
        self.template = """
                You are a high performance computing software engineer.  \n
                and your task is to apply the action on the given code and your final goal is to get a CUDA version of the code \n
                Perform the actions: {action} on the code to get the CUDA version \n 
                Input code given: \n
                {cpp_code} \n
                Please think and reason step by step to achieve code porting task and
                Output should clearly differentiate between the thought process, summary and the code. \n
                Always provide the translated code and always place the translated code in the section call "translated code" \n
                Use the same variable names as in the input code. \n
                Start the output with '<think>\n' \n
                """
        self.template_analysis = """
                You are a high performance computing software engineer. I want you to analyze the code and provide the summary of the code. \n
                I have provided the Abstract Syntax Tree (AST) of the code below. \n
                AST of the Input code given: \n
                {cpp_code} \n
                Please think and reason step by step to provide answers to the following questions\n
                {analysis_actions} analysis \n
                \n
                Output should clearly differentiate between the thought process, the analysis of the code \n \n \n
                Start the output with '<think>\n' \n
                """
        self.questions = """
        1. Are the loops embarrassingly_parallel?
        2. Does the calculations inside loop has_data_dependencies?
        3. Are there any loop_carried dependencies?
        4. Are the loops fusable?
        5. Are the loops splittable?
        6. Are the loops compute_bound or memory_bound?
        7. Does the loop body have conditional exits (break, return, etc.)?
        """
 

    def parsemessage(self, ai_message: AIMessage) -> str:
        """Parse the AI message."""
        message = ai_message.content.swapcase()
        match_translated = re.search(r"```\s*(.*?)```", message, re.DOTALL | re.IGNORECASE)

        if match_translated:
            translated_code = match_translated.group(1).strip()
            #print(translated_code)
        else:
            print("No translated code found.")

        match = re.search(r"<think>(.*?)</think>", message, re.DOTALL | re.IGNORECASE)
        code_match = re.search(r"```(.*?)```", message, re.DOTALL)
        kernel_match = re.search(r"__GLOBAL__\s+VOID\s+\w+\(.*?\)\s*{.*?}", message, re.DOTALL | re.IGNORECASE)

        if match:
            think_content = match.group(1).strip()
        else:
            logger.warning("No <think> tag found in the message.")
            think_content = ""

        if code_match:
            summary_content = re.sub(r"<think>.*?</think>", "", message, flags=re.DOTALL | re.IGNORECASE).strip()
            cuda_code = code_match.group(1).strip()
        else:
            logger.warning("No <code> tag found in the message.")
            summary_content = message.strip()
            cuda_code = ""
        
        if kernel_match:
            kernel_code = kernel_match.group(0).strip()
        else:
            logger.warning("No kernel code found in the message.")
            kernel_code = ""

        
        return translated_code, think_content, summary_content, cuda_code, kernel_code


    def infer(self, cpp_code):
        prompt = ChatPromptTemplate.from_template(self.template_analysis)
        formatted_prompt_template = prompt.format(cpp_code=cpp_code, analysis_actions=self.questions)
        response = self.llm.invoke(formatted_prompt_template)
        return response.content.swapcase()
        #think_content, summary_content, cuda_code, kernel_code = parsemessage(AIMessage(content=response.content))

    def infer_transform(self, cpp_code, action):
        prompt = ChatPromptTemplate.from_template(self.template)
        formatted_prompt_template = prompt.format(cpp_code=cpp_code, action=action)
        response = self.llm.invoke(formatted_prompt_template)
        print(response.content.swapcase())
        translated_code, think_content, summary_content, cuda_code, kernel_code = self.parsemessage(AIMessage(content=response.content))
        return translated_code