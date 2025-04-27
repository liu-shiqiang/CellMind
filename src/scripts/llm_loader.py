import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import ChatHuggingFace,HuggingFacePipeline
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from config.setting import settings


class ModelLoader:
    def __init__(self, model_name):
        self.model_name = model_name
        self.logger = logging.getLogger(__name__)
        self.hf_model_engines = ['Deepseek-R1-1.5b-hf',
                                    'Deepseek-R1-7b-hf',
                                    'DeepSeek-R1-32B-hf',
                                    'Llama-2-7b-hf',
                                    'Llama-2-13b-hf',
                                    'Mixtral-8x7B-Instruct-v0.1'
                                    ]
        self.gpt_model_engines = ['gpt-3.5-turbo',
                                  'gpt-4-turbo',
                                  'gpt-4',
                                  'gpt-4o',
                                  'gpt-4o-mini',
                                  'gpt-3.5-turbo-1106',
                                  'gpt-4-0613',
                                  'gpt-4-32k-0613',
                                  'gpt-4-1106-preview']
        self.ollama_engines = [ 'ollama_llama3.1',
                                'ollama_deepseek-r1',
                                'ollama_deepseek-r1:1.5b',
                                'ollama_deepseek-r1:14b',
                                'ollama_deepseek-r1:32b',
                                 ]
        self.model = self.load_model()
    
    def load_model(self):
        if self.model_name in self.hf_model_engines:
            return self.load_hf_model()
        elif self.model_name in self.gpt_model_engines:
            return self.load_gpt_model()
        elif self.model_name in self.ollama_engines:
            return self.load_ollama_model()
        else:
            self.logger.error(f"model {self.model_name} not supported")
            raise ValueError(f"not supported model: {self.model_name}")

    def load_hf_model(self):
        model_path = settings.local_MODEL_PATH + self.model_name
        self.logger.info(f"loading hf model: {model_path}")

        # try from_model_id method
        try:
            llm = HuggingFacePipeline.from_model_id(
                model_id=model_path,
                task="text-generation",
                pipeline_kwargs={
                    "max_new_tokens": 1024,
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "repetition_penalty": 1.1,
                    "device_map": "auto"
                },
            )
            return ChatHuggingFace(llm=llm)
        except Exception as e:
            self.logger.warning(f"load fail ,try to load manually: {e}")

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(
                model_path, 
                device_map="auto",
                torch_dtype=torch.float16
            )

            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=1024,
                temperature=0.3,
                top_p=0.9,
                repetition_penalty=1.1
            )
            return ChatHuggingFace(HuggingFacePipeline(pipeline=pipe))
        except Exception as e:
            self.logger.error(f"load hf model fail: {e}")
            raise RuntimeError("can not load hf model")

    def load_gpt_model(self):

        self.logger.info(f"load OpenAI GPT model: {self.model_name}")
        return ChatOpenAI(
            model=self.model_name,
            temperature=0.3,
            max_tokens=500,
            api_key=settings.OPENAI_API_KEY,
            timeout=10
        )

    def load_ollama_model(self):

        self.logger.info(f"load Ollama model: {self.model_name}")
        model_name = self.model_name.removeprefix('ollama_')
        return ChatOllama(
            model=model_name,
            temperature=0.3,
            max_tokens=500,
            base_url='http://localhost:11434'
        )

        
        
        