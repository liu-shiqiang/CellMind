import os
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

#  OpenAI 的 API Key
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

# 初始化 OpenAI 嵌入模型
embeddings = HuggingFaceEmbeddings(
            model_name='/home/share/huadjyin/home/liushiqiang/pretrained_model/all-MiniLM-L6-v2'
        )


