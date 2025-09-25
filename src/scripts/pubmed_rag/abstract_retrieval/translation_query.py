from langchain_core.prompts import PromptTemplate
from src.scripts.llm_loader import llm


def translation_chain(scientist_question: str) -> str:
    """ 中文输出翻译成英文 """
    prompt_formatted_str = translation_prompt.format(question=scientist_question)
    return llm.invoke(prompt_formatted_str).content


translation_prompt = PromptTemplate.from_template("""
  You are an expert in biomedical terminology. Your task is to translate the following Chinese question into English, ensuring the correct use of scientific and medical terms. Please focus on preserving the meaning and accurately translating any biomedical concepts.

  Chinese Question: {question}
""")