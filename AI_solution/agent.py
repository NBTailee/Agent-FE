import pdfplumber
from llama_index.core import SimpleDirectoryReader, Document, VectorStoreIndex, StorageContext,  ChatPromptTemplate
from llama_index.core.node_parser import SentenceSplitter, TokenTextSplitter, SentenceWindowNodeParser, SemanticSplitterNodeParser
# from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import Settings

from llama_index.core.extractors import (
    SummaryExtractor,
    QuestionsAnsweredExtractor,
    TitleExtractor,
    KeywordExtractor,
)


from llama_index.core.ingestion import IngestionPipeline
from llama_index.llms.huggingface import HuggingFaceLLM, HuggingFaceInferenceAPI
from llama_index.llms.openai import OpenAI
from llama_index.extractors.entity import EntityExtractor
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.query_engine import TransformQueryEngine
from llama_index.core import PromptTemplate



# from llama_index.core.postprocessor import LLMRerank, SentenceTransformerRerank
# from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response.pprint_utils import pprint_response
from llama_index.core.response.notebook_utils import display_source_node
from llama_index.core import load_index_from_storage


import openai
from pinecone import Pinecone
import pinecone
import os
from dotenv import load_dotenv
import json

import google.generativeai as genai
from small_talk_check import small_talk_check 

import asyncio
import nest_asyncio
nest_asyncio.apply()

load_dotenv()

genai.configure(api_key=os.environ["GEMINI_TOKEN"])
model = genai.GenerativeModel("gemini-1.5-flash")

pc = Pinecone(api_key = os.environ["PINECONE_API_KEY"])
openai.api_key = os.environ["OPEN_AI_KEY"]

embbeding_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")
llm = HuggingFaceInferenceAPI(model_name="mistralai/Mistral-7B-Instruct-v0.2", token=os.environ["HF_TOKEN"])

Settings.llm = llm
Settings.embed_model = embbeding_model

import re
course_code_pattern = r"\b[A-Z]{2}[0-9]{3}\b"

example_query_prompt = """
You are a Vietnamese language expert. 

### Guidelines:
1. Response with Vietnamese
2. Only based on information about course information do not care about files information
3. Use natural and conversational Vietnamese suitable for the intended audience.
4. Avoid repetition and ensure each question addresses a distinct aspect of the context.
5. remove all special token
6. you can not have an answer just reponse you don't know.
7. Do not Hallucinate if you don't know the answer.
8. Do not repeat the question.


let's think step by step

response the question based on context:

question: {query_str}

context: {context_str}

answer:

"""

hyde_prompt = """
You are a Vietnamese language expert    

    1. Please write a passage to answer the question in Vietnamese
    2. Try to include as many key details as possible.
    3. Use natural and conversational Vietnamese suitable for the intended audience.
    4. Avoid repetition and ensure each question addresses a distinct aspect of the context.
    5. remove all special token
    6. you can not have an answer just reponse you don't know.
    7. Do not Hallucinate if you don't know the answer.
    8. Only use Vietnamese for answer
        
    {context_str}
    
    
    Passage:
"""

chat_prompt = """
You are a Vietnamese language expert. 

### Guidelines:
1. Response with Vietnamese
2. Only based on information about course information do not care about files information
3. Use natural and conversational Vietnamese suitable for the intended audience.
4. Avoid repetition and ensure each question addresses a distinct aspect of the context.
5. remove all special token
6. Short answer
7. Do not Hallucinate if you don't know the answer.
8. Do not repeat the question.


let's think step by step

response the question based on context:


"""
    

def extract_years(text):
    pattern = r'20[0-9][0-9]'
    return re.findall(pattern, text)

# Load local index
storage_context = StorageContext.from_defaults(persist_dir=r"./rag_database")
index =  load_index_from_storage(storage_context)


# implement hybrid search, keywords search, metadata filtering, HyDE query transformation
retriever = QueryFusionRetriever(
    [
        index.as_retriever(similarity_top_k=10),
        BM25Retriever.from_defaults(
            docstore=index.docstore, similarity_top_k=10
        ),
    ],
    num_queries=1,
    use_async=True,
)

cohere_rerank = CohereRerank(model="rerank-v3.5" ,api_key=os.environ["COHERE_API_TOKEN"], top_n=4)
query_engine = RetrieverQueryEngine(retriever=retriever, node_postprocessors=[cohere_rerank])


example_query_prompt = PromptTemplate(example_query_prompt)

query_engine.update_prompts(
    {"response_synthesizer:text_qa_template": example_query_prompt},
)

hyde = HyDEQueryTransform(include_original=True, llm=llm, hyde_prompt=PromptTemplate(hyde_prompt))
hyde_query_engine = TransformQueryEngine(query_engine, hyde)

chat_engine = index.as_chat_engine(chat_mode="condense_plus_context", system_prompt=(chat_prompt) ,query_engine = query_engine ,verbose=True, llm=llm, max_tokens = 1000)




# Ham xuat ket qua cho Pinecone
def result_content(results):
    matches = results['matches']  
    response_texts = []
    for match in matches:
        node_content = json.loads(match['metadata']['_node_content'])
        match_text = node_content['text']
        response_texts.append(match_text)
    response = "\n".join(response_texts)
    return response

# connect voi Pinecone Index
pinecone_index = pc.Index(host="https://agent-quoor8m.svc.aped-4627-b74a.pinecone.io")

def get_embed(text):
    return embbeding_model._embed(text)


def get_chat(query_str):
    """
    Params:
    year -> list: contain extracted years from query
    query_embedding: tu biet di
    query_str -> str: tu biet di
    
    Output:
    str: response
    """
    if(small_talk_check(query_str)):
        res = model.generate_content(contents=query_str)
        return res.text
    

    query_embedding = get_embed(query_str)
    year = extract_years(query_str)
   
    if year:
        results = pinecone_index.query(
            vector= query_embedding,
            filter={"year": {"$in": year}},
            top_k=5,
            include_metadata=True
        )
        res = result_content(results)
        react_response = chat_engine.chat(query_str + "\n" + res)
        chat_engine.reset()
        return react_response.response
    else:
        react_response = chat_engine.chat(query_str)
        chat_engine.reset()
        return react_response.response
    
