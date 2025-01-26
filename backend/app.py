from flask import Flask, request, jsonify
from flask import Flask, request, jsonify, render_template
# from llama_index.memory.mem0 import Mem0Memory
import re
import json
from dotenv import load_dotenv
from pinecone import Pinecone
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.core import load_index_from_storage, StorageContext, Settings
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.query_engine import TransformQueryEngine
from llama_index.core import PromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine
import unicodedata
from rapidfuzz import fuzz
import asyncio
import nest_asyncio
import google.generativeai as genai
import unicodedata
from rapidfuzz import fuzz
from flask_cors import CORS
import cohere
import os
from flask import g
import time


app = Flask(__name__)
CORS(app)

nest_asyncio.apply()

load_dotenv()
cohere_api = os.environ["COHERE_API_TOKEN"]
co = cohere.ClientV2(api_key=cohere_api)

def to_lower_case(text):
    text = text.lower()
    text = ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
    return text


def small_talk_check(sentence, threshold=80):
    clean_text = to_lower_case(sentence)
    
    
    small_talk_keywords = [
        "xin chao", "hom nay", "khoe khong", "the nao", "cam thay", "noi chuyen", "vui ve", "tro chuyen",
        "toi ten la", "ten toi la", "toi o dau", "ban o dau", "co met khong", "ban nghi sao", "anh nghi sao",
        "cam on", "xin loi", "toi buon", "toi vui", "troi dep", "troi mua", "co bi khong", "co that khong",
        "co dung khong", "thay co tot khong", "cai nay sao", "ban thay sao", "co bip hay khong", "dung that khong",
        "thay dạy như cứt", "dạy như thế nào", "thầy dạy môn này sao", "thầy dạy môn này chán", "môn này như cứt",
        "co dạy như thế nào", "co ABC", "co dạy môn này sao", "co dạy môn này chán", "co như thế nào", "co bip hay khong",
        "tệ", "chán", "xấu", "như cứt", "quá tệ", "cực kỳ tệ", "không hay", "chán ngắt", "cái này chán", "dở",
        "có gì mới không", "bạn nghĩ sao", "bạn có khỏe không", "bạn làm gì thế", "bạn thích ăn gì", "tôi buồn", "tôi vui",
        "tôi mệt", "thật tệ", "rất tốt", "cảm ơn", "cảm ơn bạn", "xin lỗi", "chào bạn", "hôm nay bạn thế nào",
        "bạn đang làm gì", "thầy giảng như thế nào", "môn này có khó không", "môn này như thế nào", "môn học này có gì hay",
        "có thầy cô nào dạy tốt không", "thầy dạy như thế nào", "co bip hay khong", "dạy như thế nào", "hello", "hi"
    ]
    
    
    courses_info_keywords = [
        "ma mon hoc", "mon","mon hoc", "sinh vien", "noi dung", "kien thuc", "giang day", "phuong phap", 
        "bai tap", "thoi khoa bieu", "ky thi", "dai hoc", "giang duong", "bai giang", "hoc ky", 
        "tai lieu", "truong hoc", "ky nang", "ky thuat", "thuc hanh", "kiem tra", "de thi", "sach giao khoa"
    ]
    
    
    small_talk_match = any(fuzz.partial_ratio(clean_text, kw) >= threshold for kw in small_talk_keywords)
    
    
    courses_info_match = any(fuzz.partial_ratio(clean_text, kw) >= threshold for kw in courses_info_keywords)
    
    
    if small_talk_match and not courses_info_match:
        return 1
    return 0



pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"]) 
embedding_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")
llm = HuggingFaceInferenceAPI(model_name="mistralai/Mistral-7B-Instruct-v0.2", token=os.environ["HF_TOKEN"])
Settings.llm = llm
Settings.embed_model = embedding_model


example_query_prompt = """
You are a Vietnamese language expert. 

### Guidelines:
1. Response with Vietnamese
2. Only based on information about course information do not care about files information
3. Use natural and conversational Vietnamese suitable for the intended audience.
4. Avoid repetition and ensure each question addresses a distinct aspect of the context.
5. remove all special token
6. you can not have an answer just response you don't know.
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
    6. you can not have an answer just response you don't know.
    7. Do not Hallucinate if you don't know the answer.
    8. Only use Vietnamese for answer

{context_str}

Passage:
"""

chat_prompt = """
You are a Vietnamese language expert. 

### Guidelines:
1. Response with Vietnamese
2. Only based on information about course information do not care about files information. 
3. Use natural and conversational Vietnamese suitable for the intended audience.
4. Avoid repetition and ensure each question addresses a distinct aspect of the context.
5. remove all special token
6. Short answer
7. Do not Hallucinate if you don't know the answer.
8. Do not repeat the question

let's think step by step

response the question based on context:
"""

@app.before_request
def before_request():
    g.start = time.time()

@app.after_request
def after_request(response):
    diff = time.time() - g.start
    if ((response.response) and (200 <= response.status_code < 300)):
        print(f"[{request.method}] {request.path} executed in {diff:.4f} seconds")
    return response


def extract_years(text):
    pattern = r'20[0-9][0-9]'
    return re.findall(pattern, text)


def result_content(results):
    matches = results['matches']
    response_texts = []
    for match in matches:
        node_content = json.loads(match['metadata']['_node_content'])
        match_text = node_content['text']
        response_texts.append(match_text)
    response = "\n".join(response_texts)
    return response


storage_context = StorageContext.from_defaults(persist_dir=r"C:\Users\leduc\OneDrive\Desktop\NLP\LLM-Agent\AI _solution\rag_database")
index = load_index_from_storage(storage_context)

retriever = QueryFusionRetriever(
    [
        index.as_retriever(similarity_top_k=10),
        BM25Retriever.from_defaults(docstore=index.docstore, similarity_top_k=10)
    ],
    num_queries=1,
    use_async=True
)

cohere_rerank = CohereRerank(model="rerank-v3.5", api_key=cohere_api, top_n=4)
query_engine = RetrieverQueryEngine(retriever=retriever, node_postprocessors=[cohere_rerank])

example_query_prompt = PromptTemplate(example_query_prompt)

query_engine.update_prompts({"response_synthesizer:text_qa_template": example_query_prompt})

hyde = HyDEQueryTransform(include_original=True, llm=llm, hyde_prompt=PromptTemplate(hyde_prompt))
hyde_query_engine = TransformQueryEngine(query_engine, hyde)

chat_engine = index.as_chat_engine(
    chat_mode="condense_plus_context",
    system_prompt=(chat_prompt),
    query_engine=query_engine,
    verbose=True,
    llm=llm,
    max_tokens=1000, 
)


def get_embed(text):
    return embedding_model._embed(text)


@app.route('/agent', methods=["POST"])
def handle_query():
    """
    Params:
    year -> list: contain extracted years from query
    query_embedding: vector embedding of the query
    query_str -> str: raw query string
    
    Output:
    str: response
    """
    data = request.json
    user_id = data.get("user_id", "user_1")
    query = data.get("query", "")

    pinecone_index = pc.Index(host="https://agent-quoor8m.svc.aped-4627-b74a.pinecone.io")
        
    year = extract_years(query)
    query_embedding = get_embed(query)
    small_talk = small_talk_check(query)
    if year:
        try:
            print("year")
            results = pinecone_index.query(
                vector=query_embedding,
                filter={"year": {"$in": year}},
                top_k=5,
                include_metadata=True
            )
            res = result_content(results)
            react_response = chat_engine.chat(query + "\n" + res)
            chat_engine.reset()
            return jsonify({
                "query": query,
                "response": react_response.response.split("user:")[0].strip(),
                "status": "success"
            })
        except Exception as e:
            return jsonify({
                "query": query,
                "response": "Lỗi khi truy vấn dữ liệu.",
                "status": "error",
                "error": str(e)
            })
    elif(small_talk == 0 and year == []):
        print("react")
        react_response = chat_engine.chat(query)
        chat_engine.reset()
        
        return jsonify({
            "query": query,
            "response": react_response.response.split("user:")[0].strip(),
            "status": "success"
        })
    

    if small_talk_check(query):
        try:
            print("small talk")
            response = co.chat(
                model="command-r-plus-08-2024",  
                messages=[{"role": "user", "content": query}],
            )
            cohere_response = response.message.content[0].text
            response = {
                "query": query,
                "response": cohere_response,
                "status": "success"
            }
        except Exception as e:
            response = {
                "query": query,
                "response": "Xin lỗi, hiện tại tôi không thể trả lời câu hỏi này.",
                "status": "error",
                "error": str(e)
            }
        return jsonify(response)
       
    

   
    
    chat_engine.reset()

if __name__ == "__main__":
    app.run(debug=True)
