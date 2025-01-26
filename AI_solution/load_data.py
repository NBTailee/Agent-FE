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


import openai
from pinecone import Pinecone
import pinecone
import os
from dotenv import load_dotenv


import asyncio
import nest_asyncio
nest_asyncio.apply()

load_dotenv()

# Loading things
pc = Pinecone(api_key = os.environ["PINECONE_API_KEY"])
openai.api_key = os.environ["OPEN_AI_KEY"]
embbeding_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")
llm = HuggingFaceInferenceAPI(model_name="mistralai/Mistral-7B-Instruct-v0.2", token=os.environ["HF_TOKEN"])
Settings.llm = llm
Settings.embed_model = embbeding_model



# Loading data and transform nigga
reader = SimpleDirectoryReader(input_dir=r"C:\Users\leduc\OneDrive\Desktop\NLP\LLM-Agent\AI _solution\data_rag")
documents = reader.load_data(num_workers=4)


splitter = SemanticSplitterNodeParser(
    buffer_size=2, breakpoint_percentile_threshold=80, embed_model=embbeding_model
)

# PROMPTS
FAQ_prompt = """Ensure the questions are well-structured, grammatically correct, and contextually accurate.
You are a Vietnamese language expert and question-generation specialist. Based on the following context, generate exactly three concise, thoughtful, and relevant questions in Vietnamese. 

### Context:
{context_str}

### Guidelines:
1. The questions should focus on extracting meaningful insights or prompting further discussion about the context.
2. Only based on information about course information do not care about files information
3. Use natural and conversational Vietnamese suitable for the intended audience.
4. Avoid repetition and ensure each question addresses a distinct aspect of the context.
5. remove all special token
6. Generate {num_questions}

### Example Output:
If the context is about environmental protection:
1. Bạn nghĩ gì về vai trò của giáo dục trong việc bảo vệ môi trường?
2. Những giải pháp nào có thể giúp giảm thiểu rác thải nhựa trong cuộc sống hàng ngày?
3. Làm thế nào cộng đồng có thể hợp tác để cải thiện chất lượng không khí?

### Output:



"""

keyword_prompt = {
  "context": "You are a Vietnamese language expert and Vietnamese keywords identifier specialist.",
  "task": "Generate {keywords} unique keywords for this document.",
  "document": "{context_str}",
  "guidelines": [
    "Only based on information about course information do not care about files information.",
    "MUST NOT return any explanation.",
    "Answer in Vietnamese.",
  ],
  "examples":[
    "CS115, Toán, Khoa Học Máy Tính, 2023",
    "CE231, Tin học, Phần mềm",
    "SE112, Robotic, Vi mạch"  
  ],
  "output_format": "keywords: "
}
import json

keyword_prompt = json.dumps(keyword_prompt, ensure_ascii=False, indent=4)


transformations = [
    splitter,
    KeywordExtractor(keywords=1, llm=llm, prompt_template= keyword_prompt),
    QuestionsAnsweredExtractor(questions=2, llm=llm, prompt_template= FAQ_prompt)
]

pipeline = IngestionPipeline(transformations=transformations)

nodes = pipeline.run(documents=documents)

print("\n\n\n")
print(len(nodes))

import re
course_code_pattern = r"\b[A-Z]{2}[0-9]{3}\b"

def extract_years(text):
    pattern = r'20[0-9][0-9]'
    return re.findall(pattern, text)



def clean_text(text):
    text = re.sub(r'\d+\.\s*', '', text)  
    text = re.sub(r'-', ' ', text)
    text = re.sub(r'"', ' ', text)
    text = text.replace('\n', ' ').strip()
    text = re.sub(r'\s+', ' ', text) 
    text = re.sub(r'\bkeywords: ', '', text)
    text = text.strip()
    return text


for i in range(0,len(nodes)):
    nodes[i].metadata["course"] = re.findall(course_code_pattern, nodes[i].text)
    nodes[i].metadata["year"] = extract_years(nodes[i].text)
    nodes[i].metadata["excerpt_keywords"] = clean_text(nodes[i].metadata["excerpt_keywords"])

splitted_documents = [Document(text=node.text, metadata={"course": node.metadata["course"], "year": node.metadata["year"],"excerpt_keywords": node.metadata["excerpt_keywords"], "questions_this_excerpt_can_answer": node.metadata["questions_this_excerpt_can_answer"]}) for node in nodes]

docstore = SimpleDocumentStore()
docstore.add_documents(splitted_documents)

storage_context = StorageContext.from_defaults(
    docstore=docstore
)

index = VectorStoreIndex(splitted_documents, storage_context=storage_context)

index.storage_context.persist("database/")

pinecone_index = pinecone.Index(host="https://agent-quoor8m.svc.aped-4627-b74a.pinecone.io", api_key= os.environ["PINECONE_API_KEY"])
storage_context_pinecone = StorageContext.from_defaults(
    vector_store=PineconeVectorStore(pinecone_index)
)
index = VectorStoreIndex.from_documents(splitted_documents, storage_context=storage_context_pinecone, embed_model= embbeding_model)