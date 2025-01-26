# UIT Course Agent


> Smarter , Flexible, State-Aware Conversation Agent for Your Quick UIT Course information

## System Features

- **Small Talk Detection**: The system performs a "Small Talk Check" to determine if the query is casual conversation. If identified, it routes the query to **Gemini** for appropriate handling.

- **Dynamic Query Transformation**: For more complex queries, the system uses **HyDE Query Transformation** to reformulate the query for better search performance.

- **Hybrid Search Integration**: Combines **BM25** (traditional keyword-based search) and **Semantic Search** to deliver accurate and contextually relevant results.

- **Metadata Filtering and Retrieval**: Queries that require historical context are routed through metadata filtering to ensure precise and relevant data retrieval.

- **HuggingFace LLM Integration**: Leverages the **HuggingFace Language Model** for natural language understanding and response generation, ensuring high-quality answers.

- **Query Re-ranking**: Uses a reranker mechanism to optimize the order of search results based on relevance and context before providing a final response.

- **Multi-Stage Query Pipeline**: The system seamlessly switches between different modules (Gemini, retrieval, and reranking) based on query type to ensure efficient and accurate responses.

- **End-to-End Query Resolution**: Designed to handle queries from initial input to final answer delivery, with built-in scalability and modularity.

---

This system is designed to process a wide range of queries with precision and efficiency, ensuring robust performance and adaptability.


## Tech Stack

- [Pinecone](https://www.pinecone.io/): Pinecone is a vector database designed for building scalable machine learning applications with high-dimensional data. We utilized Pinecone for efficient similarity search and recommendation systems.

- [LlamaIndex](https://llamalabs.io/llamaindex): LlamaIndex is an API for querying real-time financial data and market indices. We integrated LlamaIndex to provide up-to-date financial information in our applications.

- [Hugging Face](https://huggingface.co/): Hugging Face provides state-of-the-art natural language processing (NLP) models and libraries. We leveraged Hugging Face's transformers for advanced NLP tasks like sentiment analysis and text generation.

- [Gemini](https://gemini.com/): Gemini is a cryptocurrency exchange and custodian that allows users to buy, sell, and store various cryptocurrencies. We integrated Gemini's API to enable cryptocurrency trading functionalities within our platform.

- [Cohere](https://cohere.ai/): Cohere offers tools for building conversational AI applications. We utilized Cohere's natural language understanding (NLU) capabilities to enhance our chatbot's ability to comprehend and respond to user queries.

- [Flask](https://flask.palletsprojects.com/): Flask is a lightweight web framework for Python. We employed Flask to build RESTful APIs and backend services that power our applications.

- [ReactJS](https://reactjs.org/): ReactJS is a JavaScript library for building user interfaces. We developed our frontend using ReactJS to create interactive and responsive UI components, ensuring a seamless user experience.

---

By integrating these technologies into our development stack, we've built a robust and scalable application ecosystem that meets our AI-driven engineering goals effectively.

## System architecture review
![system_2 drawio](https://github.com/user-attachments/assets/5a66aff1-94d4-4a37-9cd8-b594ddbeea28)

## How to set up
This guide will walk you through the process of setting up the Karna Chatbot on your local machine. Please make sure that you have the following prerequisites installed before proceeding:

- Node.js (v14 or higher)
- ReactJS (v18 or higher)

**Step 1: Clone the repository**

```sh
git clone git@github.com:NBTailee/Agent-UIT-Course.git
```

**Step 2: Setup and get API Credentials**
- Follow this doc to set up Gemini access api on: [Gemini API](https://ai.google.dev/gemini-api/docs/api-key)
- Follow this doc to set up CohereAI access api on: [CohereAI API](https://docs.cohere.com/)
- Follow this doc to set up HuggingFace access api on: [HuggingFace](https://huggingface.co/docs/hub/security-tokens)
- Follow this doc to set up Pinecone access api on: [Pinecone API](https://docs.pinecone.io/guides/get-started/overview)
  
**Step 3: Setup Env File**

Create a `.env` file in the root directory of the project and add the following environment variables:

```sh
GEMINI_TOKEN = 
PINECONE_API_KEY = 
OPEN_AI_KEY = 
HF_TOKEN = 
COHERE_API_TOKEN = 

```

## Contribution
We welcome contributions to this project. If you find a bug or would like to suggest a new feature, please create a pull request or submit an issue. Before submitting your pull request, make sure to run the tests and ensure that they pass. We also ask that you follow our coding guidelines.


## Issues
If you encounter any issues while using this project, please create a new issue on our GitHub repository. We will do our best to address the issue as soon as possible. When creating an issue, please provide as much detail as possible, including steps to reproduce the issue and any error messages you received.
