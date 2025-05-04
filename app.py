import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

HF_TOKEN = os.environ.get("HF_TOKEN")
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
DB_FAISS_PATH = "faiss_database"

CUSTOM_PROMPT_TEMPLATE = """
You're a mental health chatbot assistant. Help people with their mental health issues.
Use the following pieces of context to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Start the answer directly.
"""

# Load resources once
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
llm = HuggingFaceEndpoint(
    repo_id=HUGGINGFACE_REPO_ID,
    temperature=0.5,
    task="text-generation",
    model_kwargs={"token": HF_TOKEN, "max_length": "512"}
)

prompt = PromptTemplate(template=CUSTOM_PROMPT_TEMPLATE, input_variables=["context", "question"])
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    query = data.get("query", "")
    if not query:
        return jsonify({"error": "Query not provided"}), 400

    result = qa_chain.invoke({"query": query})
    answer = result["result"]
    sources = [doc.page_content for doc in result["source_documents"]]

    return jsonify({"answer": answer, "sources": sources})

if __name__ == "__main__":
    app.run(port=5000, debug=True)
