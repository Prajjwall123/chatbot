import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from langchain_core.callbacks import CallbackManager
from langchain_core.language_models import LLM
from langchain_core.outputs import LLMResult, Generation
from typing import Optional, List
from transformers import Pipeline

print("CUDA Available:", torch.cuda.is_available())
print("Device:", "GPU" if torch.cuda.is_available() else "CPU")

load_dotenv()

app = Flask(__name__)
CORS(app)

DB_FAISS_PATH = "faiss_database"
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" 

# Prompt template
CUSTOM_PROMPT_TEMPLATE = """
You're a mental health chatbot assistant. Help people with their mental health issues.
Use the following pieces of context to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Start the answer directly.
"""

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"} 
)

try:
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
except Exception as e:
    print(f"Error loading FAISS database: {e}")
    db = None  

print(f"Loading {MODEL_NAME} model locally...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32, 
    device_map="cpu",
)

text_generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=100, 
    temperature=0.5,
    do_sample=True, 
    device_map="auto", 
)

class LocalLLM(LLM):
    pipeline: Pipeline
    stop: Optional[List[str]] = None
    callbacks: Optional[List] = None

    def __init__(self, pipeline: Pipeline, stop: Optional[List[str]] = None, callbacks: Optional[List] = None):
        object.__setattr__(self, "pipeline", pipeline)
        object.__setattr__(self, "stop", stop)
        object.__setattr__(self, "callbacks", callbacks)

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        outputs = self.pipeline(prompt)
        return outputs[0]["generated_text"][len(prompt):].strip()

    @property
    def _llm_type(self) -> str:
        return "local-hf"

    def generate(self, prompt_strings: List[str], stop: Optional[List[str]] = None, callbacks: Optional[CallbackManager] = None, **kwargs) -> LLMResult:
        if callbacks:
            if isinstance(callbacks, CallbackManager):
                for callback in callbacks.handlers:
                    if hasattr(callback, "on_generate"):
                        callback.on_generate(prompt_strings)
            else:
                for callback in callbacks:
                    if hasattr(callback, "on_generate"):
                        callback.on_generate(prompt_strings)

        generations = []
        for prompt in prompt_strings:
            text = self._call(prompt, stop)
            generations.append([Generation(text=text)])

        return LLMResult(generations=generations)

local_llm = LocalLLM(text_generator)
prompt = PromptTemplate(template=CUSTOM_PROMPT_TEMPLATE, input_variables=["context", "question"])

qa_chain = RetrievalQA.from_chain_type(
    llm=local_llm,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 1}) if db else None,  
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    query = data.get("query", "")
    if not query:
        return jsonify({"error": "Query not provided"}), 400

    try:
        result = qa_chain.invoke({"query": query})
        answer = result["result"]
        sources = [doc.page_content for doc in result["source_documents"]]
    except Exception as e:
        print(f"Error during QA chain invocation: {e}")
        answer = "I'm sorry, I don't know the answer."
        sources = []

    return jsonify({"answer": answer, "sources": sources})

if __name__ == "__main__":
    app.run(port=5000, debug=False, threaded=True)