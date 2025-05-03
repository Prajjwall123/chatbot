import os
import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv


load_dotenv()


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

@st.cache_resource
def load_llm():
    return HuggingFaceEndpoint(
        repo_id=HUGGINGFACE_REPO_ID,
        temperature=0.5,
        task="text-generation",
        model_kwargs={"token": HF_TOKEN, "max_length": "512"}
    )

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def load_vector_store(_embedding_model):
    return FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

@st.cache_resource
def setup_qa_chain(_llm, _db):
    prompt = PromptTemplate(template=CUSTOM_PROMPT_TEMPLATE, input_variables=["context", "question"])
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

st.set_page_config(page_title="Mental Health Chatbot", layout="wide")
st.title("Your Mental Health Chatbot")

embedding_model = load_embeddings()
db = load_vector_store(embedding_model)
llm = load_llm()
qa_chain = setup_qa_chain(llm, db)

query = st.text_input("How are you feeling today? Ask me anything about your mental health:")

if query:
    with st.spinner("Thinking..."):
        result = qa_chain.invoke({"query": query})
        st.markdown("Answer:")
        st.write(result["result"])

        with st.expander("Source Documents"):
            for i, doc in enumerate(result["source_documents"]):
                st.markdown(f"**Source {i+1}:**")
                st.write(doc.page_content)
