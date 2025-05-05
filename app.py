import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from langchain_core.language_models import LLM
from langchain_core.outputs import LLMResult, Generation
from typing import Optional, List

print("CUDA Available:", torch.cuda.is_available())
print("Device:", "GPU" if torch.cuda.is_available() else "CPU")

app = Flask(__name__)
CORS(app)

DB_FAISS_PATH = "faiss_database"
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

CUSTOM_PROMPT_TEMPLATE = """
<|im_start|>system
You are a helpful and supportive mental health assistant. Your primary goal is to provide empathetic, non-judgmental, and safe support. Always:
- If the user greets you with 'hi', 'hello', or similar phrases, respond exactly with 'Hi, how are you feeling today?' and nothing else.
- Avoid giving advice that could be harmful or misinterpreted, especially on sensitive topics like suicide or self-harm.
- If the user mentions suicide, self-harm, or crisis, respond with empathy and direct them to professional help (e.g., 'I'm here for you. Please contact a helpline like 1166 or a local crisis service.').
- If the user asks for examples (e.g., how to validate feelings or avoid harmful advice), provide concise examples such as: 
  - To validate feelings: 'It sounds like you're going through a tough time, and that’s okay to feel that way.'
  - To avoid harmful advice: Instead of suggesting actions, say 'I’m here to listen and support you.'
- Keep responses concise, positive, and focused on support. Avoid formal sign-offs or including a name. Ensure responses are complete and do not cut off mid-sentence, especially when providing lists.
<|im_end|>
<|im_start|>user
{context}
{question}
<|im_end|>
<|im_start|>assistant
"""

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

try:
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    print("FAISS database loaded successfully.")
except Exception as e:
    print(f"Error loading FAISS database: {e}")
    db = None

class LocalLLM(LLM):
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    max_new_tokens: int = 200
    temperature: float = 0.5
    do_sample: bool = True
    device: str = "cpu"
    eos_token_id: Optional[int] = None

    def __init__(self, model_name, max_new_tokens=200, temperature=0.5, do_sample=True, device="cpu"):
        object.__setattr__(self, "tokenizer", AutoTokenizer.from_pretrained(model_name))
        object.__setattr__(self, "model", AutoModelForCausalLM.from_pretrained(model_name).to(device))
        object.__setattr__(self, "max_new_tokens", max_new_tokens)
        object.__setattr__(self, "temperature", temperature)
        object.__setattr__(self, "do_sample", do_sample)
        object.__setattr__(self, "device", device)
        if "<|im_end|>" in self.tokenizer.special_tokens_map:
            object.__setattr__(self, "eos_token_id", self.tokenizer.encode("<|im_end|>")[0])
        else:
            object.__setattr__(self, "eos_token_id", None)

    def _call(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        generate_ids = self.model.generate(
            inputs.input_ids,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=self.do_sample,
            eos_token_id=self.eos_token_id
        )
        input_len = inputs.input_ids.shape[1]
        generated_ids = generate_ids[0, input_len:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        truncate_markers = ["<|assistant|>", "<|user|>"]
        truncate_pos = min((generated_text.find(marker) for marker in truncate_markers if generated_text.find(marker) != -1), default=len(generated_text))
        if truncate_pos != len(generated_text):
            generated_text = generated_text[:truncate_pos].strip()
        
        generated_text = self._ensure_complete_sentences(generated_text)
        return generated_text.strip()

    def _ensure_complete_sentences(self, text: str) -> str:
        sentences = text.split('. ')
        seen = set()
        unique_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and sentence not in seen:
                seen.add(sentence)
                unique_sentences.append(sentence)
        
        result = '. '.join(unique_sentences)
        if unique_sentences and not unique_sentences[-1].endswith('.'):
            if any(unique_sentences[-1].startswith(str(i)) for i in range(1, 10)):
                return result 
        return result + ('.' if unique_sentences and result else '')

    @property
    def _llm_type(self) -> str:
        return "local-hf"

    def generate(self, prompt_strings: List[str], stop: Optional[List[str]] = None, **kwargs) -> LLMResult:
        generations = []
        for prompt in prompt_strings:
            text = self._call(prompt)
            generations.append([Generation(text=text)])
        return LLMResult(generations=generations)

local_llm = LocalLLM(model_name=MODEL_NAME, device="cpu")
prompt = PromptTemplate(
    template=CUSTOM_PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)

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
    query = data.get("query", "").lower()
    if not query:
        return jsonify({"error": "Query not provided"}), 400

    crisis_keywords = ["suicide", "self-harm", "kill myself", "end my life"]
    if any(keyword in query for keyword in crisis_keywords):
        safe_response = (
            "I'm so sorry you're feeling this way. I'm here for you. Please contact a helpline like 1166. "
            "or a local crisis service for immediate support. You are not alone."
        )
        return jsonify({"answer": safe_response, "sources": []})

    try:
        result = qa_chain.invoke({"query": query})
        answer = result["result"]
        sources = [doc.page_content for doc in result["source_documents"]]
        return jsonify({"answer": answer, "sources": sources})
    except Exception as e:
        print(f"Error during QA chain invocation: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("Starting Flask server...")
    app.run(debug=False)