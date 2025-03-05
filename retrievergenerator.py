from fastapi import FastAPI
import pdfplumber
import textwrap
import torch
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import re
import json
import ast


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
RESUME_PATH = "Numair_AI_RESUME.pdf"

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

tokenizer = AutoTokenizer.from_pretrained("D:/projects/tesseract/trial/chatbot/dweepseekmodel")
model = AutoModelForCausalLM.from_pretrained("D:/projects/tesseract/trial/chatbot/dweepseekmodel")

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text + "\n"
    return text.strip()

def chunk_text(text, chunk_size=300):
    return textwrap.wrap(text, width=chunk_size)

def normalize(embeddings):
    return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

def create_faiss_index(chunks):
    embeddings = embed_model.encode(chunks)
    embeddings = normalize(embeddings)
    
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    
    return index, chunks

# Retrieve relevant chunks
def retrieve_info(question, top_k=4):
    q_embed = embed_model.encode([question])
    q_embed = normalize(q_embed)
    
    _, idx = faiss_index.search(q_embed, top_k)
    results = [stored_chunks[i] for i in idx[0]]
    
    return results


def generate_answer(retrieved_chunks, question):
    context = " ".join(retrieved_chunks)

    system_instruction = """You are an AI assistant that briefly answers the users questions using only the provided context.  
    If the answer is not found in the context.

    ### Rules:
    - If the message is just greeting, then dont use context, just reply with a greeting. for example: if user says "hello", you should reply with "hello! How can I assist you today ?".
    - Do **not** generate or infer new information outside the context.
    - If no relevant information is found, respond with: "I don't have relevant information for that."
    - do not generate whole conversation by yourself, just answer the user query. once !
    - Generate a brief answer.
    
    ### Always return the response in this strict JSON format:
    { "AI": "response" }
    """

    prompt = f"{system_instruction}\nUser: {question}\ncontext: {context}\nAI (respond strictly in the JSON format given above):"
    
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).input_ids
    attention_mask = torch.ones_like(inputs)

    
    output_ids = model.generate(inputs, attention_mask=attention_mask, max_length=1024)
    generated_answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(generated_answer)

    response_text = generated_answer.replace(prompt, "").strip()
    parsed_response = json.loads(response_text) 
    try:
        ai_answer = parsed_response["AI"]
    except:
        ai_answer = parsed_response
    print(ai_answer)
    if ai_answer:
        return ai_answer
    else:
        print(f"Failed to extract JSON from: {generated_answer}")  
        return "I don't have relevant information for that."


text = extract_text_from_pdf(RESUME_PATH)
chunks = chunk_text(text)
faiss_index, stored_chunks = create_faiss_index(chunks)


@app.get("/ask")
async def ask_question(query: str):
    print(f"question: {query}\nGenerating Response...")
    if query.lower() in ["hello", "hi"]:
        return {"question": query, "answer": "Hello! How can I assist you today?"}
    retrieved_chunks = retrieve_info(query)
    answer = generate_answer(retrieved_chunks, query)

    print(f"Returning API Response: {answer}")  # Log the response before sending
    return {"question": query, "answer": answer}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
