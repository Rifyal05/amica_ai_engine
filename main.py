import os
import asyncio
import json
import requests
import datetime
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from llama_cpp import Llama
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Amica AI Engine")

SECRET_KEY = os.getenv("AMICA_API_KEY")
GROQ_KEYS = [k.strip() for k in os.getenv("GROQ_API_KEYS", "").split(",") if k.strip()]

RELEVANCE_THRESHOLD = 0.8
MAX_CTX = 8192
MAX_GEN = 1024

def log_debug(tag, message):
    now = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] [{tag}] {message}")

class GroqRotator:
    def __init__(self, keys):
        self.keys = keys
        self.current_idx = 0
    def get_key(self):
        if not self.keys: return None
        return self.keys[self.current_idx]
    def rotate(self):
        self.current_idx = (self.current_idx + 1) % len(self.keys)

groq_manager = GroqRotator(GROQ_KEYS)

llm = Llama(
    model_path="./models/gemma-3-1b-it-q4_km.gguf",
    n_ctx=MAX_CTX,
    n_threads=2,
    n_batch=1024,
    use_mmap=True,
    verbose=False
)

embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embed_model)

@app.post("/v1/ingest")
async def ingest_data(request: Request, x_amica_key: str = Header(None, alias="X-Amica-Key")):
    if SECRET_KEY and x_amica_key != SECRET_KEY: raise HTTPException(status_code=401)
    data = await request.json()
    articles = data.get("articles", [])
    docs, doc_ids = [], []
    for a in articles:
        uid = str(a['id'])
        ctype = a.get('chunk_type', 'reference')
        final_id = f"{uid}_{ctype}"
        doc_ids.append(final_id)
        docs.append(Document(
            page_content=f"TOPIK: {a['title']}\nKONTEN: {a['content']}",
            metadata={"id": uid, "title": a['title'], "chunk_type": ctype, "source_url": a.get('source_url', '')}
        ))
    if docs:
        try:
            res = vector_db.get(ids=doc_ids)
            if res and res['ids']: vector_db.delete(ids=res['ids'])
        except: pass
        vector_db.add_documents(docs, ids=doc_ids)
        return {"status": "success", "count": len(docs)}
    return {"status": "error"}

@app.post("/v1/search")
async def search_only(request: Request, x_amica_key: str = Header(None, alias="X-Amica-Key")):
    if SECRET_KEY and x_amica_key != SECRET_KEY: raise HTTPException(status_code=401)
    data = await request.json()
    query = data.get("query", "")
    docs = vector_db.similarity_search_with_score(query, k=5)
    results = []
    for doc, score in docs:
        results.append({
            "article_id": doc.metadata.get("id"),
            "title": doc.metadata.get("title"),
            "score": score
        })
    return {"results": results}

@app.post("/v1/chat/stream")
async def chat_stream(request: Request, x_amica_key: str = Header(None, alias="X-Amica-Key")):
    if SECRET_KEY and x_amica_key != SECRET_KEY: raise HTTPException(status_code=401)
    
    try:
        data = await request.json()
    except:
        raise HTTPException(status_code=400, detail="Invalid JSON")
        
    message = data.get("message", "")
    
    async def event_generator():
        if await request.is_disconnected():
            return

        greetings = ["hai", "halo", "hi", "pagi", "siang", "sore", "malam", "amica"]
        is_greeting = any(k in message.lower() for k in greetings) and len(message.split()) < 2
        rag_content, source_links = "", []
        
        if not is_greeting:
            log_debug("RAG", f"Searching: {message}")
            scored_docs = vector_db.similarity_search_with_score(message, k=3)
            seen_urls = set()
            for doc, score in scored_docs:
                if score < RELEVANCE_THRESHOLD:
                    rag_content += doc.page_content + "\n\n"
                    url = doc.metadata.get("source_url")
                    title = doc.metadata.get("title", "Referensi")
                    if url and url not in seen_urls:
                        source_links.append(f"[{title}]({url})")
                        seen_urls.add(url)

        sys_p = f"""<start_of_turn>system
Kamu adalah Amica, Asisten Edukasi Anti-Bullying. Saat ditanya siapa dirimu, jawab bahwa kamu adalah Amica asisten edukasi anti bullying bertujuan untuk memberikan edukasi anti bullying kepada Ayah/Bunda.
Tugasmu adalah memberikan dukungan dan informasi kepada orang tua (Ayah/Bunda) tentang bullying.

INSTRUKSI KHUSUS (WAJIB PATUH):
1. JAWAB DENGAN SINGKAT.
2. Berikan penjelasan bahwa kamu adalah Amica asisten edukasi anti bullying bertujuan untuk memberikan edukasi anti bullying kepada Ayah/Bunda.
3. DILARANG MENULIS LINK/URL DALAM TEKS JAWABAN. Hapus semua https:// atau www.
4. Gunakan Bahasa Indonesia yang ramah dan hangat.
5. Jika ada REFERENSI di bawah, gunakan faktanya. Jika tidak, gunakan pengetahuan umum tentang anti-bullying.
6. Akhiri dengan disclaimer bahwa anda adalah AI dan bukan pengganti professional."

CONTOH JAWABAN YANG BENAR:
"Halo Bunda, tanda bullying bisa berupa perubahan sikap mendadak atau enggan ke sekolah. Coba ajak bicara pelan-pelan saat santai. Tetap dampingi buah hati ya, Ayah/Bunda."
"""

        if rag_content:
            sys_p += f"\n\nDATA REFERENSI:\n{rag_content}"
        
        sys_p += "<end_of_turn>"
        final_prompt = f"{sys_p}\n<start_of_turn>user\n{message}. system : 'Jangan memberikan jawaban yang sangat panjang kalau gak diminta'<end_of_turn>\n<start_of_turn>model\n"
        
        stream = llm(final_prompt, max_tokens=MAX_GEN, stream=True, stop=["<end_of_turn>"], temperature=0.2)
        
        for chunk in stream:
            if await request.is_disconnected():
                break 
                
            token = chunk["choices"][0]["text"] # type: ignore
            yield token
            
            await asyncio.sleep(0.01)
            
        if source_links and not await request.is_disconnected():
            yield "\n\n📚 **Bacaan terkait:** " + ", ".join(source_links)
            
    return StreamingResponse(event_generator(), media_type="text/plain")

@app.post("/v1/audit/grade")
async def audit_grade(request: Request, x_amica_key: str = Header(None, alias="X-Amica-Key")):
    if SECRET_KEY and x_amica_key != SECRET_KEY: raise HTTPException(status_code=401)
    data = await request.json()
    for _ in range(len(GROQ_KEYS)):
        k = groq_manager.get_key()
        try:
            res = requests.post("https://api.groq.com/openai/v1/chat/completions", headers={"Authorization": f"Bearer {k}"}, json={"model": "llama-3.3-70b-versatile", "messages": [{"role": "user", "content": f"Grade accuracy: {data}"}], "response_format": {"type": "json_object"}}, timeout=15)
            if res.status_code == 200: return res.json()
            elif res.status_code == 429: groq_manager.rotate()
        except: groq_manager.rotate()
    raise HTTPException(status_code=503)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=7860, log_level="info")