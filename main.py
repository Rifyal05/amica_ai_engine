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
SAFE_LIMIT = MAX_CTX - MAX_GEN

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
    verbose=True
)

embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embed_model)

def manage_context(system_p, history_p, user_p):
    turns = history_p.split("<start_of_turn>")
    turns = [f"<start_of_turn>{t}" for t in turns if t.strip()]
    while True:
        current_history = "".join(turns)
        full_prompt = f"{system_p}\n{current_history}\n{user_p}"
        tokens = llm.tokenize(full_prompt.encode('utf-8'))
        if len(tokens) <= SAFE_LIMIT or not turns:
            return full_prompt
        turns.pop(0)

def make_standalone(message, history):
    if not history: return message
    hist_short = "".join(history.split("<start_of_turn>")[-3:])
    prompt = f"<start_of_turn>user\nBerdasarkan history, buat 1 pertanyaan pencarian singkat: {message}\nMandiri:<end_of_turn>\n<start_of_turn>model\n"
    res = llm(prompt, max_tokens=64, stop=["<end_of_turn>"])
    return res["choices"][0]["text"].strip() # type: ignore

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

@app.post("/v1/chat/stream")
async def chat_stream(request: Request, x_amica_key: str = Header(None, alias="X-Amica-Key")):
    if SECRET_KEY and x_amica_key != SECRET_KEY: raise HTTPException(status_code=401)
    data = await request.json()
    message, history = data.get("message", ""), data.get("history", "")
    
    async def event_generator():
        greetings = ["hai", "halo", "hi", "pagi", "siang", "sore", "malam", "amica"]
        is_greeting = any(k in message.lower() for k in greetings) and len(message.split()) < 2
        rag_content, source_links = "", []
        
        if not is_greeting:
            q = make_standalone(message, history)
            scored_docs = vector_db.similarity_search_with_score(q, k=3)
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
Kamu Amica, asisten parenting profesional. ingat untuk memanggil user, gunakan Ayah/Bunda. 
ATURAN KETAT:
1. Usahakan untuk menjawab dengan singkat, padat, dan langsung ke inti.
2. JANGAN berikan link, URL, atau 'Sumber Daya Tambahan' apa pun dari imajinasimu. [URL ARE FORBIDDEN]
3. Hanya gunakan link yang ada di bagian REFERENSI di bawah.
4. Jika REFERENSI kosong atau tidak relevan dengan pertanyaan, abaikan saja dan jawab berdasarkan pengetahuanmu secara umum tanpa menyebutkan sumber.
5. DILARANG MENAMBAHKAN URL KE DALAM JAWABANMU contoh = Sumber Daya Tambahan: • https://www.bullying.org/ • https://www.childhelp.org/ (jangan tambahkan link seperti ini)
6. prioritaskan jawaban dengan teks yang ada di data referensi.
7. If it can be answered in a paragraph, answer in a paragraph.
8. jika itu salam atau tanya tentang dirimu, tambahkan konteks bahwa kamu adalah Amica asisten AI anti bullying
9. selalu tambahkan disclaimer disetiap akhir respon atau jawabanmu
10. tidak perlu memberikan url ke respon atau jawabanmu. url udah di handle sama metadata. jadi, kamu gak perlu kasih url di dalam respon jawabanmu
"""

        if rag_content:
            sys_p += f"\n\nREFERENSI:\n{rag_content}"
        
        sys_p += "<end_of_turn>"
        user_p = f"<start_of_turn>user\n{message}<end_of_turn>\n<start_of_turn>model\n"
        
        final_prompt = manage_context(sys_p, history, user_p)
        stream = llm(final_prompt, max_tokens=MAX_GEN, stream=True, stop=["<end_of_turn>"], temperature=0.3)
        
        for chunk in stream:
            token = chunk["choices"][0]["text"] # type: ignore
            yield token
            await asyncio.sleep(0)
            
        if source_links:
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
    uvicorn.run(app, host="127.0.0.1", port=7860)