import os
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

app = FastAPI(title="Local AI PoC Server")

# Carpetas
TMP_DIR = "./tmp_files"
CHROMA_DIR = "./chroma_db"

os.makedirs(TMP_DIR, exist_ok=True)
os.makedirs("static", exist_ok=True)

# Modelos
llm = ChatOllama(model="llama3.2", temperature=0.3)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

vector_store = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings
)

# Frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def get_index():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

# CHAT NORMAL
@app.post("/chat")
async def chat_endpoint(message: str = Form(...)):
    try:
        response = llm.invoke(message)
        return {"response": response.content}
    except Exception as e:
        return {"error": str(e)}

# SUBIR DOCUMENTOS
@app.post("/upload")
async def upload_endpoint(file: UploadFile = File(...)):
    try:
        file_path = os.path.join(TMP_DIR, file.filename)

        with open(file_path, "wb") as f:
            f.write(await file.read())

        # Loader según tipo
        if file.filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        else:
            loader = TextLoader(file_path)

        documents = loader.load()

        # Split
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        docs = splitter.split_documents(documents)

        # Guardar en vector DB
        vector_store.add_documents(docs)

        return {"message": f"{len(docs)} fragmentos indexados correctamente"}

    except Exception as e:
        return {"error": str(e)}

# CONSULTA RAG
@app.post("/query")
async def query_endpoint(message: str = Form(...)):
    try:
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})

        prompt = ChatPromptTemplate.from_template("""
Responde usando SOLO el contexto proporcionado.

Contexto:
{context}

Pregunta:
{question}
""")

        def format_docs(docs):
            return "\n\n".join([doc.page_content for doc in docs])

        chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        response = chain.invoke(message)

        return {"response": response}

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)