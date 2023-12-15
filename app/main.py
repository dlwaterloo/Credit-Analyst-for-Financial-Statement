from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from .pdf_processor import process_pdf
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()


origins = [
    "http://localhost:8000",
    "https://pdf-bot-eight.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("index.html", "r") as f:
        return f.read()

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    if file.content_type != 'application/pdf':
        return {"error": "File is not a PDF."}
    return {"filename": file.filename}

@app.post("/process-pdf/")
async def process_pdf_endpoint(file: UploadFile = File(...)):
    result = process_pdf(file)
    return result
