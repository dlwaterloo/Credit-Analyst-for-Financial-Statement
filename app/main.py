from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from .pdf_processor import process_pdf

app = FastAPI()

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
