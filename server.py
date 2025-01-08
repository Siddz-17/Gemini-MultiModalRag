from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import Dict
import os
from rag import RAGApplication

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite's default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/analyze")
async def analyze_pdf(
    file: UploadFile,
    question: str = Form(...),
) -> Dict:
    try:
        # Save uploaded file temporarily
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Initialize RAG application
        api_key = 'AIzaSyC4ejc8TdzFx92chh4FlEXGDnQZ_rrRuAE'
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
            
        app = RAGApplication(api_key)
        
        # Process PDF and get answer
        app.process_pdf(temp_path)
        answers = app.answer_questions([question])
        
        # Clean up temporary file
        os.remove(temp_path)
        
        return answers[0] if answers else {
            "question": question,
            "answer": "No answer could be generated",
            "source": "Error processing document"
        }
        
    except Exception as e:
        return {
            "question": question,
            "answer": f"Error: {str(e)}",
            "source": "Error"
        }

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)