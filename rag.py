import os
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import google.generativeai as genai
from dataclasses import dataclass
from PIL import Image
import time
from ratelimit import limits, sleep_and_retry
import io
import fitz
import pytesseract

@dataclass
class Config:
    MODEL_NAME: str = "gemini-pro-vision"
    TEXT_MODEL_NAME: str = "gemini-pro"
    TEXT_EMBEDDING_MODEL_NAME: str = "embedding-001"
    DPI: int = 300

class PDFProcessor:
    @staticmethod
    def extract_text_and_images(pdf_path: str) -> List[Tuple[str, Image.Image]]:
        """Extract both text and images from PDF pages."""
        results = []
        pdf_document = fitz.open(pdf_path)
        
        for page_number in range(pdf_document.page_count):
            page = pdf_document[page_number]
            
            # Extract text directly from PDF
            text = page.get_text()
            
            # Get page image
            pix = page.get_pixmap(matrix=fitz.Matrix(Config.DPI / 72, Config.DPI / 72))
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            
            # If text extraction failed, try OCR
            if not text.strip():
                try:
                    text = pytesseract.image_to_string(img)
                except Exception as e:
                    print(f"OCR failed for page {page_number + 1}: {e}")
                    text = ""
            
            results.append((text, img))
            
        pdf_document.close()
        return results

class GeminiClient:
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("API Key is required")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(Config.MODEL_NAME)
        self.text_model = genai.GenerativeModel(Config.TEXT_MODEL_NAME)

    def analyze_page(self, text: str, image: Image.Image) -> str:
        """Analyze both text and image content of a page."""
        prompt = """Analyze this research paper page and provide a detailed summary.
        Focus on:
        1. Key findings and conclusions
        2. Important data from tables
        3. Graph interpretations
        4. Methodology details
        5. Statistical results
        
        Combine both the text content and visual elements in your analysis.
        Be precise with numerical data and scientific terminology."""
        
        try:
            # First analyze the text
            text_prompt = f"{prompt}\n\nText content:\n{text}"
            text_response = self.text_model.generate_content(text_prompt)
            
            # Then analyze the image
            image_response = self.model.generate_content([prompt, image])
            
            # Combine both analyses
            combined_analysis = f"Text Analysis:\n{text_response.text}\n\nVisual Analysis:\n{image_response.text}"
            return combined_analysis
        except Exception as e:
            print(f"Error analyzing page: {e}")
            return text if text else ""

    @sleep_and_retry
    @limits(calls=10, period=60)
    def create_embeddings(self, text: str) -> List[float]:
        try:
            embedding = genai.embed_content(
                model=Config.TEXT_EMBEDDING_MODEL_NAME,
                content=text,
                task_type="retrieval_document"
            )
            return embedding['embedding']
        except Exception as e:
            print(f"Error creating embeddings: {e}")
            return []

    def find_best_passage(self, query: str, passages: List[Dict]) -> Dict:
        try:
            query_embedding = genai.embed_content(
                model=Config.TEXT_EMBEDDING_MODEL_NAME,
                content=query,
                task_type="retrieval_query"
            )
            
            max_similarity = -1
            best_passage = None

            for passage in passages:
                similarity = np.dot(passage['embedding'], query_embedding['embedding'])
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_passage = passage

            return best_passage or {'page': 0, 'content': ""}
        except Exception as e:
            print(f"Error finding best passage: {e}")
            return {'page': 0, 'content': ""}

    def generate_answer(self, query: str, passage: Dict) -> str:
        prompt = f"""Based on the following research paper content, answer this question: {query}
        
        Content: {passage['content']}
        
        Provide a clear, accurate, and scientific answer. Include relevant numerical data and citations if present in the content."""
        
        try:
            response = self.text_model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Error generating answer: {e}")
            return f"Error generating answer: {str(e)}"

class RAGApplication:
    def __init__(self, api_key: str):
        self.client = GeminiClient(api_key)
        self.passages = []

    def process_pdf(self, pdf_path: str):
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        page_contents = PDFProcessor.extract_text_and_images(pdf_path)
        
        for i, (text, image) in enumerate(page_contents):
            content = self.client.analyze_page(text, image)
            if content:
                embedding = self.client.create_embeddings(content)
                self.passages.append({
                    'page': i + 1,
                    'content': content,
                    'embedding': embedding
                })

    def answer_questions(self, questions: List[str]) -> List[Dict[str, str]]:
        answers = []
        for question in questions:
            try:
                best_passage = self.client.find_best_passage(question, self.passages)
                answer = self.client.generate_answer(question, best_passage)
                answers.append({
                    'question': question,
                    'answer': answer,
                    'source': f"Page {best_passage['page']}"
                })
            except Exception as e:
                print(f"Error processing question: {e}")
                answers.append({
                    'question': question,
                    'answer': f"Error: {str(e)}",
                    'source': "Error"
                })
        return answers