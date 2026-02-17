import os
import re
import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import warnings
import requests


warnings.filterwarnings('ignore')

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False


class DocumentAssistant:
    def __init__(
        self,
        embedding_model: str = 'paraphrase-multilingual-MiniLM-L12-v2',
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        top_k: int = 3,
        ollama_url: Optional[str] = None,
        llm_model: Optional[str] = None,
        use_mock_llm: bool = False
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.llm_model = llm_model
        self.use_mock_llm = use_mock_llm
        
        self.chunks: List[str] = []
        self.embeddings: Optional[np.ndarray] = None
        self.document_sources: List[str] = []
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            self.embedding_model = SentenceTransformer(embedding_model)
        else:
            raise ImportError("sentence-transformers не установлен. Установите: pip install sentence-transformers")
        
        self.ollama_url = ollama_url
    
    def _extract_text_from_txt(self, file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    
    def _extract_text_from_docx(self, file_path: str) -> str:
        if not DOCX_AVAILABLE:
            raise ImportError("python-docx не установлен. Установите: pip install python-docx")
        doc = Document(file_path)
        return '\n'.join([paragraph.text for paragraph in doc.paragraphs])
    
    def _extract_text_from_pdf(self, file_path: str) -> str:
        if not PDF_AVAILABLE:
            raise ImportError("PyPDF2 не установлен. Установите: pip install PyPDF2")
        text = []
        with open(file_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for page in pdf_reader.pages:
                text.append(page.extract_text() or '')
        return '\n'.join(text)
    
    def _extract_text(self, file_path: str) -> str:
        ext = Path(file_path).suffix.lower()
        if ext == '.txt':
            return self._extract_text_from_txt(file_path)
        elif ext == '.docx':
            return self._extract_text_from_docx(file_path)
        elif ext == '.pdf':
            return self._extract_text_from_pdf(file_path)
        else:
            raise ValueError(f"Неподдерживаемый формат файла: {ext}")
    
    def _split_into_chunks(self, text: str) -> List[str]:
        text = re.sub(r'\s+', ' ', text).strip()
        
        if len(text) <= self.chunk_size:
            return [text] if text else []
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            if end >= len(text):
                chunk = text[start:].strip()
                if chunk:
                    chunks.append(chunk)
                break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - self.chunk_overlap
        
        return chunks
    
    def index_documents(self, documents: List[str]) -> None:
        self.chunks = []
        self.document_sources = []
        
        for doc_path in documents:
            if not os.path.exists(doc_path):
                continue
            
            try:
                text = self._extract_text(doc_path)
                doc_chunks = self._split_into_chunks(text)
                
                for chunk in doc_chunks:
                    self.chunks.append(chunk)
                    self.document_sources.append(doc_path)
            except Exception:
                continue
        
        if self.chunks:
            self.embeddings = self.embedding_model.encode(
                self.chunks, 
                convert_to_numpy=True,
                show_progress_bar=False
            )
        else:
            self.embeddings = None
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
        b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
        return np.dot(a_norm, b_norm.T)
    
    def _find_relevant_chunks(self, query: str) -> List[Tuple[str, float]]:
        if self.embeddings is None or len(self.chunks) == 0:
            return []
        
        query_embedding = self.embedding_model.encode(
            [query], 
            convert_to_numpy=True,
            show_progress_bar=False
        )
        
        similarities = self._cosine_similarity(query_embedding, self.embeddings)[0]
        top_indices = np.argsort(similarities)[-self.top_k:][::-1]
        
        return [(self.chunks[i], float(similarities[i])) for i in top_indices]
    
    def _mock_llm_call(self, prompt: str) -> str:
        lines = prompt.split('\n')
        chunks_started = False
        chunks = []
        question = ""
        
        for line in lines:
            if 'Используй только следующие фрагменты' in line:
                chunks_started = True
                continue
            if 'Вопрос:' in line:
                question = line.replace('Вопрос:', '').strip()
                break
            if chunks_started and line.strip() and not line.startswith('---'):
                chunks.append(line.strip())
        
        if not chunks:
            return "[MOCK] Нет доступных фрагментов для ответа."
        
        response = f"[MOCK Ответ]\n\nНа основе найденных фрагментов:\n\n"
        for i, chunk in enumerate(chunks, 1):
            response += f"{i}. {chunk[:200]}{'...' if len(chunk) > 200 else ''}\n\n"
        response += f"\nВопрос: {question}\n\n"
        response += "Примечание: Это mock-ответ. Для получения реальных ответов настройте подключение к Ollama."
        
        return response
    
    def _call_llm(self, prompt: str) -> str:
        if self.use_mock_llm:
            return self._mock_llm_call(prompt)
        if not REQUESTS_AVAILABLE:
            raise ImportError("Requests не установлен. Установите: pip install requests")
        try:       
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.llm_model,
                    "prompt": prompt,
                    "system": "Вы полезный ассистент. Отвечайте только на основе предоставленных фрагментов документов.",
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "max_tokens": 1000
                    }
                },
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json()["response"]
            else:
                return f"[Ollama Error] {response.status_code}. Переключение на mock-режим...\n\n{self._mock_llm_call(prompt)}"
                
        except Exception as e:
            return f"[Ollama Connection Error] {str(e)}. Убедитесь что Ollama запущена. Переключение на mock-режим...\n\n{self._mock_llm_call(prompt)}"
    
    def answer_query(self, query: str) -> Dict:
        relevant_chunks = self._find_relevant_chunks(query)
        
        if not relevant_chunks:
            return {
                'query': query,
                'answer': 'Не найдено релевантных фрагментов в документах.',
                'chunks_used': 0,
                'chunks': [],
                'parameters': {
                    'chunk_size': self.chunk_size,
                    'chunk_overlap': self.chunk_overlap,
                    'top_k': self.top_k
                }
            }
        
        chunks_text = '\n\n---\n\n'.join([f"[{i+1}] {chunk}" for i, (chunk, _) in enumerate(relevant_chunks)])
        
        prompt = f"""Используй только следующие фрагменты документов для ответа:

            {chunks_text}

            Вопрос: {query}

            Ответ:"""
        
        answer = self._call_llm(prompt)
        
        return {
            'query': query,
            'answer': answer,
            'chunks_used': len(relevant_chunks),
            'chunks': [
                {
                    'text': chunk,
                    'similarity': score,
                    'source': self.document_sources[self.chunks.index(chunk)] if chunk in self.chunks else 'unknown'
                }
                for chunk, score in relevant_chunks
            ],
            'parameters': {
                'chunk_size': self.chunk_size,
                'chunk_overlap': self.chunk_overlap,
                'top_k': self.top_k
            }
        }
    
    def save_result(self, result: Dict, output_path: str) -> None:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    
    def get_stats(self) -> Dict:
        return {
            'total_chunks': len(self.chunks),
            'unique_documents': len(set(self.document_sources)) if self.document_sources else 0,
            'embedding_model': self.embedding_model.get_sentence_embedding_dimension() if hasattr(self.embedding_model, 'get_sentence_embedding_dimension') else 'unknown',
            'llm_model': self.llm_model if not self.use_mock_llm else 'mock',
            'parameters': {
                'chunk_size': self.chunk_size,
                'chunk_overlap': self.chunk_overlap,
                'top_k': self.top_k
            }
        }
