import os
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings  # Новый импорт
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, UnstructuredMarkdownLoader
from config import DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDINGS_MODEL, VECTOR_DB_PATH

# class RAGProcessor:
#     def __init__(self):
#         # Инициализация с новой версией HuggingFaceEmbeddings
#         self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)
#         self.vector_db = None
        
#     def _ensure_vector_db_dir(self):
#         """Создаёт папку для векторной БД"""
#         os.makedirs(VECTOR_DB_PATH, exist_ok=True)
#         return VECTOR_DB_PATH
        
#     def load_and_process_documents(self):
#         """Загружает и обрабатывает документы"""
#         if not os.path.exists(DATA_DIR):
#             os.makedirs(DATA_DIR, exist_ok=True)
#             print(f"Папка {DATA_DIR} создана. Добавьте документы.")
#             return
        
#         db_path = self._ensure_vector_db_dir()
        
#         # Пробуем загрузить существующую БД с новым параметром безопасности
#         try:
#             if os.path.exists(db_path / "index.faiss"):
#                 self.vector_db = FAISS.load_local(
#                     db_path, 
#                     self.embeddings,
#                     allow_dangerous_deserialization=True  # Разрешаем загрузку pickle
#                 )
#                 return
#         except Exception as e:
#             print(f"Ошибка загрузки БД: {e}. Создаём новую.")
            
#         # Обработка документов
#         documents = []
#         for root, _, files in os.walk(DATA_DIR):
#             for file in files:
#                 file_path = Path(root) / file
#                 try:
#                     if file.endswith('.md'):
#                         loader = UnstructuredMarkdownLoader(str(file_path))
#                     else:
#                         loader = TextLoader(str(file_path), encoding='utf-8')
#                     documents.extend(loader.load())
#                 except Exception as e:
#                     print(f"Ошибка загрузки {file_path}: {e}")
#                     continue
        
#         if not documents:
#             print("Нет документов для обработки.")
#             return
            
#         # Разделение на чанки
#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=CHUNK_SIZE,
#             chunk_overlap=CHUNK_OVERLAP
#         )
#         chunks = text_splitter.split_documents(documents)
        
#         # Создание и сохранение БД
#         try:
#             self.vector_db = FAISS.from_documents(chunks, self.embeddings)
#             self.vector_db.save_local(db_path)
#             print(f"Данные сохранены в {db_path}")
#         except Exception as e:
#             print(f"Ошибка сохранения: {e}")
#             self.vector_db = FAISS.from_documents(chunks, self.embeddings)
    
#     def search_relevant_documents(self, query: str, k: int = 3):
#         """Поиск релевантных документов"""
#         return [doc.page_content for doc in self.vector_db.similarity_search(query, k=k)] if self.vector_db else []
    



import re
from typing import List, Dict
from langchain.docstore.document import Document


class LawTextProcessor:
    @staticmethod
    def split_by_articles(text: str, law_name: str) -> List[Dict]:
        """Разбивает текст закона на статьи с сохранением метаданных"""
        pattern = r"(Статья \d+\.?\s*.+?)(?=(Статья \d+|$))"
        articles = re.findall(pattern, text, re.DOTALL)
        
        result = []
        for article in articles:
            article_text = article[0].strip()
            article_num = re.search(r"Статья (\d+\.?\d*)", article_text).group(1)
            result.append({
                "text": article_text,
                "metadata": {
                    "law": law_name,
                    "article": article_num,
                    "source": f"{law_name} Ст. {article_num}"
                }
            })
        return result

class RAGProcessor:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)
        self.vector_db = None

    def load_and_process_documents(self):
        """Обрабатывает юридические документы с разделением по статьям"""
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)
            return

        documents = []
        for file in os.listdir(DATA_DIR):
            file_path = os.path.join(DATA_DIR, file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                # Извлекаем название закона из имени файла
                law_name = os.path.splitext(file)[0]
                
                # Разбиваем текст на статьи
                articles = LawTextProcessor.split_by_articles(text, law_name)
                
                for article in articles:
                    doc = Document(
                        page_content=article["text"],
                        metadata=article["metadata"]
                    )
                    documents.append(doc)
                    
            except Exception as e:
                print(f"Ошибка обработки файла {file}: {str(e)}")

        if documents:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP
            )
            chunks = text_splitter.split_documents(documents)
            self.vector_db = FAISS.from_documents(chunks, self.embeddings)
            self.vector_db.save_local(VECTOR_DB_PATH)

    def search_relevant_documents(self, query: str, k: int = 3) -> List[str]:
        if not self.vector_db:
            return []
        
        docs = self.vector_db.similarity_search(query, k=k)
        
        results = []
        for doc in docs:
            law = doc.metadata.get("law", "Закон")
            article = doc.metadata.get("article", "N/A")
            results.append(
                f"📖 {law}\n"
                f"📝 Статья {article}\n"
                f"📄 {doc.page_content[:300]}..."
            )
        return results