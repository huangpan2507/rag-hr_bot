from langchain.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import PyPDF2


class make_db:
    def __init__(self) -> None:
        self._pdf_path = "hr_material/"
        # self._persist_directory = 'data_base/vector_db/chroma_pdf_2'        
        # self._persist_directory = 'data_base/vector_db/chroma_docs'
        # self._persist_directory = 'data_base/vector_db/chroma_partition'
        self._persist_directory = 'data_base/vector_db/chroma_multip'        

    def make_vector_db(self, pdf_paths):
        documents = []
        for pdf_path in pdf_paths:
            print(f"Processing PDF: {pdf_path}")
            with open(pdf_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    text = page.extract_text()
                    document = Document(page_content=text)
                    documents.append(document)

        # data_base
        embeddings = HuggingFaceEmbeddings(model_name="/data/AI/models/embedding_model")

        vectorstore = Chroma(
            persist_directory=self._persist_directory,
            embedding_function=embeddings
        )
        
        vectorstore.add_documents(documents)
    
    def create_db(self):
        # fetch all the files
        # Collect paths of PDF files in the directory
        pdf_paths = [os.path.join(self._pdf_path, filename) for filename in os.listdir(self._pdf_path)
                    if os.path.isfile(os.path.join(self._pdf_path, filename)) and filename.lower().endswith(".pdf")]

        self.make_vector_db(pdf_paths)

# Run the make_db
if __name__ == "__main__":
    db = make_db()
    db.create_db()