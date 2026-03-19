from langchain.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter


class make_db:
    def __init__(self) -> None:
        self._pdf_path = "/mnt/AI/hr_material/"
        self._persist_directory = '/mnt/AI/data_base/vector_db/chroma_docs'
        # self._persist_directory = 'data_base/vector_db/chroma_partition'

    def make_vector_db(self, pdf_paths):
        docs = []
        for file_name in pdf_paths:
            print(f"Processing PDF: {file_name}")
            # load file
            loader = PyPDFLoader(file_name)
            docs.extend(loader.load())

        # 对文本进行分块
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=150)
        split_docs = text_splitter.split_documents(docs)
        # data_base
        embeddings = HuggingFaceEmbeddings(model_name="/mnt/AI/models/embedding_model")

        vectorstore = Chroma(
            persist_directory=self._persist_directory,
            embedding_function=embeddings
        )
        
        vectorstore.add_documents(split_docs)
    
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
