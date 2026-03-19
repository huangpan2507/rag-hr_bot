from typing import Any
from langchain_community.chat_models import ChatOllama, ChatZhipuAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from unstructured.partition.pdf import partition_pdf
from langchain_core.documents import Document
from langchain.document_loaders import PyPDFLoader
import uuid
from langchain.storage import LocalFileStore
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_community.embeddings import GPT4AllEmbeddings, HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import os
import subprocess
import glob


class make_db:
    def __init__(self) -> None:
        self._pdf_path = "hr_material/"
        self._img_path = "Papers/figure/"
        self._persist_directory = 'data_base/vector_db/chroma'
        
        # data_base
        embeddings = HuggingFaceEmbeddings(model_name="/data/AI/models/embedding_model")
        self._vectorstore = Chroma(
            persist_directory=self._persist_directory,
            embedding_function=embeddings
        )

    def make_vector_db(self, pdf_paths):
        raw_pdf_elements = []
        for file_name in pdf_paths:
            print(f"Processing PDF: {file_name}")
            # Get elements
            one_raw_pdf_elements = partition_pdf(
                filename=file_name,
                # strategy='hi_res',
                # Using pdf format to find embedded image blocks
                extract_images_in_pdf=True,
                # Use layout model (YOLOX) to get bounding boxes (for tables) and find titles
                # Titles are any sub-section of the document
                infer_table_structure=True,
                # Post processing to aggregate text once we have the title
                chunking_strategy="by_title",
                extract_image_block_output_dir=self._img_path,
                form_extraction_skip_tables = False
            )
            raw_pdf_elements.extend(one_raw_pdf_elements)
            # load file
            loader = PyPDFLoader(file_name)

        class Element(BaseModel):
            type: str
            text: Any

        # Categorize by type
        categorized_elements = []
        for element in raw_pdf_elements:
            if "unstructured.documents.elements.Table" in str(type(element)):
                categorized_elements.append(Element(type="table", text=str(element)))
            elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
                categorized_elements.append(Element(type="text", text=str(element)))

        # Tables
        table_elements = [e for e in categorized_elements if e.type == "table"]

        # Text
        text_elements = [e for e in categorized_elements if e.type == "text"]

        # Add texts
        texts = [i.text for i in text_elements if i.text != ""]
        self._vectorstore.add_texts(texts)
        # todo: 
        
        self._vectorstore.add_documents()
        # Add tables
        tables = [i.text for i in table_elements]
        self._vectorstore.add_texts(tables)
        # call script to handle images. all the images' summaries files are stored in img_path.
        subprocess.run(["bash", "./imagesummary.sh"])
        figure_file_paths = glob.glob(os.path.expanduser(os.path.join(self._img_path, "*.txt")))
        # Read each file and store its content in a list
        img_summaries = []
        for file_path in figure_file_paths:
            print(file_path)
            with open(file_path, "r") as file:
                img_summaries.append(file.read())
                print("img_summaries : {}\n".format(img_summaries))

        # Add image
        self._vectorstore.add_texts(img_summaries)
    
    def create_db(self):
        # fetch all the files
        # Collect paths of PDF files in the directory
        pdf_paths = [os.path.join(self._pdf_path, filename) for filename in os.listdir(self._pdf_path)
                    if os.path.isfile(os.path.join(self._pdf_path, filename)) and filename.lower().endswith(".pdf")]

        self.make_vector_db(pdf_paths)

# Run the chatbot
if __name__ == "__main__":
    db = make_db()
    db.create_db()