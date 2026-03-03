# from langchain.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders import UnstructuredPDFLoader
from unstructured.chunking.basic import chunk_elements
from typing import Iterable, Optional
from unstructured.documents.elements import Table, Title
from unstructured.documents.elements import Element
from typing import Any
from pydantic import BaseModel
# from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.storage import LocalFileStore
from langchain.storage._lc_store import create_kv_docstore
from langchain.retrievers import MultiVectorRetriever
import base64
import os
import subprocess
import glob
import uuid
from langchain_core.documents import Document
from langchain.chat_models import ChatOpenAI
from unstructured.partition.auto import partition

# os.environ["TABLE_IMAGE_CROP_PAD"] = "2"

class make_db:
    def __init__(self) -> None:
        self._pdf_path = "/mnt/AI/hr_material_test_huangpan/"
        self._img_path = "/mnt/AI/Papers/figure_test_huangpan/"  
        # self._persist_directory = '/mnt/AI/data_base/vector_db/chroma_partition_summary_v1.8_only_two_file'
        # self._doc_directory = '/mnt/AI/data_base/vector_db/chroma_doc_v1.8_only_two_file'
        # self._pdf_path = "/mnt/AI/hr_material_v1.5/" 
        # self._img_path = "/mnt/AI/Papers/figure_v1.5/"
        # self._persist_directory = '/mnt/AI/data_base/vector_db/chroma_partition_summary_v1.5'
        # self._doc_directory = '/mnt/AI/data_base/vector_db/chroma_doc_v1.5'
        self._embedding_model = '/mnt/AI/models/embedding_model'
        self._llm_model = '/mnt/AI/models/internlm2_chat_7b/'

    def make_vector_db(self, pdf_paths):
        documents = []
        for file_name in pdf_paths:
            print(f"\n file_name: {file_name} \n ")
            loader = UnstructuredPDFLoader(file_path=file_name, 
                                            mode="elements",
                                            include_page_break=False,
                                            infer_table_structure=True,
                                            languages=["Eng","chi_sim"],
                                            strategy="hi_res",
                                            extract_images_in_pdf=True,
                                            # Post processing to aggregate text once we have the title
                                            extract_image_block_output_dir=self._img_path + os.path.basename(file_name),
                                            form_extraction_skip_tables = False,
                                            include_metadata=True)
            document = loader.load()
            documents.extend(document)

        # elements = partition(filename="/mnt/AI/Papers/figure_test_huangpan/Finance-Guidance-China.pdf")
        # print(f"\n documents: {elements}\n")

        print(f"\n documents: {documents}\n")

        # chunks = chunk_elements(self.combine_title_elements(elements))
        # new_documents = [
        #         e
        #         for e in chunks if e.page_content != ""
        # ]
        # print(f"\n new_documents: {new_documents} \n ")
        
        elememts = self.combine_title_elements(documents)
        print(f"\n elememts: {self.combine_title_elements(documents)} \n ")
        
        new_documents = [
                Document(page_content=e.page_content.replace('\n\n', ' '), metadata={"title": os.path.basename(file_name)})
                for e in elememts if e.page_content != ""
        ]
        print(f"\n new_documents: {new_documents} \n ")


    class Element(BaseModel):
        type: str
        text: Any

    def combine_title_elements(self, elements: Iterable[Element]):
        title = None
        for e in elements:
            print(f'\n e: {e} \n')
            print(f'\n e.category: {type(e.metadata["category"])} \n')
            # -- case where Title immediately follows a Title --
            if (e.metadata["category"]== "Title"):
                if title:
                    yield title
                title = e
            # -- case when prior element was a title --
            elif title:
                yield self.combine_title_with_element(title, e)
                title = None
            # -- "normal" case when prior element was not a title --
            else:
                yield e

        # -- handle case when last element is a Title --
        if title:
            yield title

    def combine_title_with_element(self, title_element: Title, next_element: Element) -> Element:
        next_element.page_content = f"{title_element.page_content} {next_element.page_content}".strip()
        return next_element
    
    def combine_item_with_parent_id(self, element: list[Element]) -> Element:
        new_element = [Document(page_content='', metadata={"title": ''})]
        new_document = []
        for i in range(len(element) - 1):
            current_element = element[i]
            next_element = element[i + 1]
            if current_element.metadata["parent_id"] == next_element.metadata["parent_id"]:
                new_element.page_content = f"{current_element.page_content} {next_element.page_content}".strip()
            new_element.page_content = f"{current_element.page_content}".strip()
            new_document.extend(new_element)
        return new_document
    
    def create_db(self):
        # fetch all the files
        # Collect paths of PDF files in the directory
        pdf_paths = [os.path.join(self._pdf_path, filename) for filename in os.listdir(self._pdf_path)
                    if os.path.isfile(os.path.join(self._pdf_path, filename)) and filename.lower().endswith(".pdf")]
        print(f'\n pdf_paths: {pdf_paths}\n')
        self.make_vector_db(pdf_paths)

# Run the make_db
if __name__ == "__main__":
    db = make_db()
    db.create_db()

