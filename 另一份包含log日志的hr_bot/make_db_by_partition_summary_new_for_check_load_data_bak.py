from typing import Any
from pydantic import BaseModel
from unstructured.partition.pdf import partition_pdf
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.output_parsers import StrOutputParser
from langchain.storage import LocalFileStore
from langchain.storage._lc_store import create_kv_docstore
from langchain.retrievers import MultiVectorRetriever
from langchain_text_splitters import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import ParentDocumentRetriever
import base64
import os
import subprocess
import glob
import uuid
import pickle
from langchain_core.documents import Document
from langchain.chat_models import ChatOpenAI


class make_db:
    def __init__(self) -> None:
        self._pdf_path = "/mnt/AI/hr_material_test_huangpan/"
        self._img_path = "/mnt/AI/Papers/figure_test_huangpan/"  
        self._persist_directory = '/mnt/AI/data_base/vector_db/chroma_partition_summary_for_test'
        #self._persist_directory = '/mnt/AI/data_base/vector_db/chroma_new_huangpan'
        #self._pdf_path = "/mnt/AI/hr_material_all/"
        #self._img_path = "/mnt/AI/Papers/figure/"
        #self._persist_directory = '/mnt/AI/data_base/vector_db/chroma_partition_summary_all'      # text_summary + table_summary + image_summary
        #self._persist_directory = '/mnt/AI/data_base/vector_db/chroma_partition'
        #self._persist_directory = '/mnt/AI/data_base/vector_db/chroma_amy_partitionsummary'

    def make_vector_db(self, pdf_paths):
        raw_pdf_elements = []
        for file_name in pdf_paths:
            print(f"Processing PDF: {file_name}")
            # Get elements
            one_raw_pdf_elements = partition_pdf(
                filename=file_name,
                languages=["chi_sim", "eng"],
                strategy='hi_res',
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

        # # Optional: Enforce a specific token size for texts
        # text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        #     chunk_size=4000, chunk_overlap=0
        # )
        # joined_texts = " ".join(text_elements)
        # texts_4k_token = text_splitter.split_text(joined_texts)

        # data_base
        embeddings = HuggingFaceEmbeddings(model_name="/mnt/AI/models/embedding_model")

        vectorstore = Chroma(
            persist_directory=self._persist_directory,
            embedding_function=embeddings
        )

        parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
        fs = LocalFileStore("/mnt/AI/data_base/vector_db/chroma_new_huangpan")
        store = create_kv_docstore(fs)
        id_key = "doc_id"
        retriever = ParentDocumentRetriever(
                vectorstore=vectorstore,
                docstore=store,
                child_splitter=child_splitter,
                parent_splitter=parent_splitter,
            )
        # retriever.add_documents(one_raw_pdf_elements)

        # id_key = "doc_id"
        # fs = LocalFileStore("/mnt/AI/data_base/vector_db/chroma_db_filestore")
        # store = create_kv_docstore(fs)
        # retriever = MultiVectorRetriever(
        #     vectorstore=vectorstore,
        #     docstore=store,
        # )

        # Prompt
        prompt_text = """You are an assistant tasked with summarizing tables and text. \
        Give a concise summary of the table or text. Table or text chunk: {element} """
        prompt = ChatPromptTemplate.from_template(prompt_text)
        # Summary chain
        # model = ChatZhipuAI(
        #         temperature=0.5,
        #         api_key="0d22b6473af1ee55c96532e1292d9941.EoCkGgyoRdic05fT",
        #         model="glm-4"
        #     )
        model = ChatOpenAI(
            streaming=False,
            verbose=True,
            openai_api_key="EMPTY",
            openai_api_base="http://localhost:8000/v1",
            model_name="/mnt/AI/models/internlm2_chat_7b/"
        )
        summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

        # Initialize empty summaries
        text_summaries = []
        table_summaries = []

        # Add texts
        texts = [i.text for i in text_elements if i.text != ""]
        print(f"\n texts: {texts}, \n")
        text_summaries = summarize_chain.batch(texts, {"max_concurrency": 5})       # origin： texts
        doc_ids_text = [str(uuid.uuid4()) for _ in texts]
        summary_texts = [
            Document(page_content=s, metadata={id_key: doc_ids_text[i]})
            for i, s in enumerate(text_summaries)
        ]
        #vectorstore.add_documents(summary_texts)
        origin_texts = [
            Document(page_content=s, metadata={id_key: doc_ids_text[i]})
            for i, s in enumerate(texts)
        ]
        retriever.vectorstore.add_documents(summary_texts)
        retriever.docstore.mset(list(zip(doc_ids_text, origin_texts)))
        print("__________________________________________________________________")
        print("doc_ids_text:{} \n summary_texts:{}".format(doc_ids_text, summary_texts))
        print("__________________________________________________________________")
        #print(f"\n 培训提议: {retriever.get_relevant_documents("培训提议")[0].page_content}, \n")
        print(f"\n 培训提议里面详细的内容: {retriever.get_relevant_documents("培训提议")}, \n")

        # Add tables
        tables = [i.text for i in table_elements]
        print(f"\n tables: {tables}, \n")
        table_summaries = summarize_chain.batch(tables, {"max_concurrency": 5})
        doc_ids_table = [str(uuid.uuid4()) for _ in tables]
        summary_tables = [
            Document(page_content=s, metadata={id_key: doc_ids_table[i]})
            for i, s in enumerate(table_summaries)
        ]
        retriever.vectorstore.add_documents(summary_tables)
        retriever.docstore.mset(list(zip(doc_ids_table,  [Document(page_content=i.text) for i in table_elements])))
        print("__________________________________________________________________")
        print("doc_ids_table:{} \n summary_tables:{}".format(doc_ids_table, summary_tables))

        # call script to handle images. all the images' summaries files are stored in img_path.
        subprocess.run(["bash", "/mnt/AI/gen_db/imagesummary.sh", self._img_path])
        img_base64_list = []
        for img_file in sorted(os.listdir(self._img_path)):
            if img_file.endswith(".jpg"):
                img_path = os.path.join(self._img_path, img_file)
                with open(img_path, "rb") as image_file:
                     base64_image =  base64.b64encode(image_file.read()).decode("utf-8")
                img_base64_list.append(base64_image)

        figure_file_paths = glob.glob(os.path.expanduser(os.path.join(self._img_path, "*.txt")))
        # Read each file and store its content in a list
        img_summaries = []
        for file_path in figure_file_paths:
            print(file_path)
            with open(file_path, "r") as file:
                img_summaries.append(file.read())
                print("img_summaries : {}\n".format(img_summaries))


    
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
