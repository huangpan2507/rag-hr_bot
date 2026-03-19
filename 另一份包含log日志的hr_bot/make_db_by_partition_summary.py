from typing import Any
from pydantic import BaseModel
from unstructured.partition.pdf import partition_pdf
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.output_parsers import StrOutputParser
import os
import subprocess
import glob
from langchain_core.documents import Document


class make_db:
    def __init__(self) -> None:
        self._pdf_path = "/mnt/AI/hr_material/"
        self._img_path = "/mnt/AI/Papers/figure/"
        #self._persist_directory = '/mnt/AI/data_base/vector_db/chroma_partition'
        self._persist_directory = '/mnt/AI/data_base/vector_db/chroma_amy_partitionsummary'

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

        # data_base
        embeddings = HuggingFaceEmbeddings(model_name="/mnt/AI/models/embedding_model")

        vectorstore = Chroma(
            persist_directory=self._persist_directory,
            embedding_function=embeddings
        )
        # Prompt
        prompt_text = """You are an assistant tasked with summarizing tables and text. \
        Give a concise summary of the table or text. Table or text chunk: {element} """
        prompt = ChatPromptTemplate.from_template(prompt_text)
        # Summary chain
        model = ChatZhipuAI(
                temperature=0.5,
                api_key="0d22b6473af1ee55c96532e1292d9941.EoCkGgyoRdic05fT",
                model="glm-4"
            )
        summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()
        # Add texts
        texts = [i.text for i in text_elements if i.text != ""]
        text_summaries = summarize_chain.batch(texts, {"max_concurrency": 5})
        summary_texts = [
            Document(page_content=s)
            for i, s in enumerate(text_summaries)
        ]
        vectorstore.add_documents(summary_texts)

        # Add tables
        tables = [i.text for i in table_elements]
        table_summaries = summarize_chain.batch(tables, {"max_concurrency": 5})
        summary_tables = [
            Document(page_content=s)
            for i, s in enumerate(table_summaries)
        ]
        vectorstore.add_documents(summary_tables)

        # call script to handle images. all the images' summaries files are stored in img_path.
        subprocess.run(["bash", "/mnt/AI/gen_db/imagesummary.sh", self._img_path])
        figure_file_paths = glob.glob(os.path.expanduser(os.path.join(self._img_path, "*.txt")))
        # Read each file and store its content in a list
        img_summaries = []
        for file_path in figure_file_paths:
            print(file_path)
            with open(file_path, "r") as file:
                img_summaries.append(file.read())
                print("img_summaries : {}\n".format(img_summaries))

        # Add image
        vectorstore.add_texts(img_summaries)
    
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
