from typing import Any
from pydantic import BaseModel
from unstructured.partition.pdf import partition_pdf
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from typing import IO, TYPE_CHECKING, Any, Optional, cast
import os
import subprocess
import glob
# from PIL import Image
# from PIL import ImageFile

# ImageFile.LOAD_TRUNCATED_IMAGES = True
# # Image.MAX_IMAGE_PIXELS = None
# Image.MAX_IMAGE_PIXELS = 3429049986648



class make_db:
    def __init__(self) -> None:
        self._pdf_path = "/mnt/AI/hr_material_test_huangpan/"
        self._img_path = "/mnt/AI/Papers/figure_test_huangpan/"
        #self._pdf_path = "/mnt/AI/hr_material_all/"
        #self._img_path = "/mnt/AI/Papers/figure/"
        #self._img_path = "/mnt/AI/Papers/figure_for_demo/"
        self._persist_directory = '/mnt/AI/data_base/vector_db/chroma_partition_all_for_demo'
        #self._persist_directory = '/mnt/AI/data_base/vector_db/chroma_docs_for_test'

    def make_vector_db(self, pdf_paths):
        raw_pdf_elements = []
        for file_name in pdf_paths:
            print(f"Processing PDF: {file_name}")
            # Get elements
            one_raw_pdf_elements = partition_pdf(
                filename=file_name,
                languages=["chinese",],
                strategy='hi_res',
                # Using pdf format to find embedded image blocks
                extract_images_in_pdf=False,       # True
                # Use layout model (YOLOX) to get bounding boxes (for tables) and find titles
                # Titles are any sub-section of the document
                infer_table_structure=True,
                # Post processing to aggregate text once we have the title
                chunking_strategy="by_title",
                # Require maximum chunk size of 4000 chars
                # Attempt to create a new chunk at 3800 chars
                # Attempt to keep chunks > 2000 chars 
                max_characters=4000,                      # 4000
                new_after_n_chars=3800,                   # 3800
                combine_text_under_n_chars=2000,          # 2000
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
        # Add texts
        texts = [i.text for i in text_elements if i.text != ""]
        vectorstore.add_texts(texts)

        # Add tables
        tables = [i.text for i in table_elements]
        vectorstore.add_texts(tables)

        # call script to handle images. all the images' summaries files are stored in img_path.
        subprocess.run(["bash", "./imagesummary.sh", self._img_path])
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
