# from langchain.document_loaders import UnstructuredPDFLoader
# #from langchain_community.document_loaders import UnstructuredPDFLoader
# from typing import Any
# from langchain_experimental.text_splitter import SemanticChunker
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import Chroma
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain.storage import LocalFileStore
# from langchain.storage._lc_store import create_kv_docstore
# from langchain.retrievers import MultiVectorRetriever
# import base64
# import os
# import subprocess
# import glob
# import uuid
# from langchain_core.documents import Document
# from langchain.chat_models import ChatOpenAI

# # os.environ["TABLE_IMAGE_CROP_PAD"] = "2"

# class make_db:
#     def __init__(self) -> None:
#         self._pdf_path = "/mnt/AI/hr_material_test_huangpan/"
#         self._img_path = "/mnt/AI/Papers/figure_test_huangpan/"  
#         self._persist_directory = '/mnt/AI/data_base/vector_db/chroma_partition_summary_v1.8_only_two_file_by_semanticChunker'
#         self._doc_directory = '/mnt/AI/data_base/vector_db/chroma_doc_v1.8_only_two_file_by_semanticChunker'
#         # self._pdf_path = "/mnt/AI/hr_material_v1.5/" 
#         # self._img_path = "/mnt/AI/Papers/figure_v1.5/"
#         # self._persist_directory = '/mnt/AI/data_base/vector_db/chroma_partition_summary_v1.5'
#         # self._doc_directory = '/mnt/AI/data_base/vector_db/chroma_doc_v1.5'
#         self._embedding_model = '/mnt/AI/models/embedding_model'
#         self._llm_model = '/mnt/AI/models/MiniCPM-Llama3-V-2_5/'
#         # self._llm_model = '/mnt/AI/models/internlm2_chat_7b/'

#     def make_vector_db(self, pdf_paths):
#         documents = []
#         for file_name in pdf_paths:
#             print(f"\n file_name: {file_name} \n ")
#             loader = UnstructuredPDFLoader(file_path=file_name, 
#                                             mode="elements",
#                                             include_page_break=False,
#                                             infer_table_structure=True,
#                                             languages=["Eng","chi_sim"],
#                                             strategy="hi_res",
#                                             extract_images_in_pdf=True,
#                                             # Post processing to aggregate text once we have the title
#                                             extract_image_block_output_dir=self._img_path + os.path.basename(file_name),
#                                             form_extraction_skip_tables = False,
#                                             include_metadata=True)
#             document = loader.load()
#             documents.extend(document)

#         print(f"\n documents: {documents}\n")
      
#         embeddings = HuggingFaceEmbeddings(model_name=self._embedding_model)

#         text_splitter = SemanticChunker(
#             embeddings, breakpoint_threshold_type="standard_deviation"
#         )
        
#         # texts = [e.page_content.replace('\n\n', ' ') for e in documents if e.page_content != ""]
#         # print(f"\n texts: {texts}\n")
#         # print(f"\n type texts: {type(texts)}\n")
#         # 生成 texts 和 metadatas 列表时确保长度一致
#         filtered_documents = [e for e in documents if e.page_content != ""]
#         texts = [e.page_content.replace('\n\n', ' ') for e in filtered_documents]
#         metadatas = [{"title": e.metadata['filename']} for e in filtered_documents]

#         docs = text_splitter.create_documents(texts=texts, metadatas=metadatas)

#         # docs = text_splitter.create_documents(texts = [e.page_content.replace('\n\n', ' ') for e in documents if e.page_content != ""], metadatas=[{"title": e.metadata['filename'] for e in documents if e.page_content != ""}])
#         # new_documents = [
#         #     Document(page_content=e.page_content.replace('\n\n', ' '), metadata={"title": os.path.basename(file_name)})
#         #     for e in documents if e.page_content != ""
#         # ]
#         print(f"\n docs: {docs} \n ")

#         # 创建数据库
#         vectorstore = Chroma(
#             persist_directory=self._persist_directory,
#             embedding_function=embeddings
#         )
#         # 创建本地存储对象用于存放切片后的文档
#         fs = LocalFileStore(self._doc_directory)
#         store = create_kv_docstore(fs)
#         id_key = "doc_id"
#         retriever = MultiVectorRetriever(
#             vectorstore=vectorstore,
#             docstore=store,
#             id_key=id_key,
#         )


#     def create_db(self):
#         # fetch all the files
#         # Collect paths of PDF files in the directory
#         pdf_paths = [os.path.join(self._pdf_path, filename) for filename in os.listdir(self._pdf_path)
#                     if os.path.isfile(os.path.join(self._pdf_path, filename)) and filename.lower().endswith(".pdf")]
#         print(f'\n pdf_paths: {pdf_paths}\n')
#         self.make_vector_db(pdf_paths)

# # Run the make_db
# if __name__ == "__main__":
#     db = make_db()
#     db.create_db()



from langchain.document_loaders import UnstructuredPDFLoader
#from langchain_community.document_loaders import UnstructuredPDFLoader
from typing import Any
# from langchain_experimental.text_splitter import SemanticChunker
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.embeddings import HuggingFaceEmbeddings
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

# os.environ["TABLE_IMAGE_CROP_PAD"] = "2"

class make_db:
    def __init__(self) -> None:
        self._pdf_path = "/mnt/AI/hr_material_test_huangpan/"
        self._img_path = "/mnt/AI/Papers/figure_test_huangpan/"  
        self._persist_directory = '/mnt/AI/data_base/vector_db/chroma_partition_summary_v1.8_only_two_file_by_title_internlm'
        self._doc_directory = '/mnt/AI/data_base/vector_db/chroma_doc_v1.8_only_two_file_by_title_internlm'
        self._embedding_model = '/mnt/AI/models/embedding_model'
        # self._llm_model = '/mnt/AI/models/MiniCPM-Llama3-V-2_5/'
        self._llm_model = '/mnt/AI/models/internlm2_chat_7b/'

    def make_vector_db(self, pdf_paths):
        documents = []
        new_documents = []
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
                                            chunking_strategy="by_title",
                                            include_metadata=True)
            document = loader.load()
            print(f"\n document: {document}\n")
            # documents.extend(document)
            new_documents.extend([
                Document(page_content=e.page_content.replace('\n\n', ' '), metadata={"title": os.path.basename(file_name)})
                for e in document if e.page_content != ""])
 
        print(f"\n new_documents: {new_documents} \n ")
      
        embeddings = HuggingFaceEmbeddings(model_name=self._embedding_model)

        # 创建数据库
        vectorstore = Chroma(
            persist_directory=self._persist_directory,
            embedding_function=embeddings
        )
        # 创建本地存储对象用于存放切片后的文档
        fs = LocalFileStore(self._doc_directory)
        store = create_kv_docstore(fs)
        id_key = "doc_id"
        retriever = MultiVectorRetriever(
            vectorstore=vectorstore,
            docstore=store,
            id_key=id_key,
        )


        # Prompt
        prompt_text = """You are an assistant tasked with summarizing tables and text. \
        Give a concise summary of the table or text. Table or text chunk: {element} """
        prompt = ChatPromptTemplate.from_template(prompt_text)

        model = ChatOpenAI(
            streaming=False,
            verbose=True,
            openai_api_key="none",
            openai_api_base="http://localhost:8000/v1",
            model_name=self._llm_model
        )
        summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

        # Initialize empty summaries
        text_summaries = []

        # Add texts
        text_summaries = summarize_chain.batch(new_documents, {"max_concurrency": 5})
        doc_ids_text = [str(uuid.uuid4()) for _ in new_documents]

        summary_texts = [
                Document(page_content=s, metadata={id_key: doc_ids_text[i], "title": new_documents[i].metadata["title"]})
                for i, s in enumerate(text_summaries)
            ]
        print(f"\n summary_texts: {summary_texts} \n :")

        origin_texts = [
                Document(page_content=doc.page_content, metadata={id_key: doc_ids_text[i], "title": doc.metadata["title"]})
                for i, doc in enumerate(new_documents)
            ]
        
        print(f"\n origin_texts: {origin_texts} \n ")
        retriever.vectorstore.add_documents(summary_texts)
        retriever.docstore.mset(list(zip(doc_ids_text, origin_texts)))

        bm25_retriever = BM25Retriever.from_documents(
            origin_texts
        )
        bm25_retriever.k = 2

        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, retriever], weights=[0.5, 0.5]
        )
        result = ensemble_retriever.get_relevant_documents("员工个人原因给公司造成8000元损失，公司会怎么处理?")
        # result = ensemble_retriever.invoke("未使用的年假能带到下一年吗？")
        print(f'\n result: {result}\n')
    
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
