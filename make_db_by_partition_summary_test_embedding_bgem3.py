from langchain.document_loaders import UnstructuredPDFLoader
#from langchain_community.document_loaders import UnstructuredPDFLoader
from typing import Any
from pydantic import BaseModel
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
from pprint import pprint


class make_db:
    def __init__(self) -> None:
        # self._pdf_path = "/mnt/AI/hr_material_v1/"
        self._pdf_path = "/mnt/AI/hr_material_all/"
        # # self._img_path = "/mnt/AI/Papers/figure_v1/"  
        self._img_path = "/mnt/AI/Papers/figure_v1.5/"
        self._persist_directory = '/mnt/AI/data_base/vector_db/chroma_partition_summary_test_embedding_bge'
        self._doc_directory = '/mnt/AI/data_base/vector_db/chroma_doc_test_embedding_bge'
        # self._embedding_model = '/mnt/AI/models/embedding_model'
        self._llm_model = '/mnt/AI/models/internlm2_chat_7b/'
        # self._embeddings = HuggingFaceEmbeddings(model_name='BAAI/bge-m3')
        self._embeddings = HuggingFaceEmbeddings(model_name='/mnt/AI/models/bge-m3')
        self._retriever = self._db_init()

    def _db_init(self):
        # 创建数据库
        vectorstore = Chroma(
            persist_directory=self._persist_directory,
            embedding_function=self._embeddings
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
        return retriever

    def serach(self, query): 
        print("-------------------Begin to search the relevance---------------------")
        #print(f"\n summary文档中关于 {query} 的内容及其得分：{self._retriever.vectorstore.similarity_search_with_relevance_scores(query)} \n")
        
        # 使用文本进行语义相似度搜索
        # docs = self._retriever.vectorstore.similarity_search_with_relevance_scores(query)
        # print(f"\n =====1=====vectorstore.asimilarity_search_with_relevance_scores 文档中关于 {query} 的内容：{docs} \n")

        # # 使用嵌入向量进行语义相似度搜索
        # embedding_vector = self._embeddings.embed_query(query)
        # out_about_query = self._retriever.vectorstore.similarity_search_by_vector(embedding_vector)
        # print(f"\n =====2=====summary文档中关于 {query} 向量化后的所有相关文档：{out_about_query} \n")

        res_doc = self._retriever.get_relevant_documents(query)
        print(f"\n =====3=====父文档doc store中关于 {query} 的内容: {res_doc}, \n")


    def make_vector_db(self, pdf_paths):
        print("--------------Begin to read files-----------------")
        for file_name in pdf_paths:
            loader = UnstructuredPDFLoader(file_path=file_name, 
                                            mode="elements",
                                            include_page_break=False,
                                            infer_table_structure=True,
                                            languages=["Eng","chi_sim"],
                                            strategy="hi_res",
                                            extract_images_in_pdf=False,
                                            # Post processing to aggregate text once we have the title
                                            #chunking_strategy="by_title",   # 不能使用chunking_strategy="by_title，否则输出内容将不含有坐标信息，下面的坐标代码将会报错
                                            #extract_image_block_output_dir=self._img_path,
                                            form_extraction_skip_tables = False,
                                            include_metadata=True,
                                            max_characters=3500,
                                            new_after_n_chars=1500,
                                            combine_text_under_n_chars=250)
            documents = loader.load()
        
        new_documents = [
                Document(page_content=e.page_content.replace('\n\n', ' '), metadata={"title": os.path.basename(file_name)})
                for e in documents if e.page_content != ""
        ]
        print(f"\n new_documents: {new_documents} \n ")

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
            model_name=self._llm_model
        )
        summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

        # Initialize empty summaries
        text_summaries = []

        # Add texts
        text_summaries = summarize_chain.batch(new_documents, {"max_concurrency": 5})
        doc_ids_text = [str(uuid.uuid4()) for _ in new_documents]
        # summary_texts = [
        #     Document(page_content=s, metadata={id_key: doc_ids_text[i]})
        #     for i, s in enumerate(text_summaries)
        # ]
        # add by jianfeng
        id_key = "doc_id"
        summary_texts = [
                Document(page_content=s, metadata={id_key: doc_ids_text[i], "title": new_documents[i].metadata["title"]})
                for i, s in enumerate(text_summaries)
            ]
        
        print(f"\n summary_texts: {summary_texts} \n :")

        #vectorstore.add_documents(summary_texts)
        # origin_texts = [
        #     Document(page_content=s, metadata={id_key: doc_ids_text[i]})
        #     for i, s in enumerate(new_documents)
        # ]
        # add by jianfeng
        origin_texts = [
                Document(page_content=doc.page_content, metadata={id_key: doc_ids_text[i], "title": doc.metadata["title"]})
                for i, doc in enumerate(new_documents)
            ]
        
        print(f"\n origin_texts: {origin_texts} \n ")
        self._retriever.vectorstore.add_documents(summary_texts)
        self._retriever.docstore.mset(list(zip(doc_ids_text, origin_texts)))


    def create_db(self):
        # fetch all the files
        # Collect paths of PDF files in the directory
        pdf_paths = [os.path.join(self._pdf_path, filename) for filename in os.listdir(self._pdf_path)
                    if os.path.isfile(os.path.join(self._pdf_path, filename)) and filename.lower().endswith(".pdf")]

         # only choose the Remote-work-Guidance.pdf
        Remote_Work_Guidance_file_name = 'mobile'
        pdf_paths = [e for e in pdf_paths if Remote_Work_Guidance_file_name in e]
        
        # Travel_rule = 'Travel rule_v1_1'
        # pdf_paths = [e for e in pdf_paths if Remote_Work_Guidance_file_name in e or Finance_Guidance_China in e or Travel_rule in e]
        self.make_vector_db(pdf_paths)

# Run the make_db
if __name__ == "__main__":
    db = make_db()
    #db.create_db()
    db.serach(query = "每月工作2天，有通讯补贴吗")
