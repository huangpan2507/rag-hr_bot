from langchain.document_loaders import UnstructuredPDFLoader
#from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.retrievers import BM25Retriever, EnsembleRetriever
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
        self._pdf_path = "/mnt/AI/hr_material_test_huangpan/"
        # # self._img_path = "/mnt/AI/Papers/figure_v1/"  
        self._img_path = "/mnt/AI/Papers/figure_test_huangpan/"
        # self._persist_directory = '/mnt/AI/data_base/vector_db/chroma_partition_summary_test_embedding'
        # self._doc_directory = '/mnt/AI/data_base/vector_db/chroma_doc_test_embedding'
        self._persist_directory = '/mnt/AI/data_base/vector_db/chroma_partition_summary_v1.8_only_two_file'
        self._doc_directory = '/mnt/AI/data_base/vector_db/chroma_doc_v1.8_only_two_file'
        
        # self._embedding_model = '/mnt/AI/models/embedding_model'
        self._llm_model = '/mnt/AI/models/internlm2_chat_7b/'
        self._embeddings = HuggingFaceEmbeddings(model_name='/mnt/AI/models/embedding_model')
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

        documents = []
        for root, dirs, files in os.walk('/mnt/AI/data_base/vector_db/chroma_doc_v1.8_only_two_file'):
            for file in files:
                # 检查文件是否没有后缀
                if '.' not in file:
                    file_path = os.path.join(root, file)
                    print(f'Reading file: {file_path}')
                    try:
                        # 打开文件并读取内容
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            print(f'\n content: {content}\n')
                            print(f'\n type content: {type(content)}\n')
                            documents.extend(content)
                    except Exception as e:
                        print(f'Error reading file {file_path}: {e}')

        print(f'\n documents: {documents}\n')
        # initialize the bm25 retriever and faiss retriever
        bm25_retriever = BM25Retriever.from_documents(
            documents
        )
        bm25_retriever.k = 2

        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, retriever], weights=[0.5, 0.5]
        )
        return ensemble_retriever

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
        # res_doc = self._retriever.invoke(query)
        print(f"\n =====3=====父文档doc store中关于 {query} 的内容: {res_doc}, \n")


# Run the make_db
if __name__ == "__main__":
    db = make_db()
    #db.create_db()
    db.serach(query = "未使用的年假能带到下一年吗？")
