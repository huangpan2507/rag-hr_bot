from langchain_community.chat_models import ChatZhipuAI
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers import BM25Retriever, EnsembleRetriever
# from langchain_community.retrievers import BM25Retriever, EnsembleRetriever
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.storage import LocalFileStore
from langchain.storage._lc_store import create_kv_docstore
from langchain.retrievers import MultiVectorRetriever
from feedback_handler import FeedBack
from langchain.memory import ConversationBufferWindowMemory
from operator import itemgetter
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from langchain_elasticsearch import ElasticsearchStore, BM25Strategy
import math
import asyncio
import jieba, json, pdfplumber
import os

class LoadEnsembleRetriever:
    def __init__(self):
        self._persist_directory = '/mnt/AI/data_base/vector_db/chroma_partition_summary_v1.8_only_two_file_by_title_qwen2_by_bgeM3'
        self._doc_directory = '/mnt/AI/data_base/vector_db/chroma_doc_v1.8_only_two_file_by_title_by_qwen2_by_bgeM3' 
        self._embedding_model = '/mnt/AI/models/embedding_model/bge-m3/'
    
    def load_docstore(self):
        # 加载文档存储
        fs = LocalFileStore(self._doc_directory)
        store = create_kv_docstore(fs)
        return store
    
    def fetch_bm25_vectorizer(self, retriever):
        # 通过访问 retriever.vectorizer 获取 BM25Okapi 对象
        bm25_vectorizer = retriever.vectorizer
        return bm25_vectorizer
    
    def create_elasticsearch_retriever(self):
        store = self.load_docstore()
        
        # 使用 yield_keys() 方法获取所有文档的键
        all_doc_ids = list(store.yield_keys(prefix=""))  # 空前缀获取所有键
        all_docs = store.mget(all_doc_ids)  # 获取所有文档内容
        print(f'\n  all_docs: {all_docs}\n')

        # 检查是否获取到了文档
        if not all_docs or all_docs[0] is None:
            raise ValueError("文档加载失败或格式不正确。请检查文档存储中的内容。")
        
        # 确保文档的格式是适合 BM25Retriever 的
        print(f'\n type doc.page_content: {type(all_docs[0].page_content)} \n')

        username = 'elastic'
        password = os.getenv('ELASTIC_PASSWORD')
        elasticsearch_db = ElasticsearchStore(
            es_url="http://localhost:9200",
            index_name="langchain_index",
            strategy=BM25Strategy(),
            es_user=username,
            es_password="Hp1124_29",
        )

        valid_docs = [(doc.page_content).strip()
                      for doc in all_docs if doc.page_content]

        metadatas = [doc.metadata
                      for doc in all_docs if doc.page_content]

        if not valid_docs:
            raise ValueError("有效文档为空，请确认文档内容正确。")
        
        # bm25_retriever = BM25Retriever.from_texts(texts=valid_docs, metadatas=metadatas, preprocess_func=self.preprocessing_func)
        # bm25_retriever.k = 6     # 6
        elasticsearch_db.add_texts(valid_docs, metadatas)
        elasticsearch_retriever = elasticsearch_db.as_retriever(search_kwargs={"k": 6})
        print("elasticsearch_retriever: ", elasticsearch_retriever.invoke("薪酬结构是什么？"))

        return elasticsearch_retriever, valid_docs
    
    def create_multiVector_retriever(self):
        embeddings = HuggingFaceEmbeddings(model_name=self._embedding_model)
        vectordb = Chroma(
            persist_directory= self._persist_directory,
            embedding_function=embeddings
        )

        fs = LocalFileStore(self._doc_directory)
        store = create_kv_docstore(fs)
        id_key = "doc_id"
        retriever = MultiVectorRetriever(
            vectorstore=vectordb,
            docstore=store,
            id_key=id_key,
            search_kwargs={"k": 4}
        )
        return retriever

    def create_ensembleRetriever(self, elasticsearch_retriever):
        multiVectorRetriever = self.create_multiVector_retriever()
        ensemble_retriever = EnsembleRetriever(
            retrievers=[elasticsearch_retriever, multiVectorRetriever], weights=[0.9, 0.1], c=30
        )
        return ensemble_retriever

    def fetch_relevant_documents(self, query):
        elasticsearch_retriever, valid_docs = self.create_elasticsearch_retriever()
        ensemble_retriever = self.create_ensembleRetriever(elasticsearch_retriever)
        results = ensemble_retriever.invoke(query)
        return results
    
    def preprocessing_func(self, text:str):
        return list(jieba.lcut(text.strip()))

# 使用示例
db_loader = LoadEnsembleRetriever()
# query = "员工最多可以申请多少天的全薪病假？"  #ok
# query = "如果我是国内出差到北京，工作8个小时，可以领取多少出差津贴?"
# query = "新员工入职程序有哪些?"
# query = "新员工试用期不符合录用的情形有哪些?" #ok,but the chunck is not very good
# query = "试用期评估不通过，公司可以怎么处理?"   #wrong answer
# query = "新员工不符合工作要求"                   #wrong answer
# query = "If the newcomer fails to meet the job requirements, the company will do what?"  #wrong answer
# query = "如果新员工不符合工作要求,公司将对新员工采取什么措施？"     #wrong answer
# query = "新员工在试用期内不符合招聘条件，公司将会怎么做？"            #wrong answer
# query = "新员工在试用期不符合要求怎么办？"                            #wrong answer
# query = "新员工在试用期不符合要求会有什么后果？"
# query = "新员工在试用期不合格会有什么后果？"
query = "南京员工有多少天婚假？"
# query = "标准工时指什么？"
# query = "休年假最少不得低于多少天"
# query = "年休假每次休假时间以多少天起计算？"    #ok
# query = "Minimum number of days of annual leave?"
# query = "Tietoevry公司的公章可以出借吗？"
# query = "What is the maximum number of days an employee can request for sick leave with full pay?"
relevant_documents = db_loader.fetch_relevant_documents(query)

# 打印检索结果
print(f'\n 问题: {query}')
print(f'\n -----------------打印结果如下--------------：\n')
for doc in relevant_documents:
    print(f"doc: {doc}\n")