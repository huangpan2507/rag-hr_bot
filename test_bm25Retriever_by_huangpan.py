# from langchain_community.chat_models import ChatZhipuAI
# from langchain_core.output_parsers import StrOutputParser
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.retrievers import BM25Retriever, EnsembleRetriever
# # from langchain_community.retrievers import BM25Retriever, EnsembleRetriever
# from langchain_community.vectorstores import Chroma
# from langchain_core.runnables import RunnablePassthrough
# from langchain.prompts import PromptTemplate
# from langchain.chat_models import ChatOpenAI
# from langchain.storage import LocalFileStore
# from langchain.storage._lc_store import create_kv_docstore
# from langchain.retrievers import MultiVectorRetriever
# from feedback_handler import FeedBack
# from langchain.memory import ConversationBufferWindowMemory
# from operator import itemgetter
# from langchain_core.documents import Document
# import asyncio
# import jieba
# memory = ConversationBufferWindowMemory(k=5,return_messages=True)

# class LoadBM25Retriever:
#     def __init__(self):
#         self._persist_directory = '/mnt/AI/data_base/vector_db/chroma_partition_summary_v1.8_only_two_file'
#         self._doc_directory = '/mnt/AI/data_base/vector_db/chroma_doc_v1.8_only_two_file'
#         self._embedding_model = '/mnt/AI/models/embedding_model'
    
#     def load_docstore(self):
#         # 加载文档存储
#         fs = LocalFileStore(self._doc_directory)
#         store = create_kv_docstore(fs)
#         return store
    
#     def create_bm25_retriever(self):
#         store = self.load_docstore()
        
#         # 使用 yield_keys() 方法获取所有文档的键
#         all_doc_ids = list(store.yield_keys(prefix=""))  # 空前缀获取所有键
#         all_docs = store.mget(all_doc_ids)  # 获取所有文档内容
#         print(f'\n  all_docs: {all_docs}\n')

#         # 检查是否获取到了文档
#         if not all_docs or all_docs[0] is None:
#             raise ValueError("文档加载失败或格式不正确。请检查文档存储中的内容。")
        
#         # 确保文档的格式是适合 BM25Retriever 的
#         print(f'\n type doc.page_content: {type(all_docs[0].page_content)} \n')
#         valid_docs = [
#                 Document(page_content=' '.join(jieba.lcut((doc.page_content).strip())), metadata=doc.metadata)
#                 for doc in all_docs if doc.page_content]

#         # list(jieba.lcut(text.strip()))   jieba.lcut(doc.strip())


#         if not valid_docs:
#             raise ValueError("有效文档为空，请确认文档内容正确。")

#         # 打印文档信息以验证加载
#         print(f"Loaded {len(valid_docs)} documents.")
#         for doc in valid_docs[:3]:  # 仅打印前3个文档以验证
#             print(f"Document Content: {doc.page_content[:100]}")
#             print(f"Document Metadata: {doc.metadata}")

#         # 创建 BM25 检索器，传入所有文档
#         bm25_retriever = BM25Retriever.from_documents(documents=valid_docs)
#         bm25_retriever.k = 2
#         return bm25_retriever

#     def fetch_relevant_documents(self, query):
#         retriever = self.create_bm25_retriever()
#         results = retriever.invoke(query)
#         return results

# # 使用示例
# db_loader = LoadBM25Retriever()
# # query = "员工最多可以申请多少天的全薪病假？"
# query = "What is the maximum number of days an employee can request for sick leave with full pay?"
# relevant_documents = db_loader.fetch_relevant_documents(query)

# # 打印检索结果
# print(f'\n -----------------打印结果如下--------------：\n')
# for doc in relevant_documents:
#     print(f"Document Content: {doc.page_content}")
#     print(f"Document Metadata: {doc.metadata}")



# from langchain_community.chat_models import ChatZhipuAI
# from langchain_core.output_parsers import StrOutputParser
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.retrievers import BM25Retriever, EnsembleRetriever
# # from langchain_community.retrievers import BM25Retriever, EnsembleRetriever
# from langchain_community.vectorstores import Chroma
# from langchain_core.runnables import RunnablePassthrough
# from langchain.prompts import PromptTemplate
# from langchain.chat_models import ChatOpenAI
# from langchain.storage import LocalFileStore
# from langchain.storage._lc_store import create_kv_docstore
# from langchain.retrievers import MultiVectorRetriever
# from feedback_handler import FeedBack
# from langchain.memory import ConversationBufferWindowMemory
# from operator import itemgetter
# from langchain_core.documents import Document
# from rank_bm25 import BM25Okapi
# import math
# import asyncio
# import jieba, json, pdfplumber
# import os
# memory = ConversationBufferWindowMemory(k=5,return_messages=True)

# class LoadBM25Retriever:
#     def __init__(self):
#         self._persist_directory = '/mnt/AI/data_base/vector_db/chroma_partition_summary_v1.8_only_two_file'
#         self._doc_directory = '/mnt/AI/data_base/vector_db/chroma_doc_v1.8_only_two_file'
#         self._embedding_model = '/mnt/AI/models/embedding_model'
    
#     def load_docstore(self):
#         # 加载文档存储
#         fs = LocalFileStore(self._doc_directory)
#         store = create_kv_docstore(fs)
#         return store
    
#     def fetch_bm25_vectorizer(self, retriever):
#         # 通过访问 retriever.vectorizer 获取 BM25Okapi 对象
#         bm25_vectorizer = retriever.vectorizer
#         return bm25_vectorizer
    
#     def create_bm25_retriever(self):
#         store = self.load_docstore()
        
#         # 使用 yield_keys() 方法获取所有文档的键
#         all_doc_ids = list(store.yield_keys(prefix=""))  # 空前缀获取所有键
#         all_docs = store.mget(all_doc_ids)  # 获取所有文档内容
#         print(f'\n  all_docs: {all_docs}\n')

#         # 检查是否获取到了文档
#         if not all_docs or all_docs[0] is None:
#             raise ValueError("文档加载失败或格式不正确。请检查文档存储中的内容。")
        
#         # 确保文档的格式是适合 BM25Retriever 的
#         print(f'\n type doc.page_content: {type(all_docs[0].page_content)} \n')
#         valid_docs = [(doc.page_content).strip()
#                       for doc in all_docs if doc.page_content]

#         # list(jieba.lcut(text.strip()))   jieba.lcut(doc.strip())


#         if not valid_docs:
#             raise ValueError("有效文档为空，请确认文档内容正确。")

#         # 打印文档信息以验证加载
#         # print(f"Loaded {len(valid_docs)} documents.")
#         # for doc in valid_docs[:3]:  # 仅打印前3个文档以验证
#         #     print(f"Document Content: {doc.page_content[:100]}")
#         #     print(f"Document Metadata: {doc.metadata}")

#         # 创建 BM25 检索器，传入所有文档
        
#         bm25_retriever = BM25Retriever.from_texts(texts=valid_docs, preprocess_func=self.preprocessing_func)
#         bm25_retriever.k = 2

#         return bm25_retriever, valid_docs

#     def fetch_relevant_documents(self, query):
#         retriever, valid_docs = self.create_bm25_retriever()

#         tokenized_corpus = [jieba.lcut(doc) for doc in valid_docs]
#         bm25_vectorizer = self.fetch_bm25_vectorizer(retriever)

#         tokenized_query = jieba.lcut(query)
        
#         # 使用 BM25Okapi 计算相似度得分
#         scores = bm25_vectorizer.get_scores(tokenized_query)
#         print(f'\n scores of query: {scores}\n')
        
#         # 获取排名靠前的文档
#         top_n = bm25_vectorizer.get_top_n(tokenized_query, tokenized_corpus, n=5)
#         print(f'\n top_n doc: {top_n}\n')



#         # print(f'\n retriever.vectorizer: {retriever.vectorizer}\n')
#         # bm25 = retriever.vectorizer(valid_docs)
#         # tokenized_query = list(jieba.cut(query))
#         # scores = bm25.get_scores(tokenized_query)

#         # for sentence, score in zip(valid_docs, scores):
#         #     print(f"Sentence: {sentence}\nScore: {score}\n")


#         results = retriever.invoke(query)
#         return results
    
#     def preprocessing_func(self, text:str):
#         return ' '.join(list(jieba.lcut(text.strip())))
    
#     # def load_pdf_content(self, pdf_files):    
#     #     self.pdf_content = []
#     #     for file in pdf_files:
#     #             pdf = pdfplumber.open(file)
#     #             for page_idx in range(len(pdf.pages)):
#     #                 page_num = page_idx + 1
#     #                 page_key = os.path.basename(file) + '_page_' + str(page_num)
#     #                 self.pdf_content.append({
#     #                   'page': page_key,
#     #                   'content': pdf.pages[page_idx].extract_text()
#     #                 })
#     #             pdf.close()
    
# # class MyBM250Kapi(BM25Okapi):
# #     def _calc_idf(self, nd):
# #         idf_sum = 0
# #         negative_idfs = []
# #         for word, freq in nd.items():
# #             idf = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)
# #             self.idf[word] = idf
# #             idf_sum += idf
# #             if idf <= 0:
# #                 negative_idfs.append(word)
# #         self.average_idf = idf_sum / len(self.idf) 

# # 使用示例
# db_loader = LoadBM25Retriever()
# # query = "员工最多可以申请多少天的全薪病假？"
# # query = "标准工时指什么？"
# # query = "Tietoevry公司的公章可以出借吗？"
# query = "What is the maximum number of days an employee can request for sick leave with full pay?"
# relevant_documents = db_loader.fetch_relevant_documents(query)

# # 打印检索结果
# print(f'\n -----------------打印结果如下--------------：\n')
# for doc in relevant_documents:
#     print(f"doc: {doc}")



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
import math
import asyncio
import jieba, json, pdfplumber
import os
memory = ConversationBufferWindowMemory(k=5,return_messages=True)

class LoadBM25Retriever:
    def __init__(self):
        self._persist_directory = '/mnt/AI/data_base/vector_db/chroma_partition_summary_v1.8_only_two_file_by_title_qwen2_by_bgeM3'
        self._doc_directory = '/mnt/AI/data_base/vector_db/chroma_doc_v1.8_only_two_file_by_title_by_qwen2_by_bgeM3'
        
    
    def load_docstore(self):
        # 加载文档存储
        fs = LocalFileStore(self._doc_directory)
        store = create_kv_docstore(fs)
        return store
    
    def fetch_bm25_vectorizer(self, retriever):
        # 通过访问 retriever.vectorizer 获取 BM25Okapi 对象
        bm25_vectorizer = retriever.vectorizer
        return bm25_vectorizer
    
    def create_bm25_retriever(self):
        store = self.load_docstore()
        
        # 使用 yield_keys() 方法获取所有文档的键
        all_doc_ids = list(store.yield_keys(prefix=""))  # 空前缀获取所有键
        all_docs = store.mget(all_doc_ids)  # 获取所有文档内容
        # print(f'\n  all_docs: {all_docs}\n')

        # 检查是否获取到了文档
        if not all_docs or all_docs[0] is None:
            raise ValueError("文档加载失败或格式不正确。请检查文档存储中的内容。")
        
        # 确保文档的格式是适合 BM25Retriever 的
        print(f'\n type doc.page_content: {type(all_docs[0].page_content)} \n')
        valid_docs = [(doc.page_content).strip()
                      for doc in all_docs if doc.page_content]
        
        metadatas = [doc.metadata
                      for doc in all_docs if doc.page_content]


        if not valid_docs:
            raise ValueError("有效文档为空，请确认文档内容正确。")
        
        bm25_retriever = BM25Retriever.from_texts(texts=valid_docs, metadatas=metadatas, preprocess_func=self.preprocessing_func)
        bm25_retriever.k = 6

        return bm25_retriever, valid_docs

    def fetch_relevant_documents(self, query):
        retriever, valid_docs = self.create_bm25_retriever()
        results = retriever.invoke(query)
        return results
    
    def preprocessing_func(self, text:str):
        return list(jieba.lcut(text.strip()))

# 使用示例
db_loader = LoadBM25Retriever()
# query = "员工最多可以申请多少天的全薪病假？"  #ok
# query = "如果我是国内出差到北京，工作8个小时，可以领取多少出差津贴?"
# query = "新员工入职程序有哪些?"
# query = "新员工在试用期不符合要求会有什么后果？"   #ok
# query = "南京员工有多少天婚假？"  # ['南京', '员工', '有', '多少', '天', '婚假', '？']
query = "在南京上班的员工有多少天婚假？"    # ['在', '南京', '上班', '的', '员工', '有', '多少', '天', '婚假', '？']
print(f'\n {query}分词为：{jieba.lcut(query)}\n')  #分词为：['新', '员工', '在', '试用期', '不', '符合要求', '会', '有', '什么', '后果', '？']
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
    print(f"\ndoc: {doc}")
    print(f"doc title: {doc.metadata.get('title','')}\n")
