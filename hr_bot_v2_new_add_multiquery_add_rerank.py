from langchain_community.chat_models import ChatZhipuAI
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers import BM25Retriever, EnsembleRetriever
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
from langchain.prompts import ChatPromptTemplate
from langchain.load import dumps, loads
from FlagEmbedding import FlagReranker
import asyncio
import re

memory = ConversationBufferWindowMemory(k=5,return_messages=True)

class HR_BOT:
    def __init__(self) -> None:
        # self._persist_directory = '/mnt/AI/data_base/vector_db/chroma_partition_summary_v1.5'
        # self._doc_directory = '/mnt/AI/data_base/vector_db/chroma_doc_v1.5'
        # self._persist_directory = '/mnt/AI/data_base/vector_db/chroma_partition_summary_v1.8_only_two_file'
        # self._doc_directory = '/mnt/AI/data_base/vector_db/chroma_doc_v1.8_only_two_file'
        # self._persist_directory = '/mnt/AI/data_base/vector_db/chroma_partition_summary_v1.8_only_two_file_by_title_internlm'
        # self._doc_directory = '/mnt/AI/data_base/vector_db/chroma_doc_v1.8_only_two_file_by_title_internlm'
        # self._persist_directory = '/mnt/AI/data_base/vector_db/chroma_partition_summary_v1.8_only_two_file_by_basic'
        # self._doc_directory = '/mnt/AI/data_base/vector_db/chroma_doc_v1.8_only_two_file_by_basic'
        # self._persist_directory = '/mnt/AI/data_base/vector_db/chroma_partition_summary_v1.8_only_two_file_by_basic_qwen2'
        # self._doc_directory = '/mnt/AI/data_base/vector_db/chroma_doc_v1.8_only_two_file_by_basic_by_qwen2'
        # self._persist_directory = '/mnt/AI/data_base/vector_db/chroma_partition_summary_v1.8_only_two_file_by_title_qwen2_by_bgeM3'
        # self._doc_directory = '/mnt/AI/data_base/vector_db/chroma_doc_v1.8_only_two_file_by_title_by_qwen2_by_bgeM3' 
        self._persist_directory = '/mnt/AI/data_base/vector_db/chroma_partition_summary_v1.8_only_two_file_by_title_qwen2_by_bgeM3_max_chars'
        self._doc_directory = '/mnt/AI/data_base/vector_db/chroma_doc_v1.8_only_two_file_by_title_by_qwen2_by_bgeM3_max_chars' 
        self._rerank_model = '/mnt/AI/models/embedding_model/bge-reranker-base'
        self._embedding_model = '/mnt/AI/models/embedding_model/bge-m3/'
        self._model = ChatOpenAI(
            streaming=False,
            verbose=True,
            openai_api_key="none", # "EMPTY",
            openai_api_base="http://localhost:8000/v1",
            # model_name='/mnt/AI/models/internlm2_chat_7b/'
            model_name = 'Qwen2-7B-Instruct'
        )
        
    # define the simple fun to summary each page text
    def cutOff_text(self, text, max_length=1500):
            if len(text) > max_length:
                return text[:max_length] + "..."
            return text
        
    # limit the content size
    def limit_total_length(self, docs, max_total_length=6000):
        combined_content = ""
        current_length = 0
        limited_docs = []
 
        for doc in docs:
            doc_length = len(doc['page_content'])
            if current_length + doc_length > max_total_length:
                break
            combined_content += f"\nTitle: {doc['doc_title']}\nContent: {doc['page_content']}\n"
            current_length += doc_length
            limited_docs.append(doc)
 
        return combined_content, limited_docs
    
    # define the function to generate multi-queries
    def generate_multi_queries(self, retriever, original_query):
        
        # set the multi-query numbers
        multi_query_nums = 3
        
        # set the prompt for the multi-query generation
        multi_query_template = f"""You are an AI language model assistant. Your task is to generate {multi_query_nums} 
        different versions of the given user question. Provide these alternative questions separated by newlines.
        Original question: {{question}}"""
        
        prompt = ChatPromptTemplate.from_template(multi_query_template)
        
        generate_queries = (
            prompt
            | self._model
            | StrOutputParser()
            | (lambda x: x.split("\n"))
        )
        
        # start multi-query generation
        alternative_queries = generate_queries.invoke({"question": original_query})
        for i ,query in enumerate(alternative_queries,1):
            print(f"进行多路查询，改写问题 {i}:{query}")
            
        # retrieval_chain = generate_queries | retriever.map() | self.get_unique_union 
        # docs = retrieval_chain.invoke({"question": query}) 
        
        retrieval_chain_rag_fusion = generate_queries | retriever.map()
        docs = retrieval_chain_rag_fusion.invoke({"question": original_query})
        return docs
    
    def check_relevant_documents(self, query, limited_output):
        template = """请回复数字0或者1。
        判断知识库能不能回答以下的问题。
        如果能请返回1，否则请返回0。
        知识库：{context}
        问题: {question}
        注意：你的答案只能是0或者1，不要回复其他内容。"""
        prompt = PromptTemplate(input_variables=["context","question"],template=template)

        chain = (
            prompt
            | self._model
            | StrOutputParser()
        )
        response = chain.invoke({"context": limited_output, "question":query})
        print('===============relevant check result', response)
        if response == '0':
            return  [{'doc_title':'', 'page_content': ''}]
        
        return limited_output
  
        # query the doc title and content by retriever
    def query_doc_by_retriever(self, retriever, query):
        # relevant_docs = retriever.get_relevant_documents(query)

        # start to multi-query
        multi_query_relevant_docs = self.generate_multi_queries(retriever, query)
        
        # fused_scores = {}
        fused_scores = []
    
        # Setting use_fp16 to True speeds up computation with a slight performance degradation
        reranker = FlagReranker(model_name_or_path=self._rerank_model, use_fp16=True, device="cuda") 

        # Iterate through each list of ranked documents
        for docs in multi_query_relevant_docs:
            # Iterate through each document in the list, with its rank (position in the list)
            for i, doc in enumerate(docs):
                # print(f'\n rank: {i}, doc: {doc}\n') #rank: 0, doc: page_content='Compensation structure 薪酬结构' metadata={'doc_id': '10db3f95-0964-42d8-84cf-c23ef6b54a36', 'title': 'Tietoevry-China-Employee-Handbook.pdf'}
                doc_score = reranker.compute_score([query, doc.page_content])

                # If the document is not yet in the fused_scores dictionary, add it with an initial score of 0
                if not any(d['doc'].page_content.replace('\n', '').strip() == doc.page_content.replace('\n', '').strip() for d in fused_scores):
                    fused_scores.append({
                        "doc": doc,
                        "score": doc_score
                    })
     
        replace_pattern = [r'\n\n', r'Ws tietoeucy', r'© Tietoevry Tietoevry China Employee Handbook we tietoeucy', r'ANNEX a', r'We tietoeucy', r'\b[Pp]age \d+/\d+\b']
        # 将所有模式用 | 分隔，形成一个大的正则表达式
        replace_patterns = '|'.join(replace_pattern)
        
        # for item in fused_scores:
        #     print(f"\n type(item[doc]), item[doc]: {type(item['doc'])}, {item['doc']}， {item['score']}\n")
        # Sort the documents based on their fused scores in descending order to get the final reranked results
        reranked_results = [
            {
                    "doc_title":doc['doc'].metadata['title'],
                    "page_content":self.cutOff_text(re.sub(replace_patterns, ' ', doc['doc'].page_content))
            } for doc in sorted(fused_scores, key=lambda x: x['score'], reverse=True)[:4] 
        ]

        print(f'\n ==重排结果如下==: {reranked_results}\n')
        
         # 限制总上下文长度
        combined_content, limited_output = self.limit_total_length(reranked_results)
        print('>>before check limited_output:',limited_output)
        # relevant_documents
        limited_output = self.check_relevant_documents(query, limited_output)
        print('>>after check limited_output:',limited_output)
        return combined_content, limited_output
    
    def get_language(self, text):
        for char in text:
         if '\u4e00' <= char <= '\u9fff':
           return '中文'
         return '英文' 
        
    async def hr_bot(self, request):
        print("Received request data: ", request)
        query = request

        # Prompt template
        template = """你是一个HR知识库问答小助手。根据以下知识库做出判断，并给出你的答案。
         并且记录之前谈话的相关片段:{history},
        (You do not need to use these pieces of information if not relevant)
        如果你不知道答案，就说你不知道，不要试图编造答案。尽量使答案简明扼要。
        知识库：{context}
        问题: {question}
        注意：请使用{language}回答。"""
        
        language = self.get_language(query)
        prompt = PromptTemplate(input_variables=["history","context","question","language"],template=template)

        # 获取context的内容
        # for message in request.messages:
        #     context = message.content

        
        # RAG pipeline
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
            id_key=id_key
        )
        
        chain = (
            prompt
            | self._model
            | StrOutputParser()
        )

        db_loader = LoadBM25Retriever()
        bm25_retriever, relevant_documents = db_loader.fetch_relevant_documents(query)
    
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, retriever], weights=[0.5, 0.5], c=30
        )
        combined_content, limited_output = self.query_doc_by_retriever(ensemble_retriever, query)
        response =await chain.ainvoke({"context": combined_content, "question":query,"history": memory.load_memory_variables({}),"language":language})
        memory.save_context({"input": query},{"output":response } )
         
        print("历史记录：",memory.load_memory_variables({}))
        # feedback = request.get("feedback")
        # if feedback:
        #     fb = FeedBack()
        #     fb.write_feedback(feedback)

        print("========================================")
        print("User:", query)
        print("Chatbot:", response)
        print("Relevant documents:", limited_output)
        return response, limited_output

import jieba

class LoadBM25Retriever:
    def __init__(self):
        # self._doc_directory = '/mnt/AI/data_base/vector_db/chroma_doc_v1.8_only_two_file_by_title_internlm'
        # self._doc_directory = '/mnt/AI/data_base/vector_db/chroma_doc_v1.8_only_two_file_by_basic'
        # self._doc_directory = '/mnt/AI/data_base/vector_db/chroma_doc_v1.8_only_two_file_by_basic_by_qwen2'
        # self._doc_directory = '/mnt/AI/data_base/vector_db/chroma_doc_v1.8_only_two_file_by_basic_by_qwen2_by_bgeM3'
        # self._doc_directory = '/mnt/AI/data_base/vector_db/chroma_doc_v1.8_only_two_file_by_title_by_qwen2_by_bgeM3'
        self._doc_directory = '/mnt/AI/data_base/vector_db/chroma_doc_v1.8_only_two_file_by_title_by_qwen2_by_bgeM3_max_chars' 
    
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
        return retriever, results
    
    def preprocessing_func(self, text:str):
        return list(jieba.lcut(text.strip()))


# Run the chatbot
if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    hr = HR_BOT()

    while True:
        msg = input("Input your message (type 'exit' to quit): ")
        if msg == 'exit' or msg == 'quit':
            break
        # 运行异步主函数
        feedback = 'The answer is very helpful,Not factually correct,Like the style'       
        # user_query = {"content": msg, "feedback":feedback}
        user_query = msg  
        loop.run_until_complete(hr.hr_bot(user_query))

    loop.close()


