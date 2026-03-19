
# 首先导入所需第三方库
#from langchain.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import UnstructuredFileLoader
from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma
#from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from tqdm import tqdm
import os
#from sentence_transformers import SentenceTransformer

#OPENAI_API_KEY = 'sk-7VrEAM9ieJh6bzDIBbF1018a7f844d25B534CdF9E569B5A8'
# 获取文件路径函数
def get_files(dir_path):
    file_list = []
    extensions = [".pdf", ".md", ".txt", ".py"]

    for dirpath, _, filenames in os.walk(dir_path):
        for filename in filenames:
            if any(filename.endswith(ext) for ext in extensions):
                file_list.append(os.path.join(dirpath, filename))

    return file_list


# 加载文件函数
def get_text(dir_path):
    file_lst = get_files(dir_path)
    docs = []

    loaders = {
        'pdf': PyPDFLoader,
        'md': UnstructuredMarkdownLoader,
        'txt': UnstructuredFileLoader,
        'py': UnstructuredFileLoader
    }

    for one_file in tqdm(file_lst):
        file_type = one_file.split('.')[-1]
        if file_type in loaders:
            loader = loaders[file_type](one_file)
            docs.extend(loader.load())

    return docs





# 目标文件夹
tar_dir = [
    "../hr_metrial/",
]

# 加载目标文件
docs = []
for dir_path in tar_dir:
    docs.extend(get_text(dir_path))

# 对文本进行分块
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=150)
split_docs = text_splitter.split_documents(docs)

# 加载开源词向量模型
#embeddings = HuggingFaceEmbeddings(model_name="/mnt/AI/models/embedding_model")
embeddings = OpenAIEmbeddings()



# 构建向量数据库
# 定义持久化路径
persist_directory = '../data_base/vector_db/chroma'
# 加载数据库
vectordb = Chroma.from_documents(
    documents=split_docs,
    embedding=embeddings,
    persist_directory=persist_directory  # 允许我们将persist_directory目录保存到磁盘上
)
# 将加载的向量数据库持久化到磁盘上
vectordb.persist()

