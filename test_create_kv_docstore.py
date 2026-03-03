from langchain.storage import LocalFileStore
from langchain.storage import LocalFileStore
from langchain.storage._lc_store import create_kv_docstore
from langchain_core.documents import Document

fs = LocalFileStore("/mnt/AI/data_base/vector_db/chroma_new_huangpan")
store = create_kv_docstore(fs)

# 向store写入包含中英文的文档
document = Document(page_content="这是一个测试文档。This is a test document. 你吃饭了吗，今天天气不错，weather is nice！所有外部培训供应商必须与公司签订商业合同后，才可开始提供服务。本合同应包括常规的合同信息，包括但不限于服务范围、交付时间和付款条款等。")

# 会存储中文的unicode码，而不是中文
store.mset([("doc3", document)])

# 读取文档
retrieved_docs = store.mget(["doc3"])
for doc in retrieved_docs:
    if doc is not None:
        print(f"doc.page_content:, {doc.page_content}")

# print result: doc.page_content:, 这是一个测试文档。This is a test document. 你吃饭了吗，今天天气不错，weather is nice！所有外部培训供应商必须与公司签订商业合同后，才可开始提供服务。本合同应包括常规的合同信息，包括但不限于服务范围、交付时间和付款条款等。