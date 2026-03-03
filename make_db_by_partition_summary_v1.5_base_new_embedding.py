from langchain.document_loaders import UnstructuredPDFLoader
#from langchain_community.document_loaders import UnstructuredPDFLoader
from typing import Any
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.storage import LocalFileStore
from langchain.storage._lc_store import create_kv_docstore
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.retrievers import MultiVectorRetriever
from langchain.embeddings import HuggingFaceBgeEmbeddings
from transformers import AutoTokenizer, AutoModel
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
        self._persist_directory = '/mnt/AI/data_base/vector_db/chroma_partition_summary_v1.8_only_two_file_by_title'
        self._doc_directory = '/mnt/AI/data_base/vector_db/chroma_doc_v1.8_only_two_file_by_title'
        # self._persist_directory = '/mnt/AI/data_base/vector_db/chroma_partition_summary_v1.8_only_two_file_by_title_internlm'
        # self._doc_directory = '/mnt/AI/data_base/vector_db/chroma_doc_v1.8_only_two_file_by_title_internlm'
        # self._pdf_path = "/mnt/AI/hr_material_v1.5/" 
        # self._img_path = "/mnt/AI/Papers/figure_v1.5/"
        # self._persist_directory = '/mnt/AI/data_base/vector_db/chroma_partition_summary_v1.5'
        # self._doc_directory = '/mnt/AI/data_base/vector_db/chroma_doc_v1.5'
        self._embedding_model = '/mnt/AI/models/embedding_model/bge-m3/'
        # self._llm_model = '/mnt/AI/models/MiniCPM-Llama3-V-2_5/'
        self._llm_model = '/mnt/AI/models/internlm2_chat_7b/'

    def make_vector_db(self, pdf_paths):
        documents = []
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
                                            # max_characters=3500,                      # 4000
                                            # new_after_n_chars=1500,                   # 3800
                                            # combine_text_under_n_chars=250,          # 2000
                                            # overlap = 70,
                                            include_metadata=True)
            document = loader.load()
            documents.extend(document)

        print(f"\n documents: {documents}\n")

        new_documents = [
                Document(page_content=e.page_content.replace('\n\n', ' '), metadata={"title": os.path.basename(file_name)})
                for e in documents if e.page_content != ""
        ]
        print(f"\n new_documents: {new_documents} \n ")
        

        # 定义本地embedding模型
        # model_name = "bge-m3"  
        # save_directory = "/root/autodl-tmp/bge-m3"
        # tokenizer = AutoTokenizer.from_pretrained(model_name)
        # model = AutoModel.from_pretrained(model_name)
        # embedding = BGEM3Embeddings(model, tokenizer)


        # 定义huggingface的embedding模型,使用bge-m3
        model_kwargs = {'device': 'cuda', 'trust_remote_code':True}
        encode_kwargs = {'normalize_embeddings': False}
        embeddings = HuggingFaceBgeEmbeddings(
            model_name=self._embedding_model,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
            show_progress=True
        )

        print(f"\n embeddings: {embeddings.dict()} \n")

        query_result = embeddings.embed_query("发票不慎遗失怎么办？")
        print(f"\n query_result: {query_result} \n :")
        
        # embeddings = HuggingFaceEmbeddings(model_name=self._embedding_model)
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

        # # Prompt
        # prompt_text = """You are an assistant tasked with summarizing tables and text. \
        # Give a concise summary of the table or text. Table or text chunk: {element} """
        # prompt = ChatPromptTemplate.from_template(prompt_text)

        # model = ChatOpenAI(
        #     streaming=False,
        #     verbose=True,
        #     openai_api_key="none",
        #     openai_api_base="http://localhost:8000/v1",
        #     model_name=self._llm_model
        # )
        # summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

        # Initialize empty summaries
        text_summaries = []

        # Add texts
        # text_summaries = summarize_chain.batch(new_documents, {"max_concurrency": 5})
        doc_ids_text = [str(uuid.uuid4()) for _ in new_documents]

        # summary_texts = [
        #         Document(page_content=s, metadata={id_key: doc_ids_text[i], "title": new_documents[i].metadata["title"]})
        #         for i, s in enumerate(text_summaries)
        #     ]
        # print(f"\n summary_texts: {summary_texts} \n :")

        origin_texts = [
                Document(page_content=doc.page_content, metadata={id_key: doc_ids_text[i], "title": doc.metadata["title"]})
                for i, doc in enumerate(new_documents)
            ]
        
        print(f"\n origin_texts: {origin_texts} \n ")
        # retriever.vectorstore.add_documents(summary_texts)
        retriever.docstore.mset(list(zip(doc_ids_text, origin_texts)))

        # 如下是在生成db阶段，加入了混合检索，可以正常使用
        # ------------------混合检索---------------------
        # bm25_retriever = BM25Retriever.from_documents(
        #     origin_texts
        # )
        # bm25_retriever.k = 3

        # ensemble_retriever = EnsembleRetriever(
        #     retrievers=[bm25_retriever, retriever], weights=[0.5, 0.5]
        # )
        # result = ensemble_retriever.get_relevant_documents("员工个人原因给公司造成8000元损失，公司会怎么处理?")
        # # result = ensemble_retriever.invoke("未使用的年假能带到下一年吗？")
        # print(f'\n result: {result}\n')
        # ------------------混合检索---------------------


        # # call script to handle images. all the images' summaries files are stored in img_path.
        # # 遍历根目录下的所有子目录
        # for subdir in os.listdir(self._img_path):
        #     subdir_path = os.path.join(self._img_path, subdir)
        #     if os.path.isdir(subdir_path):
        #         print(f'\nProcessing subdir_path: {subdir_path}\n')
                
        #         # 运行shell脚本处理图像并生成txt文件
        #         subprocess.run(["bash", "/mnt/AI/gen_db/imagesummary.sh", subdir_path])

        #         img_base64_list = []
        #         # 使用 glob 模块查找目录中的所有以jpg结尾的图片
        #         image_files = glob.glob(os.path.join(subdir_path, "*.jpg"))
        #         if image_files:
        #             for img_file in image_files:
        #                 img_paths = os.path.join(subdir_path, img_file)
        #                 with open(img_paths, "rb") as image_file:
        #                         base64_image =  base64.b64encode(image_file.read()).decode("utf-8")
        #                 img_base64_list.append(base64_image)

        #         figure_file_paths = glob.glob(os.path.expanduser(os.path.join(subdir_path, "*.txt")))
        #         # Read each file and store its content in a list
        #         img_summaries = []
        #         for file_path in figure_file_paths:
        #             with open(file_path, "r") as file:
        #                 img_summaries.append(file.read())
        #                 print("img_summaries : {}\n".format(img_summaries))

        #         # Add image
        #         img_ids = [str(uuid.uuid4()) for _ in img_base64_list]
        #         summary_img = [
        #             Document(page_content=s, metadata={id_key: img_ids[i], "title": os.path.basename(subdir_path)})
        #             for i, s in enumerate(img_summaries)
        #         ]
        #         image_base64 = [
        #             Document(page_content=s, metadata={id_key: img_ids[i], "title": os.path.basename(subdir_path)})
        #             for i, s in enumerate(img_base64_list)
        #         ]
        #         print(f'\n summary_img: {summary_img}\n')
        #         print(f'\n image_base64: {image_base64}\n')
        #         retriever.vectorstore.add_documents(summary_img)
        #         retriever.docstore.mset(list(zip(img_ids, image_base64)))
        #         print("__________________________________________________________________")
        #         print("img_ids:{} \n summary_img:{}".format(img_ids, summary_img))
        #         print("__________________________________________________________________")
        #         print("img_ids:{} \n summary_img:{}".format(img_ids, image_base64))


    
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
