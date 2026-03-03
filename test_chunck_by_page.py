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
import base64
import os
import subprocess
import glob
import uuid
from langchain_core.documents import Document
from langchain.chat_models import ChatOpenAI
import re

# os.environ["TABLE_IMAGE_CROP_PAD"] = "2"

class make_db:
    def __init__(self) -> None:
        self._pdf_path = "/mnt/AI/hr_material_test_huangpan/"
        self._img_path = "/mnt/AI/Papers/figure_test_huangpan/"  
        self._persist_directory = '/mnt/AI/data_base/vector_db/chroma_partition_summary_v1.8_only_two_file_by_basic'
        self._doc_directory = '/mnt/AI/data_base/vector_db/chroma_doc_v1.8_only_two_file_by_basic'
        self._embedding_model = '/mnt/AI/models/embedding_model'
        # self._llm_model = '/mnt/AI/models/MiniCPM-Llama3-V-2_5/'
        self._llm_model = '/mnt/AI/models/internlm2_chat_7b/'

    def make_vector_db(self, pdf_paths):
        documents = []
        for file_name in pdf_paths:
            print(f"\n file_name: {file_name} \n ")
            loader = UnstructuredPDFLoader(file_path=file_name, 
                                            include_page_break=False,
                                            infer_table_structure=True,
                                            languages=["Eng","chi_sim"],
                                            strategy="hi_res",
                                            extract_images_in_pdf=True,
                                            # Post processing to aggregate text once we have the title
                                            extract_image_block_output_dir=self._img_path + os.path.basename(file_name),
                                            form_extraction_skip_tables = False,
                                            # chunking_strategy="by_title",
                                            # max_characters=3500,                      # 4000
                                            # new_after_n_chars=1500,                   # 3800
                                            # combine_text_under_n_chars=250,          # 2000
                                            # overlap = 70,
                                            include_metadata=True)
            document = loader.load()
            documents.extend(document)

        print(f"\n documents: {documents}\n")
        replace_pattern = [r'\n\n', r'Ws tietoeucy', r'© Tietoevry Tietoevry China Employee Handbook we tietoeucy', r'ANNEX a']
        # 将所有模式用 | 分隔，形成一个大的正则表达式
        replace_patterns = '|'.join(replace_pattern)

        # 使用 re.sub() 函数替换所有以Page x/x 或者 page x/x的字符串
        split_pattern = r'\b[Pp]age \d+/\d+\b'

        new_documents_by_page = [
        Document(page_content=page_content.strip(), metadata={"title": os.path.basename(file_name)})
        for e in documents if e.page_content != ""
        for page_content in re.split(split_pattern, re.sub(replace_patterns, ' ', e.page_content))
        if page_content.strip() != ""
        ]

        print(f"\n new_documents: {new_documents_by_page}\n")
        print(f"\n type new_documents: {type(new_documents_by_page)}\n")
        
        #----------------------------------------------use text begin-----------------------------------------------
        # text = '''page_content='© Tietoevry Tietoevry China Employee Handbook we tietoeucy供应商请款模板17\n\npage 1/17\n\nWs tietoeucy\n\nOverview\n\n1 费用分类 Payment Categories\n\n公司的请款共分为三类，分别是员工费用报销、预支款、对供应商付款。 There are three categories of payments, which are staff reimbursement, advances and\n\npayment to suppliers.\n\n• 员工费用报销：为工作开展发生的差旅费、交通费、培训费、招待费、部门活动 费、零星采购(指非常规采购，且在同一供应商单次采购总额低于 3000 元)等Ws tietoeucy。 Staff reimbursement: Work related travel expenses, transportation expenses, training expenses, hospitality expenses, team activity expenses, retail purchases (refers to irregular purchase, and the total amount of a single purchase from the same vendor is less than 3000CNY), etc.\n\n• 员工预支款：员工预支款仅限于员工的国际出差以及超过 6 个月的国内出差，预 支款项以每人总额不超过人民币 20,000 元为限，且需附预算清单。 Staff Advances: Only applicable for international business trips and over 6 months domestic business trips. 20,000RMB is the limit for each employee and budget list must be accompanied by.\n\n• 供应商预支款: 仅限于供应商无规律性，零星发生的费用。金额超过人民币 3,000RMB 元以上的支出必须与供应商签订合同，并通过转账支付。 Supplier Prepayment: Only irregular costs or any costs over 3,000RMB must have contracts with vendors. Payments will be done only by the bank transferring, not cash.\n\n• 对供应商付款：公司采购及各项费用，包括清洁费、物管费、房屋租金、猎头 费、装修费、大宗采购等。有规律性发生的费用（如复印机租赁、清洁费等）， 以及零星发生但金额超过人民币 3,000RMB 元以上的支出必须与供应商签订合 同，并通过转账支付。\n\nPayment to vendors: Company procurement and various expenses, including cleaning fees, property management fees, rents, headhunting fees, decoration fees, bulk purchasing, etc. Regular costs (Such as copier rental, cleaning etc.) or any costs over 3,000RMB must have contracts with vendors. Payments will be done only by the bank transferring, not cash.\n\npage 2/17\n\nWs tietoeucy\n\n2 请款对象 Persons that Request the Payments\n\n本流程所有请款对象，均应为与公司签约的正式员工。\n\nPersons that request the payments should be company’s FTE.\n\n3 付款对象信息维护 Recipient information maintenance\n\n• 新进员工/员工账号发生变化：需要在 My Accounting Support 中提出更改申请，\n\n需列明开户行名称、银行账号、联行号等信息； For new employees and employees’ account changed: It is required to raise a ticket in My Accounting Support, and list the information of bank name, account number and CNAPS number.\n\n新供应商：费用申请人以邮件形式发送给财务部同事供应商的账户信息，包含供 应商公司名称、开户行名称、开户行账号。同时需提供该供应商的营业执照复印 件或扫描件。 For new suppliers, the applicant needs send the supplier\'s account information to the relevant colleagues in the finance department by email. Account information must include supplier company name, bank name and account number, and also needs to provide a copy or scanned copy of the supplier\'s business license.\n\n4 所需文件 Required Documents\n\n员工费用报销：费用报销申请单、发票原件、购货清单、其他相关资料(如合同、换汇 水单)；\n\nStaff reimbursement: Expense reimbursement application form, original invoice (FaPiao), purchase list, exchange receipt and other relevant documents (e.g. contracts, exchange bills)\n\n预支款: • 员工部分：预支申请单、审批邮件、预算明细表及其他相关支持文件；\n\nStaff advances: Advance application form, approval mail, budget list and other relevant documents.\n\n• 供应商部分：预支申请单、审批邮件、合同或订单及其他相关支持文件。 Vendor Advances: Advance application form, approval mail, contract or purchase order and other relevant documents.\n\n供应商付款：费用报销申请单、发票原件、合同、送货单或验收单。 Payments to vendor: Expense claim form, original invoice (FaPiao), contract, goods delivery\n\nnote or goods receipt note.\n\npage 3/17\n\nWs tietoeucy网站零星购买以及日用品报销 Web Retail Purchases and Commodity Reimbursement 网站零星购买需附发票及购物清单、送货单。\n\nWeb retail purchases and commodity reimbursement must be accompanied by the invoice, shopping list and delivery list.\n\npage 12/17\n\n © Tietoevry Tietoevry China Employee Handbook we tietoeucy'''

        # replace_pattern = [r'\n\n', r'Ws tietoeucy', r'© Tietoevry Tietoevry China Employee Handbook we tietoeucy', r'ANNEX a']
        # # 将所有模式用 | 分隔，形成一个大的正则表达式
        # replace_patterns = '|'.join(replace_pattern)

        # # 使用 re.sub() 函数替换所有以Page x/x 或者 page x/x的字符串
        # split_pattern = r'\b[Pp]age \d+/\d+\b'

        # # 去除文本中的换行符和 Ws tietoeucy字符       
        # text = re.sub(replace_patterns, ' ', text)

        # # 分割文本
        # pages = re.split(split_pattern, text)
        # print(f'pages:{pages}\n')
        # print(f'type pages:{type(pages)}\n')

        # # 去除分割后的空字符串（如果有）
        # pages = [page.strip() for page in pages if page.strip()]

        # for i, page in enumerate(pages):
        #     print(f"Page {i+1}:")
        #     print(page)
        #     print("--新的---")
        #-------------------------------------------------use text end--------------------------------------------------
              
    
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
