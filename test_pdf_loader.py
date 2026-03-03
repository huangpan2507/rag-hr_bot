# # 方式一：
# from langchain.document_loaders import UnstructuredPDFLoader
# #from langchain_community.document_loaders import UnstructuredPDFLoader
# loader = UnstructuredPDFLoader(file_path="/mnt/AI/hr_material_test_huangpan/Tietoevry-China-Training-Regulation---EN-CN.pdf", 
#                                mode="elements",
#                                include_page_break=False,
#                                infer_table_structure=False,
#                                languages=["Eng","chi_sim"],
#                                strategy="hi_res",
#                                include_metadata=True)
                           

# documents = loader.load()

# #print(type(documents))   <class 'list'>
# print(f"\n documents: {documents}\n")


#方式二：
from unstructured.partition.pdf import partition_pdf
from pydantic import BaseModel
from typing import Any
import os
import subprocess
import base64
import glob
import uuid

pdf_path = "/mnt/AI/hr_material_test_huangpan/"
img_path = "/mnt/AI/Papers/figure_test_huangpan/" 

pdf_paths = [os.path.join(pdf_path, filename) for filename in os.listdir(pdf_path)
                    if os.path.isfile(os.path.join(pdf_path, filename)) and filename.lower().endswith(".pdf")]

raw_pdf_elements = []
for file_name in pdf_paths:
    print(f"Processing PDF: {file_name}")
    # Get elements
    one_raw_pdf_elements = partition_pdf(
        filename=file_name,
        languages=["chi_sim", "eng"],
        strategy='hi_res',
        # Using pdf format to find embedded image blocks
        extract_images_in_pdf=True,
        # Use layout model (YOLOX) to get bounding boxes (for tables) and find titles
        # Titles are any sub-section of the document
        infer_table_structure=True,
        # Post processing to aggregate text once we have the title
        chunking_strategy="by_title",
        include_metadata=True,
        extract_image_block_output_dir=img_path + os.path.basename(file_name),
        form_extraction_skip_tables = False,
        # max_characters=20,
        # new_after_n_chars=10,
        # overlap = 3,
        # combine_text_under_n_chars=8
    )
    raw_pdf_elements.extend(one_raw_pdf_elements)

print(f'\n one_raw_pdf_elements: {one_raw_pdf_elements}\n')
print(f'\n raw_pdf_elements: {raw_pdf_elements}\n')

# # # # 如果有坐标属性则打印每个元素的坐标
# # # for i, txe in enumerate(one_raw_pdf_elements):
# # #     if hasattr(txe, 'metadata') and hasattr(txe.metadata, 'coordinates'):
# # #         coordinates = txe.metadata.coordinates
# # #         if hasattr(coordinates, 'points'):
# # #             print(f"\nElement {i} coordinates points: {coordinates.points}\n")

class Element(BaseModel):
            type: str
            text: Any

# Categorize by type
categorized_elements = []
for element in raw_pdf_elements:
    print(f'\n element in raw_pdf_elements: {element} \n')
    if "unstructured.documents.elements.Table" in str(type(element)):
        categorized_elements.append(Element(type="table", text=str(element)))
    elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
        categorized_elements.append(Element(type="text", text=str(element)))

print(f'\n categorized_elements: {categorized_elements} \n')


# # 遍历根目录下的所有子目录，该子目录名字：包括文件名后缀的全名而命令的目录
# for subdir in os.listdir(img_path):
#     subdir_path = os.path.join(img_path, subdir)
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

# print(f'\n img_base64_list: {img_base64_list}\n')
# print(f'\n img_summaries: {img_summaries}\n')
# print("All text files have been created.")






# # 方式三
# #from langchain.document_loaders import UnstructuredPDFLoader
# from langchain_community.document_loaders import UnstructuredPDFLoader

# img_path = "/mnt/AI/Papers/figure_test_huangpan/"
# loader = UnstructuredPDFLoader(file_path="/mnt/AI/hr_material_test_huangpan/Finance-Guidance-China.pdf", 
#                                mode="elements",
#                                include_page_break=False,
#                                languages=["Eng","chi_sim"],
#                                strategy="hi_res",
#                                # Using pdf format to find embedded image blocks
#                                extract_images_in_pdf=True,
#                                # Use layout model (YOLOX) to get bounding boxes (for tables) and find titles
#                                # Titles are any sub-section of the document
#                                infer_table_structure=True,
#                                # Post processing to aggregate text once we have the title
#                                chunking_strategy="by_title",
#                                extract_image_block_output_dir=img_path,
#                                form_extraction_skip_tables = False,
#                                include_metadata=True,
#                                max_characters=3500,
#                                new_after_n_chars=1500,
#                                combine_text_under_n_chars=250
#                                )
                           

# documents = loader.load()

# #new_documents = [e.page_content for e in documents if e.page_content != ""]
# new_documents = [e.page_content.replace('\n\n', ' ') for e in documents if e.page_content != ""]

# #print(type(documents))   <class 'list'>
# print(f"\n new_documents \n : {new_documents}\n")



# # 方式一改：
# from langchain.document_loaders import UnstructuredPDFLoader
# #from langchain_community.document_loaders import UnstructuredPDFLoader
# loader = UnstructuredPDFLoader(file_path="/mnt/AI/hr_material_test_huangpan/Finance-Guidance-China.pdf", 
#                                mode="elements",
#                                include_page_break=False,
#                                infer_table_structure=True,
#                                languages=["Eng","chi_sim"],
#                                strategy="hi_res",
#                                include_metadata=True,
#                                max_characters=3500,
#                                new_after_n_chars=1500,
#                                combine_text_under_n_chars=250)
                           

# documents = loader.load()
# print(f"\n documents: {documents}\n")
# new_documents = [e.page_content.replace('\n\n', ' ') for e in documents if e.page_content != ""]
# #print(type(documents))   <class 'list'>
# print(f"\n new_documents \n : {new_documents}\n")


# # 方式一改改：
# from langchain.document_loaders import UnstructuredPDFLoader
# #from langchain_community.document_loaders import UnstructuredPDFLoader
# loader = UnstructuredPDFLoader(file_path="/mnt/AI/hr_material_test_huangpan/Finance-Guidance-China.pdf", 
#                                mode="elements",
#                                include_page_break=False,
#                                infer_table_structure=False,
#                                languages=["Eng","chi_sim"],
#                                strategy="hi_res",
#                                include_metadata=True,
#                                max_characters=3500,
#                                new_after_n_chars=1500,
#                                combine_text_under_n_chars=250)
                           

# documents = loader.load()
# start_page_number = 1
# stop_page_number = 3


# # cleaned_elements = [
# #     element
# #     for element in documents
# #     # Nuke the element even if 1 point is outside the cutcoord.
# #     if all(
# #         143.61111111111123 < coord[1] < 2070.266666666667
# #         for coord in element.metadata['coordinates']['points']
# #     ) and (start_page_number < element.metadata['page_number'] < stop_page_number)
# # ]

# def find_second_min_max(values):
#     if len(values) < 2:
#         raise ValueError("List must contain at least two elements.")
    
#     unique_values = list(set(values))  # 去重
#     unique_values.sort()
    
#     if len(unique_values) < 2:
#         raise ValueError("Not enough unique elements to determine second min and max.")
    
#     second_min = unique_values[1]
#     second_max = unique_values[-2]
    
#     return second_min, second_max

# # def find_smallest_largest(values):
# #     if len(values) < 2:
# #         raise ValueError("List must contain at least two elements.")
    
# #     unique_values = list(set(values))  # 去重
# #     unique_values.sort()
    
# #     if len(unique_values) < 2:
# #         raise ValueError("Not enough unique elements to determine second min and max.")
    
# #     smallest = unique_values[0]
# #     largest = unique_values[-1]
    
# #     return smallest, largest

# # 提取所有的 y 坐标值
# y_coords = [
#     coord[1]
#     for element in documents 
#         for coord in element.metadata['coordinates']['points']
# ]

# # # 提取所有页面值
# # pages_nums = [
# #     page_number
# #     for element in documents 
# #         for page_number in element.metadata['page_number']
# # ]

# # 计算y轴坐标的次最小值和次最大值
# second_min, second_max = find_second_min_max(y_coords)

# # # 获取起始页码和最终页码
# # start_page_number, stop_page_number = find_smallest_largest(pages_nums)

# cleaned_elements = [
#     element
#     for element in documents
#     # Nuke the element even if 1 point is outside the cutcoord.
#     if all(
#         second_min < coord[1] < second_max
#         for coord in element.metadata['coordinates']['points']
#     )
# ]

# # coord_elements = [
# #     coord[1]
# #     for element in documents 
# #         for coord in element.metadata['coordinates']['points']
# # ]


# # coord_elements = [
# #     coord
# #     for element in documents 
# #         for point in element.metadata['coordinates']['points']
# #             for coord in point

# # ]

# #print(f"\n element:  {[e for e in documents]} \n")
# #print(f"\n points: {[e.metadata['coordinates']['points']  for e in documents]}\n")
# # print(f"\n points[1]: {[e.metadata['coordinates']['points'][1]  for e in documents]}\n")
# # print(f"\n points[1][1]: {[e.metadata['coordinates']['points'][1][1]  for e in documents]}\n")
# # print(f"\n coord_elements1: {coord_elements}\n")

# print(f"\n cleaned_elements: {cleaned_elements}\n")
# # new_documents = [e.page_content.replace('\n\n', ' ') for e in cleaned_elements if e.page_content != ""]

# # # #print(type(documents))   <class 'list'>
# # print(f"\n new_documents \n : {new_documents}\n")


# # 方式一改改改：
# from langchain.document_loaders import UnstructuredPDFLoader
# #from langchain_community.document_loaders import UnstructuredPDFLoader

# img_path = "/mnt/AI/Papers/figure_test_huangpan/"  
# loader = UnstructuredPDFLoader(file_path="/mnt/AI/hr_material_test_huangpan/Finance-Guidance-China.pdf", 
#                                mode="elements",
#                                include_page_break=False,
#                                infer_table_structure=True,
#                                languages=["Eng","chi_sim"],
#                                strategy="hi_res",
#                                extract_images_in_pdf=True,
#                                # Post processing to aggregate text once we have the title
#                                #chunking_strategy="by_title",
#                                extract_image_block_output_dir=img_path,
#                                form_extraction_skip_tables = False,
#                                include_metadata=True,
#                                max_characters=3500,
#                                new_after_n_chars=1500,
#                                combine_text_under_n_chars=250)
                           

# documents = loader.load()

# def find_second_min_max(values):
#     if len(values) < 2:
#         raise ValueError("List must contain at least two elements.")
    
#     unique_values = list(set(values))  # 去重
#     unique_values.sort()
    
#     if len(unique_values) < 2:
#         raise ValueError("Not enough unique elements to determine second min and max.")
    
#     second_min = unique_values[1]
#     second_max = unique_values[-2]
    
#     return second_min, second_max


# # 提取所有的 y 坐标值
# y_coords = [
#     coord[1]
#     for element in documents 
#         for coord in element.metadata['coordinates']['points']
# ]

# # 计算y轴坐标的次最小值和次最大值
# second_min, second_max = find_second_min_max(y_coords)

# cleaned_elements = [
#     element
#     for element in documents
#     # Nuke the element even if 1 point is outside the cutcoord.
#     if all(
#         second_min < coord[1] < second_max
#         for coord in element.metadata['coordinates']['points']
#     )
# ]


# print(f"\n cleaned_elements: {cleaned_elements}\n")
# new_documents = [e.page_content.replace('\n\n', ' ') for e in cleaned_elements if e.page_content != ""]

# print(f"\n new_documents \n : {new_documents}\n")



# # 方式三改
# #from langchain.document_loaders import UnstructuredPDFLoader
# from langchain_community.document_loaders import UnstructuredPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter

# img_path = "/mnt/AI/Papers/figure_test_huangpan/"
# loader = UnstructuredPDFLoader(file_path="/mnt/AI/hr_material_test_huangpan/Finance-Guidance-China.pdf", 
#                                mode="elements",
#                                include_page_break=False,
#                                languages=["Eng","chi_sim"],
#                                strategy="hi_res",
#                                # Using pdf format to find embedded image blocks
#                                extract_images_in_pdf=True,
#                                # Use layout model (YOLOX) to get bounding boxes (for tables) and find titles
#                                # Titles are any sub-section of the document
#                                infer_table_structure=True,
#                                # Post processing to aggregate text once we have the title
#                                #chunking_strategy="by_title",
#                                extract_image_block_output_dir=img_path,
#                                form_extraction_skip_tables = False,
#                                include_metadata=True
#                                )
                           

# documents = loader.load()
# print(f"\n documents \n : {documents}\n")

# #new_documents = [e.page_content for e in documents if e.page_content != ""]
# new_documents = [e.page_content.replace('\n\n', ' ') for e in documents if e.page_content != ""]

# #print(type(documents))   <class 'list'>
# print(f"\n new_documents \n : {new_documents}\n")


# # 方式一改改：
# from langchain.document_loaders import UnstructuredPDFLoader
# #from langchain_community.document_loaders import UnstructuredPDFLoader
# loader = UnstructuredPDFLoader(file_path="/mnt/AI/hr_material_test_huangpan/Finance-Guidance-China.pdf", 
#                                mode="elements",
#                                include_page_break=False,
#                                infer_table_structure=False,
#                                languages=["Eng","chi_sim"],
#                                strategy="hi_res",
#                                include_metadata=True,
#                                max_characters=3500,
#                                new_after_n_chars=1500,
#                                combine_text_under_n_chars=250)
                           
# documents = loader.load()

# def find_second_min_max(values):
#     if len(values) < 2:
#         raise ValueError("List must contain at least two elements.")
    
#     unique_values = list(set(values))  # 去重
#     unique_values.sort()
    
#     if len(unique_values) < 2:
#         raise ValueError("Not enough unique elements to determine second min and max.")
    
#     second_min = unique_values[1]
#     second_max = unique_values[-2]
    
#     return second_min, second_max

# # 提取所有的 y 坐标值
# y_coords = [
#     coord[1]
#     for element in documents 
#         for coord in element.metadata['coordinates']['points']
# ]

# second_min, second_max = find_second_min_max(y_coords)

# cleaned_elements = [
#     element
#     for element in documents
#     # Nuke the element even if 1 point is outside the cutcoord.
#     if all(
#         second_min < coord[1] < second_max
#         for coord in element.metadata['coordinates']['points']
#     )
# ]
# # coord_elements = [
# print(f"\n cleaned_elements: {cleaned_elements}\n")
