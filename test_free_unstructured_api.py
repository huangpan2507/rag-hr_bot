import json, os

import unstructured_client
from unstructured_client import UnstructuredClient
from unstructured_client.models import operations, shared

os.environ['UNSTRUCTURED_API_KEY'] = 'pqROWQOdunRwFCQXn9cHG4u2zleSmC'
os.environ['UNSTRUCTURED_API_URL'] = 'https://api.unstructured.io/general/v0/general'

client = unstructured_client.UnstructuredClient(
    api_key_auth=os.getenv("UNSTRUCTURED_API_KEY"),
    server_url=os.getenv("UNSTRUCTURED_API_URL")
)

input_filepath = "/mnt/AI/hr_material_test_huangpan/test.pdf"
output_filepath = "/mnt/AI/hr_material_test_huangpan/test.txt"

with open(input_filepath, "rb") as f:
    files = shared.Files(
        content=f.read(),
        file_name=input_filepath
    )

req = operations.PartitionRequest(
    shared.PartitionParameters(
        files=files,
        strategy=shared.Strategy.HI_RES,
        chunking_strategy=shared.ChunkingStrategy.BY_TITLE,
        languages=['chi_sim', 'eng'],
        # extract_images_in_pdf=True,
        # infer_table_structure=True,
        # form_extraction_skip_tables = False,
    )
)

try:
    res = client.general.partition(request=req)
    element_dicts = [element for element in res.elements]
    # element_text = [element.to_dict() for element in res.elements]
    element_text = [element['text'].replace('\n\n', ' ') for element in res.elements]
    json_elements = json.dumps(element_dicts, indent=2)
    
    # Print the processed data.
    print(f'\n json_elements: {json_elements} \n')
    print(f'\n element_text: {element_text} \n')

    # Write the processed data to a local file.
    with open(output_filepath, "w") as file:
      file.write(json_elements)
except Exception as e:
    print(e)