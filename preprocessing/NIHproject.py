import requests
import json
import re
import time
from tqdm import tqdm


def formatFloat(num):
    return '{:.2f}'.format(num)

files_endpt = "https://api.gdc.cancer.gov/files"

filters = {
    "op": "and",
    "content":[
        {"op": "in",
        "content":{
            "field": "cases.project.project_id",
            "value": ["TCGA-LUSC"]
            }
        }, 
        {"op": "in",
        "content":{
            "field": "cases.submitter_id",
            "value": ["TCGA-22-0940"]
            }
        }, 
        {"op":"=",
          "content":{
              "field":"files.experimental_strategy",
              "value":["Diagnostic Slide"]
              }
          },
        {"op": "in",
        "content":{
            "field": "files.data_format",
            "value": ["svs"]
            }
        }]
    }

# Here a GET is used, so the filter parameters should be passed as a JSON string.

params = {
    "filters": json.dumps(filters),
    "fields": "file_id",
    "format": "JSON",
    "size": "200"
    }

response = requests.get(files_endpt, params = params)

file_uuid_list = []
'''
# This step populates the download list with the file_ids from the previous query
for file_entry in json.loads(response.content.decode("utf-8"))["data"]["hits"]:
    file_uuid_list.append(file_entry["file_id"])
data_endpt = "https://api.gdc.cancer.gov/data"

params = {"ids": file_uuid_list}

response = requests.post(data_endpt, stream=True, data = json.dumps(params), headers = {"Content-Type": "application/json", 'Proxy-Connection':'keep-alive'})

response_head_cd = response.headers["Content-Disposition"]
length = float(response.headers['content-length'])

file_name = re.findall("filename=(.+)", response_head_cd)[0]
'''

for file_entry in json.loads(response.content.decode("utf-8"))["data"]["hits"]:
    file_id = file_entry["file_id"]
    data_endpt = "https://api.gdc.cancer.gov/data/{}".format(file_id)
                
    with requests.get(data_endpt, stream=True, headers = {"Content-Type": "application/json", 'Proxy-Connection':'keep-alive'}) as response:
        response_head_cd = response.headers["Content-Disposition"]
        file_name = re.findall("filename=(.+)", response_head_cd)[0]
        #file_name =  re.findall("filename=(.+)")
        total = int(response.headers.get('content-length', 0))
        with open(file_name, 'wb') as file, tqdm(
            desc=file_name,
            total=total,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                bar.update(size)
                