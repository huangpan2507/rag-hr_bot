import sys
import json
from typing import Any, Dict

from langchain.storage import LocalFileStore
from langchain.storage import LocalFileStore
from langchain.storage._lc_store import create_kv_docstore
from langchain_core.documents import Document


u = '你好'
str1 = u.encode('utf-8')
s = json.dumps(u, ensure_ascii=False).encode("utf-8")
print("before",s)         # b'"\xe4\xbd\xa0\xe5\xa5\xbd"'

s1 = s.decode("utf-8")    
print(s1)                 # "你好"
print(str1)               # b'\xe4\xbd\xa0\xe5\xa5\xbd'

#用loads将json编码成python
print(json.loads(s))      # 你好

# 等价于
# s = '你好 中文'
# s.decode('ascii').encode('utf-8')


def dumps(obj: Any, *, pretty: bool = False, ensure_ascii: bool = False, **kwargs: Any) -> str:
    """Return a json string representation of an object."""
    
    return json.dumps(obj, ensure_ascii=ensure_ascii, **kwargs)
        

s2 = dumps(u, ensure_ascii=False).encode("utf-8")
# print result： new dumps: b'"\xe4\xbd\xa0\xe5\xa5\xbd"'
print(f"\n new dumps: {s2} \n")

# print result: 你好
print(json.loads(s2))