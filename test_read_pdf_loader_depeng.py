import pdfplumber
import json
import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

questions = json.load(open("question.json"))
print(questions[0])


pdf = pdfplumber.open("/mnt/AI/hr_material_test_lv/Finance-Guidance-China.pdf")
len(pdf.pages)
print(pdf.pages[15].extract_text()) # 可复制的PDF


pdf_content = []
for page_idx in range(len(pdf.pages)):
    pdf_content.append({
        'page': 'page_' + str(page_idx + 1),
        'content': pdf.pages[page_idx].extract_text()
    })
#对文本进行分词
question_words = [' '.join(jieba.lcut(x['question'])) for x in questions]
pdf_content_words = [' '.join(jieba.lcut(x['content'])) for x in pdf_content]
#提取TFIDF
tfidf = TfidfVectorizer()
tfidf.fit(question_words + pdf_content_words)

question_feat = tfidf.transform(question_words) #转换为矩阵
pdf_content_feat = tfidf.transform(pdf_content_words)

question_feat = normalize(question_feat)
pdf_content_feat = normalize(pdf_content_feat)

#通过TFIDF进行检索
for query_idx, feat in enumerate(question_feat):
    #对每个提问和每页PDF进行打分
    score = feat @ pdf_content_feat.T
    score = score.toarray()[0]
    max_score_page_idx = score.argsort()[-1] + 1

    questions[query_idx]['reference'] = 'page_' + str(max_score_page_idx)

with open('submit_tfidf_retrieval_top1.json', 'w', encoding='utf8') as up:
    json.dump(questions, up, ensure_ascii=False, indent=4)