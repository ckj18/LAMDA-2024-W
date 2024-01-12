# from pathlib import Path
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import faiss
# from langchain.vectorstores import FAISS
# import pickle
# from tqdm import tqdm
# import logging
# import time
# import openai
# from dotenv import load_dotenv, find_dotenv
# import os
# from transformers import GPT2Tokenizer

# # 로깅 설정
# logging.basicConfig(level=logging.INFO)

# # 환경변수 로드
# load_dotenv(find_dotenv())

# # OpenAI API 키 설정
# openai.api_key = os.getenv('OPENAI_API_KEY')

# # 텍스트를 임베딩으로 변환하는 함수
# def embed_text(text):
#     response = openai.Embedding.create(
#         model="text-embedding-ada-002",
#         input=text
#     )
#     return response['data'][0]['embedding']

# # 데이터 로딩
# ps = list(Path("best-task-legacy/").glob("**/*.md"))

# data = []
# sources = []
# for p in tqdm(ps, desc="파일 읽기"):
#     with open(p, encoding='UTF8') as f:
#         data.append(f.read())
#     sources.append(p)
#     logging.info(f"{p} 파일 처리 완료")

# text_splitter = RecursiveCharacterTextSplitter(
#     separators=["#", "##", "###"],
#     chunk_size=500,
#     chunk_overlap=10)


# # GPT-2 토크나이저 초기화
# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# MAX_TOKENS = 8000  # 모델이 처리할 수 있는 최대 토큰 수

# def embed_text(text):
#     # 텍스트를 토큰화하고 최대 길이로 자르기
#     tokens = tokenizer.encode(text, truncation=True, max_length=MAX_TOKENS)
#     truncated_text = tokenizer.decode(tokens)

#     # 토큰 길이 확인
#     if len(tokens) > MAX_TOKENS:
#         logging.warning(f"텍스트 길이가 {MAX_TOKENS} 토큰을 초과합니다: {len(tokens)} 토큰")

#     response = openai.Embedding.create(
#         model="text-embedding-ada-002",
#         input=truncated_text
#     )
#     return response['data'][0]['embedding']

# docs = []
# metadatas = []
# for i, d in enumerate(tqdm(data, desc="문서 처리")):
#     splits = text_splitter.split_text(d)
#     docs.extend(splits)
#     metadatas.extend([{"source": sources[i]}] * len(splits))
#     logging.info(f"{sources[i]}에서 {len(splits)}개의 문서 분할 완료")

# # 배치 사이즈와 대기 시간 설정 (RPM 제한 준수)
# batch_size = 83  # 한 번에 처리할 문서의 수
# wait_time = 1    # 각 배치 처리 후 대기할 시간 (초)

# # 문서를 배치로 처리하고 임베딩 생성
# embeddings = []
# for batch_start in tqdm(range(0, len(docs), batch_size), desc="임베딩 처리"):
#     batch_docs = docs[batch_start:batch_start + batch_size]
#     # 임베딩 처리 (대기 시간 포함)
#     batch_embeddings = [embed_text(doc) for doc in batch_docs]
#     embeddings.extend(batch_embeddings)
#     logging.info(f"{batch_start}에서 {batch_start + batch_size}까지 임베딩 처리 완료")
#     time.sleep(wait_time)  # 대기

# # FAISS 저장소 생성 및 인덱스 생성
# store = FAISS(embeddings, metadatas)
# logging.info("FAISS 인덱스 생성 완료")

# # 인덱스 저장
# faiss.write_index(store.index, "docs.index")
# store.index = None

# # FAISS 저장소 저장
# with open("faiss_store.pkl", "wb") as f:
#     pickle.dump(store, f)
#     logging.info("FAISS 저장소 저장 완료")


from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path
from tqdm import tqdm
import faiss
import pickle

model_name = "BAAI/bge-base-en" # large, base
model_kwargs = {'device': 'cuda'}
encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity
model_norm = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

ps = list(Path("best-task-legacy/").glob("**/*.md"))

data = []
sources = []
for p in ps:
    with open(p, encoding='UTF8') as f:
        data.append(f.read())
    sources.append(p)


text_splitter = RecursiveCharacterTextSplitter(
    separators=["#", "##", "###"],
    chunk_size=1500,  # 텍스트 분할 크기 조절
    chunk_overlap=10)

# 문서 처리 부분
docs = []
metadatas = []
for i, d in enumerate(tqdm(data, desc="문서 처리")):  # tqdm으로 진행률 표시
    splits = text_splitter.split_text(d)
    docs.extend(splits)
    metadatas.extend([{"source": sources[i]}] * len(splits))

    
store_name = "tobigs_github"

store = FAISS.from_texts(docs, model_norm, metadatas=metadatas)

faiss.write_index(store.index, "docs.index")
store.index = None
with open("faiss_store.pkl", "wb") as f:
    pickle.dump(store, f)

