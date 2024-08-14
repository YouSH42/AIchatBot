import os
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain_teddynote.document_loaders import HWPLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS

# PDF, Word, HWP 파일들이 있는 디렉터리 경로 설정
directory_path = "/mnt/c/Users/유상현/Desktop/RAG용 자료"

# 텍스트 분할기 초기화
tik_text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base",
    chunk_size=250,
    chunk_overlap=25
)

# 모든 문서 로드
docs = []
metadata = []
for filename in os.listdir(directory_path):
    file_path = os.path.join(directory_path, filename)
    
    # 파일 확장자에 따라 로더 선택
    if filename.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif filename.endswith(".hwp"):
        loader = HWPLoader(file_path)
    else:
        print(f"Unsupported file format: {filename}")
        continue  # 지원하지 않는 형식의 파일은 무시

    # 문서를 로드하고 분할하여 docs에 추가
    loaded_docs = loader.load_and_split(text_splitter=tik_text_splitter)
    for doc in loaded_docs:
        doc.metadata = {"source": filename}  # 각 문서에 파일명을 메타데이터로 추가
        docs.append(doc)

# 임베딩 모델 설정
model_name = "intfloat/multilingual-e5-large-instruct"
model_kwargs = {'device': "cuda"}
encode_kwargs = {'normalize_embeddings': True}
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
)

# VectorDB로 Chroma 사용하여 문서 임베딩
# vectorstore = Chroma.from_documents(docs, embedding=embeddings)
vectorstore = FAISS.from_documents(docs, embedding=embeddings)
retriever = vectorstore.as_retriever()

# 질의 처리
query = "who is trump?"
start_time = time.time()
answer = retriever.invoke(query)
end_time = time.time()

# 검색 시간 측정
search_time = end_time - start_time

# 결과 출력
for i, doc in enumerate(answer, 1):
    source = doc.metadata.get("source", "Unknown source")  # 메타데이터에서 파일명 가져오기
    print(f"{i} = {doc.page_content}\n출처: {source}")
    
print(f"검색 시간: {search_time} 초")
