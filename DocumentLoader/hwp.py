from langchain_teddynote.document_loaders import HWPLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS

#HWP Loader 객체 생성
loader = HWPLoader("../Document/학생연구자지원규정.hwp")

#문서 로드
docs = loader.load()
# print(docs[0].page_content)

#문서를 로드해서 token화하는 작업
tik_text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
encoding_name="cl100k_base",
chunk_size=250,
chunk_overlap=25
)
docs = loader.load_and_split(text_splitter=tik_text_splitter)

# for i, doc in enumerate(docs, 1):
#     print(doc.page_content)
   
# 임베딩 모델
model_name = "intfloat/multilingual-e5-large-instruct"
model_kwargs = {'device': "cuda"}
encode_kwargs = {'normalize_embeddings': True}
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
)

# VectorDB로는 FAISS를 사용하여 구성하였음
vectorstore = FAISS.from_documents(docs, embedding=embeddings)
retriever = vectorstore.as_retriever()

query = "산학협력단의 의무"
answer = retriever.invoke(query)

for i, doc in enumerate(answer, 1):
    print(i,"=", doc.page_content)
