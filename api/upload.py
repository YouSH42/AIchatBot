from flask import request, redirect, url_for
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_huggingface import HuggingFaceEmbeddings
import torch
import os
from . import upload_bp

UPLOAD_FOLDER = 'uploads'

@upload_bp.route("/upload", methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        
        # 문서를 로드하고 자르는 부분
        loader = PyPDFLoader(filepath)
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size = 200,
            chunk_overlap = 0
        )

        docs = loader.load_and_split(text_splitter)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(device)
        
        # 모델 임베딩 설정
        model_name = "bespin-global/klue-sroberta-base-continue-learning-by-mnr"
        model_kwargs = {'device': device}
        encode_kwargs = {'normalize_embeddings': True}
        hf = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        # 임베딩한 정보를 벡터DB에 넣는 코드
        db = Chroma.from_documents(docs, hf, persist_directory="/home/sanghyun42/practice/web/db")
        print("Documents have been embedded and stored in the VectorDB.")
        
        return redirect(url_for('main.main'))


