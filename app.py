from flask import Flask
from api import main_bp, upload_bp
import os

app = Flask(__name__)

# 업로드된 파일을 저장할 디렉토리 설정
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 블루프린트 등록
app.register_blueprint(main_bp)
app.register_blueprint(upload_bp)

if __name__ == "__main__":
    # 업로드 폴더가 없으면 생성
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(port=8000, debug=True)
