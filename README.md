# AIchatBot

- 이 코드는 teddynote님의 유튜브 강좌를 보고 작성한 코드입니다.
[![Video Label](https://img.youtube.com/vi/VkcaigvTrug/0.jpg)](https://youtu.be/VkcaigvTrug?feature=shared)
- main.py 코드만으로 로컬에서 돌아갈 수 있는 ai챗봇입니다.
- ollama를 사용하여 모델을 로드한 다음 로컬에서 돌렸습니다.
- langchain을 사용하여 RAG도 적용하였습니다.
- 위 코드를 실행하기 위해서는 gpu 가속도 필요합니다.
- 또한 임베딩 모델을 따로 저장하는 코드가 없으므로 환경변수 설정을 해주어야합니다.(코드 참조)
```bash
pip install streamlit torch langchain langchain_core langchain_huggingface langchain_community unstructured faiss-cpu numpy pandas requests lxml python-docx
```
- 아래의 명령어로 작동이 가능합니다.
```bash
streamlit run main.py
```

### 한글(HWP) 문서도 로드할 수 있도록 TeddyNote님이 만들어주셨길래
- 아무래도 한글을 한국에서는 많이 쓰니까 요것도 적용해보려고 합니다.
- 참고문서는 TeddyNote님의 위키독스를 참고했습니다.

### pdf로더가 한글 파일을 제대로 인식하지 못하는 것 같아 몇가지 테스드도 진행하였습니다.
- DocumentLoader 폴더에서 한글(HWP)로더를 사용한 것과 PDF로더를 사용하여 임베딩한다음 결과값을 출력하였을 때 내용 구성에 있어서 상이한 점을 발견하였습니다.