import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from pydantic import BaseModel
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 환경 변수 로드
load_dotenv()

class Settings:
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "your-api-key")

settings = Settings()

# FastAPI 앱 생성
app = FastAPI()



# 임베딩 및 벡터 스토어 초기화
embeddings = OpenAIEmbeddings(openai_api_key=settings.OPENAI_API_KEY)
VECTOR_DB_PATH = "./data/faiss_index/"
SETUP_GUIDE_PATH = "./data/setup_guide.md"

# 디렉토리 생성
os.makedirs(VECTOR_DB_PATH, exist_ok=True)

def create_vector_store():
    # setup_guide.md 파일 로드
    loader = TextLoader(SETUP_GUIDE_PATH, encoding="utf-8")
    documents = loader.load()

    # 텍스트 분할
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    # 벡터 스토어 생성
    vector_store = FAISS.from_documents(texts, embeddings)
    vector_store.save_local(VECTOR_DB_PATH)
    return vector_store

# 벡터 스토어 로드 또는 생성
if os.path.exists(os.path.join(VECTOR_DB_PATH, "index.faiss")):
    vector_store = FAISS.load_local(
        folder_path=VECTOR_DB_PATH,
        embeddings=embeddings,
        allow_dangerous_deserialization=True  # 신뢰할 수 있는 소스일 때만 사용
    )
else:
    vector_store = create_vector_store()

# 검색기 초기화
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# LLM 초기화
llm = ChatOpenAI(
    openai_api_key=settings.OPENAI_API_KEY,
    model="gpt-4-turbo",
    temperature=0.5
)

# 프롬프트 템플릿 설정
prompt_template = """
당신은 환경 설정 관련 질문에 답변하는 AI 챗봇입니다. 
사용자의 질문에 대해 신뢰할 수 있는 정보를 제공하세요.

다음은 참고할 문맥 정보입니다:
{context}

사용자의 질문:
{question}

정확하고 간결한 답변을 제공하세요.
"""

prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
# RetrievalQA 체인 생성
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={
        "prompt": prompt,
        "document_variable_name": "context"  # 추가하여 프롬프트에서 `{context}`를 확실하게 매핑
    }
)

# API 모델 정의
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]  # `list` 대신 `list[str]`로 명시

# 챗봇 응답 엔드포인트
@app.post("/chat/", response_model=QueryResponse)
async def chat(query_request: QueryRequest):
    try:
        # ✅ LangChain 0.1.0 이상에서는 invoke() 사용
        result = qa_chain.invoke({"query": query_request.query})  

        return QueryResponse(
            answer=result["result"],
            sources=[doc.metadata.get("source", "Unknown") for doc in result.get("source_documents", [])]
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"입력 오류: {str(ve)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"서버 오류 발생: {str(e)}")

# 검색 API 엔드포인트
@app.get("/search/")
def search(query: str):
    docs = retriever.get_relevant_documents(query)
    return {"query": query, "results": [doc.page_content for doc in docs]}

# 루트 엔드포인트 추가 (서버 상태 확인용)
@app.get("/")
def home():
    return {"message": "RAG 기반 환경설정 가이드 챗봇이 실행 중입니다."}