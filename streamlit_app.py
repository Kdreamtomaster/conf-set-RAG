import streamlit as st
import requests
import os
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# FastAPI 서버 URL (환경 변수에서 불러오기)
API_URL = os.getenv("API_URL", "http://backend:8000") + "/chat/"
SEARCH_API_URL = os.getenv("SEARCH_API_URL", "http://backend:8000/search")

# 🖌️ Streamlit 스타일 커스터마이징 (하늘색 테마 적용)
st.markdown(
    """
    <style>
        /* 배경색 설정 */
        body {
            background-color: #E3F2FD;  /* 연한 하늘색 */
        }

        /* 기본 컨텐츠 배경 설정 */
        .stApp {
            background-color: #E3F2FD;
        }

        /* 제목 스타일 */
        .stMarkdown h1 {
            color: #0D47A1;  /* 짙은 하늘색 */
            text-align: center;
        }

        /* 버튼 스타일 */
        .stButton>button {
            background-color: #42A5F5 !important; /* 파란색 */
            color: white !important;  /* 글씨 색상 유지 */
            border-radius: 10px;
            font-size: 16px;
            padding: 8px 20px;
            transition: 0.3s;
        }

        /* 마우스 호버 시 스타일 */
        .stButton>button:hover {
            background-color: #1976D2 !important; /* 진한 파란색 */
            color: white !important; /* 🔥 글씨 색상을 하얀색으로 유지 */
        }

        /* 입력 필드 스타일 */
        .stTextInput>div>div>input {
            border-radius: 10px;
            border: 2px solid #64B5F6;
            background-color: #BBDEFB;
            color: black;
        }

        /* 로딩 스피너 색상 */
        .stSpinner {
            color: #42A5F5;
        }

        /* 검색 결과 박스 스타일 */
        .stMarkdown p {
            background-color: #BBDEFB;
            padding: 10px;
            border-radius: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# 🌤️ UI 시작
st.title("🌤️ RAG 기반 환경설정 가이드 챗봇")
st.write("환경설정 관련 질문을 입력하세요.")

# 🔍 사용자 입력
query = st.text_input("💬 질문을 입력하세요:")

if st.button("질문하기 🚀") and query:
    with st.spinner("💭 답변을 생성 중입니다..."):
        try:
            response = requests.post(API_URL, json={"query": query}, timeout=45)
            response.raise_for_status()
            data = response.json()

            st.subheader("📌 답변:")
            st.write(data["answer"])

            # 출처 문서 표시
            if data.get("sources"):
                st.subheader("📖 참고 문서:")
                for source in data["sources"]:
                    st.markdown(f"📂 **{source}**")
        except requests.exceptions.RequestException as e:
            st.error(f"⚠️ API 요청 실패: {e}")

# 🔎 **관련 문서 검색**
st.subheader("🔎 관련 문서 검색")
search_query = st.text_input("🔍 검색어를 입력하세요:")

if st.button("검색 🔍") and search_query:
    with st.spinner("📚 검색 중입니다..."):
        try:
            search_response = requests.get(SEARCH_API_URL, params={"query": search_query}, timeout=30)
            search_response.raise_for_status()
            search_data = search_response.json()

            st.write("### 📖 검색 결과")

            if search_data.get("results"):
                top_results = search_data["results"][:3]  # 상위 3개만 출력

                for idx, result in enumerate(top_results, 1):
                    with st.expander(f"📌 결과 {idx} 보기"):
                        st.write(result)

            else:
                st.write("❌ 검색 결과가 없습니다.")

        except requests.exceptions.RequestException as e:
            st.error(f"⚠️ 검색 API 요청 실패: {e}")
