import streamlit as st
import pyrebase
import time
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------
# Firebase 설정
# ---------------------
firebase_config = {
    "apiKey": "AIzaSyCswFmrOGU3FyLYxwbNPTp7hvQxLfTPIZw",
    "authDomain": "sw-projects-49798.firebaseapp.com",
    "databaseURL": "https://sw-projects-49798-default-rtdb.firebaseio.com",
    "projectId": "sw-projects-49798",
    "storageBucket": "sw-projects-49798.appspot.com",
    "messagingSenderId": "812186368395",
    "appId": "1:812186368395:web:be2f7291ce54396209d78e"
}

firebase = pyrebase.initialize_app(firebase_config)
auth = firebase.auth()
firestore = firebase.database()
storage = firebase.storage()

# ---------------------
# 세션 상태 초기화
# ---------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user_email = ""
    st.session_state.id_token = ""
    st.session_state.user_name = ""
    st.session_state.user_gender = "선택 안함"
    st.session_state.user_phone = ""
    st.session_state.profile_image_url = ""

# ---------------------
# 로그인/회원가입/기본 페이지 등 생략 (기존과 동일하게 유지)
# ---------------------

class Home:
    def __init__(self, login_page, register_page, findpw_page):
        st.title("🏠 Home")

        if st.session_state.get("logged_in"):
            st.success(f"{st.session_state.get('user_email')}님 환영합니다.")

        st.markdown("---")
        st.header("📊 Population Trends 분석 안내")

        st.markdown("""
        분석은 **탭(Tab) 구조**로 구성됩니다.  
        *(예: `"기초 통계"`, `"연도별 추이"`, `"지역별 분석"`, `"변화량 분석"`, `"시각화"`)*

        분석에는 다음이 포함되어야 합니다:

        - 🔍 **결측치 및 중복 확인**
        - 📈 **연도별 전체 인구 추이 그래프**
        - 🗺️ **지역별 인구 변화량 순위**
        - 🔼 **증감률 상위 지역 및 연도 도출**
        - 🧩 **누적영역그래프 등 적절한 시각화**
        """)

        st.info("📁 사용 파일: `population_trends.csv`\n\n- 연도별·지역별 인구, 출생아 수, 사망자 수 등 포함된 대한민국 인구 동향 데이터")

# ---------------------
# EDA 페이지 클래스
# ---------------------
class EDA: 
    def __init__(self):
        st.title("📊 Bike Sharing Demand EDA")
        uploaded = st.file_uploader("데이터셋 업로드 (train.csv)", type="csv")
        if not uploaded:
            st.info("train.csv 파일을 업로드 해주세요.")
        else:
            df = pd.read_csv(uploaded, parse_dates=['datetime'])
            st.success("train.csv 로드 완료!")

            # 탭 UI
            tabs = st.tabs([
                "1. 목적 & 절차", "2. 데이터셋 설명", "3. 품질 체크", "4. Datetime 특성", "5. 시각화",
                "6. 상관관계", "7. 이상치 제거", "8. 로그 변환"
            ])

        # ---------------------
        # 📈 Population Trends 분석
        # ---------------------
        st.title("📈 Population Trends 분석")
        pop_file = st.file_uploader("population_trends.csv 업로드", type="csv", key="pop")
        if pop_file:
            df_pop = pd.read_csv(pop_file)
            df_pop.replace("-", 0, inplace=True)
            for col in ['인구', '출생아수(명)', '사망자수(명)']:
                df_pop[col] = pd.to_numeric(df_pop[col], errors='coerce')

            tabs_pop = st.tabs([
                "기초 통계", "연도별 추이", "지역별 분석", "변화량 분석", "시각화"
            ])

            with tabs_pop[0]:
                buffer = io.StringIO()
                df_pop.info(buf=buffer)
                st.text(buffer.getvalue())
                st.dataframe(df_pop.describe())
                st.write("결측치:")
                st.dataframe(df_pop.isnull().sum())
                st.write("중복 행 개수:", df_pop.duplicated().sum())

            with tabs_pop[1]:
                df_nation = df_pop[df_pop['지역'] == '전국']
                fig, ax = plt.subplots()
                sns.lineplot(x='연도', y='인구', data=df_nation, marker='o', ax=ax)
                recent = df_nation.sort_values('연도').tail(3)
                delta = recent['출생아수(명)'].mean() - recent['사망자수(명)'].mean()
                pred = recent['인구'].iloc[-1] + delta * (2035 - recent['연도'].iloc[-1])
                ax.axhline(pred, ls='--', color='gray')
                ax.text(2034, pred, f"Predicted: {int(pred):,}")
                st.pyplot(fig)

            with tabs_pop[2]:
                pivot = df_pop.pivot(index='연도', columns='지역', values='인구')
                delta = pivot.iloc[-1] - pivot.iloc[-6]
                rate = (delta / pivot.iloc[-6]) * 100
                df_delta = pd.DataFrame({
                    '지역': delta.index,
                    '증감량': delta.values / 1000,
                    '증감률(%)': rate.values
                }).query("지역 != '전국'").sort_values("증감량", ascending=False)

                fig, ax = plt.subplots()
                sns.barplot(data=df_delta, x='증감량', y='지역', ax=ax)
                st.pyplot(fig)

            with tabs_pop[3]:
                df_temp = df_pop[df_pop['지역'] != '전국'].copy()
                df_temp.sort_values(['지역', '연도'], inplace=True)
                df_temp['증감'] = df_temp.groupby('지역')['인구'].diff()
                top = df_temp.nlargest(100, '증감')

                def highlight(val):
                    return 'background-color: #a8dadc' if val > 0 else 'background-color: #f4a261'
                st.dataframe(top.style.format({'증감': '{:,.0f}'}).applymap(highlight, subset=['증감']))

            with tabs_pop[4]:
                df_area = df_pop.pivot(index='연도', columns='지역', values='인구')
                if '전국' in df_area.columns:
                    df_area.drop(columns='전국', inplace=True)
                fig, ax = plt.subplots(figsize=(12, 6))
                df_area.plot.area(ax=ax, cmap='tab20')
                ax.set_title("Population Area by Region")
                st.pyplot(fig)

# ---------------------
# 페이지 등록 (예시)
# ---------------------
Page_EDA = st.Page(EDA, title="EDA", icon="📊", url_path="eda")

if st.session_state.logged_in:
    pages = [Page_EDA]
else:
    pages = [Page_EDA]  # 예시용

selected_page = st.navigation(pages)
selected_page.run()