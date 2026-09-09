import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# -----------------------------
# 기본 설정
# -----------------------------
st.set_page_config(
    page_title="서울 기온 예측기",
    page_icon="🌡️",
    layout="wide",
)

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)
MAX_DATA_YEAR = 2025
MIN_OBSERVATION_DAYS = 300


# -----------------------------
# 데이터 불러오기 및 전처리
# -----------------------------
@st.cache_data
def load_yearly_temperature():
    # UTF-8 인코딩으로 서울 기온 데이터 불러오기
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    required_columns = ["날짜", "지점", "평균기온", "최저기온", "최고기온"]
    missing_columns = [
        column for column in required_columns if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "데이터에 필요한 열이 없습니다: " + ", ".join(missing_columns)
        )

    # 날짜와 평균기온을 분석 가능한 형태로 변환
    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
    df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")

    # 날짜와 평균기온이 모두 있는 관측만 사용
    valid_df = df.dropna(subset=["날짜", "평균기온"]).copy()
    valid_df["연도"] = valid_df["날짜"].dt.year

    # 수업 기준 기간인 2025년까지만 사용
    valid_df = valid_df[valid_df["연도"] <= MAX_DATA_YEAR]

    # 연도별 실제 관측일 수와 연평균기온 계산
    yearly = (
        valid_df.groupby("연도", as_index=False)
        .agg(
            관측일수=("날짜", "nunique"),
            연평균기온=("평균기온", "mean"),
        )
    )

    # 관측일이 300일 이상인 해만 회귀 분석에 사용
    yearly = yearly[yearly["관측일수"] >= MIN_OBSERVATION_DAYS].copy()
    yearly = yearly.sort_values("연도").reset_index(drop=True)

    return yearly


# -----------------------------
# 단순 선형회귀 계산
# 외부 머신러닝 라이브러리 없이 최소제곱법 사용
# -----------------------------
def calculate_regression(years, temperatures):
    x = pd.Series(years, dtype="float64")
    y = pd.Series(temperatures, dtype="float64")

    x_mean = x.mean()
    y_mean = y.mean()

    denominator = ((x - x_mean) ** 2).sum()

    if denominator == 0:
        raise ValueError("회귀 직선을 계산하려면 서로 다른 연도가 필요합니다.")

    slope = ((x - x_mean) * (y - y_mean)).sum() / denominator
    intercept = y_mean - slope * x_mean
    correlation = x.corr(y)

    return slope, intercept, correlation


# -----------------------------
# 화면 구성
# -----------------------------
st.title("🌡️ 서울 기온 예측기")
st.caption(
    "1900~2100년 중 연도를 선택하면, 서울의 연평균기온 추세를 "
    "바탕으로 예상 기온을 계산합니다."
)

try:
    yearly_df = load_yearly_temperature()

    if len(yearly_df) < 2:
        st.error("조건을 만족하는 연도가 부족하여 회귀 분석을 할 수 없습니다.")
        st.stop()

    slope, intercept, correlation = calculate_regression(
        yearly_df["연도"],
        yearly_df["연평균기온"],
    )

    selected_year = st.slider(
        "예측할 연도를 선택하세요.",
        min_value=1900,
        max_value=2100,
        value=2025,
        step=1,
    )

    predicted_temperature = slope * selected_year + intercept

    # 예측값을 크게 표시
    st.metric(
        label=f"{selected_year}년 예상 연평균기온",
        value=f"{predicted_temperature:.2f} ℃",
    )

    if selected_year > MAX_DATA_YEAR:
        st.info(
            "2025년 이후 값은 관측 자료가 아니라, 2025년까지의 장기적인 "
            "선형 추세를 연장한 예측값입니다."
        )

    start_year = int(yearly_df["연도"].min())
    end_year = int(yearly_df["연도"].max())
    year_count = len(yearly_df)

    info_col1, info_col2, info_col3, info_col4 = st.columns(4)

    info_col1.metric("회귀 직선을 만든 연도 수", f"{year_count}개")
    info_col2.metric("시작 연도", f"{start_year}년")
    info_col3.metric("끝 연도", f"{end_year}년")
    info_col4.metric("연도-기온 상관계수", f"{correlation:.4f}")

    # 회귀 직선은 슬라이더 전체 범위에 맞게 표시
    line_years = pd.Series(range(1900, 2101))
    line_temperatures = slope * line_years + intercept

    fig = go.Figure()

    # 실제 연평균기온 산점도
    fig.add_trace(
        go.Scatter(
            x=yearly_df["연도"],
            y=yearly_df["연평균기온"],
            mode="markers",
            name="관측 연평균기온",
            marker=dict(
                color="#1f77b4",
                size=7,
                opacity=0.75,
            ),
            customdata=yearly_df[["관측일수"]],
            hovertemplate=(
                "<b>%{x}년</b><br>"
                "연평균기온: %{y:.2f} ℃<br>"
                "관측일수: %{customdata[0]}일"
                "<extra></extra>"
            ),
        )
    )

    # 회귀 직선
    fig.add_trace(
        go.Scatter(
            x=line_years,
            y=line_temperatures,
            mode="lines",
            name="선형 회귀 직선",
            line=dict(color="#e45756", width=3),
            hovertemplate=(
                "<b>%{x}년</b><br>"
                "회귀 예측: %{y:.2f} ℃"
                "<extra></extra>"
            ),
        )
    )

    # 사용자가 선택한 연도의 예측값
    fig.add_trace(
        go.Scatter(
            x=[selected_year],
            y=[predicted_temperature],
            mode="markers",
            name=f"{selected_year}년 예측",
            marker=dict(
                color="#ffbf00",
                size=15,
                symbol="star",
                line=dict(color="#333333", width=1),
            ),
            hovertemplate=(
                f"<b>{selected_year}년 예측</b><br>"
                f"{predicted_temperature:.2f} ℃"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=(
            "서울 연평균기온과 선형 회귀 직선"
            f"<br><sup>상관계수 r = {correlation:.4f}</sup>"
        ),
        xaxis_title="연도",
        yaxis_title="연평균기온 (℃)",
        xaxis=dict(range=[1900, 2100]),
        hovermode="closest",
        template="plotly_white",
        height=620,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
    )

    st.plotly_chart(fig, use_container_width=True)

    st.write(
        f"회귀식: **예상 연평균기온 = "
        f"{slope:.6f} × 연도 {intercept:+.6f}**"
    )

    with st.expander("분석에 사용한 연도별 데이터 보기"):
        display_df = yearly_df.copy()
        display_df["연평균기온"] = display_df["연평균기온"].round(2)

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
        )

    st.caption(
        "자료 처리 기준: 2025년 이전 또는 해당 연도 자료만 사용하며, "
        "평균기온이 기록된 관측일이 300일 이상인 해만 분석에 포함했습니다. "
        "선형 회귀 예측은 장기 추세를 단순화한 값이므로 실제 미래 기온과 "
        "차이가 날 수 있습니다."
    )

except Exception as error:
    st.error(f"데이터 처리 중 오류가 발생했습니다: {error}")
