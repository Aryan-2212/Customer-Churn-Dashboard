import json
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

from analytics_context import build_behavioral_summary
from llm_assistant import (
    generate_fallback_response,
    get_gemini_debug_status,
    generate_llm_response,
    get_gemini_api_key,
    get_gemini_model,
    is_gemini_available,
)
from prompt_template import build_dashboard_context_payload

st.write("Secrets:", st.secrets)
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
DATA_PATH = PROJECT_ROOT / "BankChurners_Cleaned.csv"
INSIGHTS_PATH = PROJECT_ROOT / "churn_insights.json"
DEFAULT_POWER_BI_REPORT_URL = (
    "https://app.powerbi.com/groups/me/reports/"
    "142efc4a-bb5c-43b5-8d54-297b41ab74d6/dc6b1f534010170e5201?experience=power-bi"
)
COLOR_ACTIVE = "#ff5563"
COLOR_CHURN = "#a6acb8"

DASHBOARD_PAGES = [
    (
        "Customer Overview",
        "High-level KPIs for customer base, attrition rate, and overall account health.",
    ),
    (
        "Demographics",
        "Age, gender, education, marital status, and income distribution patterns.",
    ),
    (
        "Card Category Analysis",
        "How card type and credit profile correlate with churn outcomes.",
    ),
    (
        "Transaction Behavior",
        "Transaction count, amount, and spend change patterns tied to churn.",
    ),
    (
        "Credit Usage",
        "Credit limit, revolving balance, and utilization relationships.",
    ),
    (
        "Customer Engagement",
        "Inactivity, contacts, tenure, and relationship signals connected to churn.",
    ),
]


st.set_page_config(
    page_title="AI-Powered Customer Churn Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, low_memory=False)

    numeric_columns = [
        "Attrition_Flag",
        "Customer_Age",
        "Dependent_count",
        "Months_on_book",
        "Total_Relationship_Count",
        "Months_Inactive_12_mon",
        "Contacts_Count_12_mon",
        "Credit_Limit",
        "Total_Revolving_Bal",
        "Avg_Open_To_Buy",
        "Total_Amt_Chng_Q4_Q1",
        "Total_Trans_Amt",
        "Total_Trans_Ct",
        "Total_Ct_Chng_Q4_Q1",
        "Avg_Utilization_Ratio",
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    category_columns = [
        "Gender",
        "Education_Level",
        "Marital_Status",
        "Income_Category",
        "Card_Category",
    ]
    for column in category_columns:
        if column in df.columns:
            df[column] = (
                df[column]
                .fillna("Unknown")
                .replace(["", "nan", "NaN", "None"], "Unknown")
                .astype(str)
            )

    return df.dropna(subset=["Attrition_Flag"])


@st.cache_data
def load_insights() -> dict:
    with INSIGHTS_PATH.open(encoding="utf-8") as insights_file:
        return json.load(insights_file)


def get_power_bi_links() -> tuple[str, str]:
    try:
        embed_url = str(st.secrets.get("POWER_BI_EMBED_URL", ""))
        report_url = str(st.secrets.get("POWER_BI_REPORT_URL", DEFAULT_POWER_BI_REPORT_URL))
    except StreamlitSecretNotFoundError:
        embed_url = ""
        report_url = DEFAULT_POWER_BI_REPORT_URL

    embed_url = embed_url or os.getenv("POWER_BI_EMBED_URL", "")
    report_url = report_url or os.getenv("POWER_BI_REPORT_URL", DEFAULT_POWER_BI_REPORT_URL)
    return embed_url, report_url


def apply_filters(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    st.sidebar.markdown('<div class="sidebar-title">Filters</div>', unsafe_allow_html=True)

    age_min = int(df["Customer_Age"].min())
    age_max = int(df["Customer_Age"].max())
    selected_age = st.sidebar.slider("Age Range", age_min, age_max, (age_min, age_max))

    inactivity_min = int(df["Months_Inactive_12_mon"].min())
    inactivity_max = int(df["Months_Inactive_12_mon"].max())
    selected_inactivity = st.sidebar.slider(
        "Inactive Months", inactivity_min, inactivity_max, (inactivity_min, inactivity_max)
    )

    gender_options = ["All"] + [
        value for value in sorted(df["Gender"].unique().tolist()) if value != "Unknown"
    ]
    selected_gender = st.sidebar.selectbox("Gender", gender_options, index=0)

    income_options = ["All"] + [
        value for value in sorted(df["Income_Category"].unique().tolist()) if value != "Unknown"
    ]
    selected_income = st.sidebar.selectbox("Income Category", income_options, index=0)

    card_options = ["All"] + [
        value for value in sorted(df["Card_Category"].unique().tolist()) if value != "Unknown"
    ]
    selected_card = st.sidebar.selectbox("Card Category", card_options, index=0)

    attrition_options = ["All", "Active", "Churned"]
    selected_attrition = st.sidebar.selectbox("Attrition Status", attrition_options, index=0)

    attrition_values = [0, 1]
    if selected_attrition == "Active":
        attrition_values = [0]
    elif selected_attrition == "Churned":
        attrition_values = [1]

    filtered_df = df[
        (df["Customer_Age"].between(selected_age[0], selected_age[1]))
        & (df["Months_Inactive_12_mon"].between(selected_inactivity[0], selected_inactivity[1]))
        & (df["Attrition_Flag"].isin(attrition_values))
    ].copy()

    if selected_gender != "All":
        filtered_df = filtered_df[filtered_df["Gender"] == selected_gender]
    if selected_income != "All":
        filtered_df = filtered_df[filtered_df["Income_Category"] == selected_income]
    if selected_card != "All":
        filtered_df = filtered_df[filtered_df["Card_Category"] == selected_card]

    filter_context = {
        "age_range": {"min": selected_age[0], "max": selected_age[1]},
        "inactive_months_range": {"min": selected_inactivity[0], "max": selected_inactivity[1]},
        "gender": selected_gender,
        "income_category": selected_income,
        "card_category": selected_card,
        "attrition_status": selected_attrition,
    }

    st.sidebar.caption(f"Filtered customers: {len(filtered_df):,}")
    return filtered_df, filter_context


def churn_label_series(df: pd.DataFrame) -> pd.Series:
    return df["Attrition_Flag"].map({0: "Active", 1: "Churned"})


def base_layout(fig: go.Figure, title: str, height: int = 320) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0.01, y=0.96, xanchor="left", yanchor="top"),
        height=height,
        template="plotly_dark",
        paper_bgcolor="#171b23",
        plot_bgcolor="#171b23",
        margin=dict(l=20, r=20, t=95, b=20),
        font=dict(color="#f3f6fb"),
        title_font=dict(size=16, color="#f7f9fc"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.18,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#2a3242", zeroline=False, color="#f3f6fb")
    fig.update_yaxes(showgrid=True, gridcolor="#2a3242", zeroline=False, color="#f3f6fb")
    return fig


def create_kpi_card(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_header(filtered_df: pd.DataFrame, full_df: pd.DataFrame, insights: dict) -> None:
    header_cols = st.columns([1.1, 1, 1, 1, 1])
    inactive_customers = int((filtered_df["Months_Inactive_12_mon"] >= 3).sum())
    avg_credit_limit = filtered_df["Credit_Limit"].mean()

    with header_cols[0]:
        st.markdown(
            """
            <div class="title-card">
                <div class="title-card-text">CUSTOMER<br>CHURN<br>DASHBOARD</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with header_cols[1]:
        create_kpi_card("Total Customer", f"{len(filtered_df)/1000:.3f}K")
    with header_cols[2]:
        create_kpi_card("Churned Customer", f"{filtered_df['Attrition_Flag'].sum()/1000:.3f}K")
    with header_cols[3]:
        create_kpi_card("Inactive Customers", f"{inactive_customers/1000:.3f}K")
    with header_cols[4]:
        create_kpi_card("Average Credit Limit", f"{avg_credit_limit/1000:.2f}K")

    st.markdown(
        f"""
        <div class="hero-note">
            Filtered churn rate: <strong>{filtered_df['Attrition_Flag'].mean() * 100:.2f}%</strong>
            &nbsp;&nbsp;|&nbsp;&nbsp;
            Full dataset customers: <strong>{len(full_df):,}</strong>
            &nbsp;&nbsp;|&nbsp;&nbsp;
            Best model: <strong>{insights.get("best_model", "Random Forest")}</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_dataset_summary(filtered_df: pd.DataFrame) -> None:
    summary_cols = st.columns([1.25, 1])

    with summary_cols[0]:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-heading">Dataset Summary</div>', unsafe_allow_html=True)
        preview_columns = [
            "Customer_Age",
            "Gender",
            "Income_Category",
            "Card_Category",
            "Months_Inactive_12_mon",
            "Total_Trans_Amt",
            "Total_Trans_Ct",
            "Avg_Utilization_Ratio",
            "Attrition_Flag",
        ]
        preview_columns = [col for col in preview_columns if col in filtered_df.columns]
        st.dataframe(filtered_df[preview_columns].head(12), use_container_width=True, height=330)
        st.markdown("</div>", unsafe_allow_html=True)

    with summary_cols[1]:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-heading">Dashboard Context</div>', unsafe_allow_html=True)
        for page_name, description in DASHBOARD_PAGES:
            st.markdown(f"**{page_name}**: {description}")
        st.markdown("</div>", unsafe_allow_html=True)


def render_model_insights(filtered_df: pd.DataFrame, insights: dict) -> dict:
    top_drivers = insights.get("top_churn_drivers", [])
    behavior = build_behavioral_summary(filtered_df)
    patterns = behavior["behavioral_patterns"]
    segment_snapshot = behavior["segment_risk_snapshot"]

    insight_cols = st.columns([1, 1, 1])
    cards = [
        (
            "Top Churn Drivers",
            "<br>".join(f"{index}. {driver}" for index, driver in enumerate(top_drivers, start=1)),
        ),
        (
            "Behavior Snapshot",
            (
                f"Transactions drop from <strong>{patterns['avg_transaction_count_retained']:.1f}</strong> to "
                f"<strong>{patterns['avg_transaction_count_churned']:.1f}</strong><br>"
                f"Inactivity rises from <strong>{patterns['avg_inactive_months_retained']:.1f}</strong> to "
                f"<strong>{patterns['avg_inactive_months_churned']:.1f}</strong>"
            ),
        ),
        (
            "Segment Risk",
            (
                f"Highest-risk card category: <strong>{segment_snapshot['highest_risk_card_category']}</strong><br>"
                f"Churn rate in that segment: <strong>{segment_snapshot['highest_risk_card_churn_rate_percent']:.2f}%</strong>"
            ),
        ),
    ]

    for column, (title, content) in zip(insight_cols, cards):
        with column:
            st.markdown(
                f"""
                <div class="section-card model-card">
                    <div class="model-card-title">{title}</div>
                    <div class="model-card-content">{content}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    return behavior


def create_visualizations(filtered_df: pd.DataFrame) -> list[go.Figure]:
    df = filtered_df.copy()
    df["Attrition_Label"] = churn_label_series(df)

    age_hist = px.histogram(
        df,
        x="Customer_Age",
        nbins=35,
        color_discrete_sequence=[COLOR_ACTIVE],
    )
    age_hist.update_traces(marker_line_width=0)
    base_layout(age_hist, "Age Distribution")

    gender_counts = df["Gender"].value_counts().reset_index()
    gender_counts.columns = ["Gender", "Count"]
    gender_donut = px.pie(
        gender_counts,
        names="Gender",
        values="Count",
        hole=0.58,
        color="Gender",
        color_discrete_sequence=[COLOR_ACTIVE, "#c2c2c2"],
    )
    base_layout(gender_donut, "Gender Distribution")

    income_order = [
        "Less than $40K",
        "$40K - $60K",
        "$60K - $80K",
        "$80K - $120K",
        "$120K +",
        "Unknown",
    ]
    income_counts = (
        df["Income_Category"]
        .value_counts()
        .rename_axis("Income_Category")
        .reset_index(name="Count")
    )
    income_counts["Income_Category"] = pd.Categorical(
        income_counts["Income_Category"], categories=income_order, ordered=True
    )
    income_counts = income_counts.sort_values("Income_Category")
    income_bar = px.bar(
        income_counts,
        x="Count",
        y="Income_Category",
        orientation="h",
        color_discrete_sequence=[COLOR_ACTIVE],
        text_auto=".2s",
    )
    base_layout(income_bar, "Income Category")

    churn_counts = (
        df["Attrition_Label"].value_counts().rename_axis("Attrition").reset_index(name="Count")
    )
    churn_donut = px.pie(
        churn_counts,
        names="Attrition",
        values="Count",
        hole=0.62,
        color="Attrition",
        color_discrete_map={"Active": COLOR_ACTIVE, "Churned": COLOR_CHURN},
    )
    base_layout(churn_donut, "Customer Churn Distribution")

    card_dist = (
        df.groupby(["Card_Category", "Attrition_Label"]).size().reset_index(name="Count")
    )
    card_bar = px.bar(
        card_dist,
        x="Count",
        y="Card_Category",
        color="Attrition_Label",
        orientation="h",
        barmode="group",
        color_discrete_map={"Active": COLOR_ACTIVE, "Churned": COLOR_CHURN},
        text_auto=".2s",
    )
    base_layout(card_bar, "Customer Distribution by Card Category")

    gender_attr = df.groupby(["Gender", "Attrition_Label"]).size().reset_index(name="Count")
    gender_bar = px.bar(
        gender_attr,
        x="Gender",
        y="Count",
        color="Attrition_Label",
        barmode="stack",
        color_discrete_map={"Active": COLOR_ACTIVE, "Churned": COLOR_CHURN},
    )
    base_layout(gender_bar, "Gender Count")

    transaction_scatter = px.scatter(
        df,
        x="Total_Trans_Ct",
        y="Total_Trans_Amt",
        color="Attrition_Label",
        color_discrete_map={"Active": COLOR_ACTIVE, "Churned": COLOR_CHURN},
        opacity=0.7,
    )
    base_layout(transaction_scatter, "Transaction Activity vs Churn")

    inactivity_bar = (
        df.groupby(["Months_Inactive_12_mon", "Attrition_Label"])
        .size()
        .reset_index(name="Count")
        .sort_values("Months_Inactive_12_mon")
    )
    inactivity_chart = px.bar(
        inactivity_bar,
        x="Count",
        y="Months_Inactive_12_mon",
        color="Attrition_Label",
        orientation="h",
        barmode="group",
        color_discrete_map={"Active": COLOR_ACTIVE, "Churned": COLOR_CHURN},
    )
    base_layout(inactivity_chart, "Customer Inactivity Analysis")

    credit_usage_scatter = px.scatter(
        df,
        x="Credit_Limit",
        y="Total_Revolving_Bal",
        color="Attrition_Label",
        color_discrete_map={"Active": COLOR_ACTIVE, "Churned": COLOR_CHURN},
        opacity=0.65,
    )
    base_layout(credit_usage_scatter, "Credit Usage vs Credit Limit")

    months_book = (
        df.groupby(["Months_on_book", "Attrition_Label"])["Months_on_book"]
        .sum()
        .reset_index(name="Sum_Months_on_book")
    )
    months_book_chart = px.bar(
        months_book,
        x="Sum_Months_on_book",
        y="Months_on_book",
        orientation="h",
        color="Attrition_Label",
        barmode="group",
        color_discrete_map={"Active": COLOR_ACTIVE, "Churned": COLOR_CHURN},
    )
    base_layout(months_book_chart, "Months on Book by Attrition")

    trans_by_inactive = (
        df.groupby(["Months_Inactive_12_mon", "Attrition_Label"])["Total_Trans_Ct"]
        .sum()
        .reset_index(name="Total_Trans_Ct_Sum")
    )
    trans_inactive_scatter = px.scatter(
        trans_by_inactive,
        x="Months_Inactive_12_mon",
        y="Total_Trans_Ct_Sum",
        color="Attrition_Label",
        color_discrete_map={"Active": COLOR_ACTIVE, "Churned": COLOR_CHURN},
    )
    base_layout(trans_inactive_scatter, "Total Transaction Count vs Inactive Months")

    trans_amount_card = (
        df.groupby(["Card_Category", "Attrition_Label"])["Total_Trans_Amt"]
        .sum()
        .reset_index(name="Total_Trans_Amt_Sum")
    )
    trans_amount_chart = px.bar(
        trans_amount_card,
        x="Total_Trans_Amt_Sum",
        y="Card_Category",
        orientation="h",
        color="Attrition_Label",
        barmode="group",
        color_discrete_map={"Active": COLOR_ACTIVE, "Churned": COLOR_CHURN},
        text_auto=".2s",
    )
    base_layout(trans_amount_chart, "Transaction Amount by Card Category")

    credit_limit_hist = px.histogram(
        df,
        x="Credit_Limit",
        nbins=100,
        color_discrete_sequence=[COLOR_ACTIVE],
    )
    base_layout(credit_limit_hist, "Count of Credit Limit")

    credit_vs_trans = px.scatter(
        df,
        x="Credit_Limit",
        y="Total_Trans_Amt",
        color="Attrition_Label",
        color_discrete_map={"Active": COLOR_ACTIVE, "Churned": COLOR_CHURN},
        opacity=0.7,
    )
    base_layout(credit_vs_trans, "Credit Limit and Transaction Amount")

    education_attr = (
        df.groupby(["Education_Level", "Attrition_Label"]).size().reset_index(name="Count")
    )
    education_chart = px.bar(
        education_attr,
        x="Count",
        y="Education_Level",
        orientation="h",
        color="Attrition_Label",
        barmode="group",
        color_discrete_map={"Active": COLOR_ACTIVE, "Churned": COLOR_CHURN},
    )
    base_layout(education_chart, "Education Level vs Attrition")

    avg_credit_income = (
        df.groupby(["Income_Category", "Attrition_Label"])["Credit_Limit"]
        .mean()
        .reset_index(name="Average_Credit_Limit")
    )
    avg_credit_income["Income_Category"] = pd.Categorical(
        avg_credit_income["Income_Category"], categories=income_order, ordered=True
    )
    avg_credit_income = avg_credit_income.sort_values("Income_Category")
    avg_credit_income_chart = px.bar(
        avg_credit_income,
        x="Average_Credit_Limit",
        y="Income_Category",
        orientation="h",
        color="Attrition_Label",
        barmode="group",
        color_discrete_map={"Active": COLOR_ACTIVE, "Churned": COLOR_CHURN},
    )
    base_layout(avg_credit_income_chart, "Average Credit Limit by Income Category")

    category_vs_churn = (
        df.groupby(["Card_Category", "Attrition_Label"])["Total_Trans_Amt"]
        .sum()
        .reset_index(name="Customer_Transaction_Value")
    )
    category_chart = px.bar(
        category_vs_churn,
        x="Card_Category",
        y="Customer_Transaction_Value",
        color="Attrition_Label",
        barmode="group",
        color_discrete_map={"Active": COLOR_ACTIVE, "Churned": COLOR_CHURN},
        text_auto=".2s",
    )
    base_layout(category_chart, "Category vs Churn")

    dependents_chart_data = (
        df.groupby(["Dependent_count", "Attrition_Label"]).size().reset_index(name="Count")
    )
    dependents_chart = px.bar(
        dependents_chart_data,
        x="Dependent_count",
        y="Count",
        color="Attrition_Label",
        barmode="group",
        color_discrete_map={"Active": COLOR_ACTIVE, "Churned": COLOR_CHURN},
        text_auto=".2s",
    )
    base_layout(dependents_chart, "Dependents vs Churn")

    segmentation_chart_data = (
        df.groupby(["Education_Level", "Gender"]).size().reset_index(name="Count")
    )
    segmentation_chart = px.bar(
        segmentation_chart_data,
        x="Education_Level",
        y="Count",
        color="Gender",
        barmode="stack",
        color_discrete_map={"F": COLOR_ACTIVE, "M": COLOR_CHURN},
        text_auto=".2s",
    )
    base_layout(segmentation_chart, "Customer Segmentation by Gender and Education")

    behavior_scatter = px.scatter(
        df,
        x="Total_Amt_Chng_Q4_Q1",
        y="Total_Trans_Ct",
        color="Attrition_Label",
        color_discrete_map={"Active": COLOR_ACTIVE, "Churned": COLOR_CHURN},
        opacity=0.7,
    )
    base_layout(behavior_scatter, "Customer Transaction Behavior vs Churn")

    inactivity_utilization = px.scatter(
        df,
        x="Months_Inactive_12_mon",
        y="Avg_Utilization_Ratio",
        color="Attrition_Label",
        color_discrete_map={"Active": COLOR_ACTIVE, "Churned": COLOR_CHURN},
        opacity=0.7,
    )
    base_layout(inactivity_utilization, "Inactivity vs Credit Utilization")

    return [
        age_hist,
        gender_donut,
        income_bar,
        churn_donut,
        card_bar,
        gender_bar,
        transaction_scatter,
        inactivity_chart,
        credit_usage_scatter,
        months_book_chart,
        trans_inactive_scatter,
        trans_amount_chart,
        credit_limit_hist,
        credit_vs_trans,
        education_chart,
        avg_credit_income_chart,
        category_chart,
        dependents_chart,
        segmentation_chart,
        behavior_scatter,
        inactivity_utilization,
    ]


def render_charts(figures: list[go.Figure]) -> None:
    st.markdown('<div class="page-heading">Interactive Charts</div>', unsafe_allow_html=True)
    for row_start in range(0, len(figures), 2):
        row = figures[row_start : row_start + 2]
        columns = st.columns(len(row))
        for column, figure in zip(columns, row):
            with column:
                st.markdown('<div class="chart-card">', unsafe_allow_html=True)
                st.plotly_chart(figure, use_container_width=True, config={"displayModeBar": False})
                st.markdown("</div>", unsafe_allow_html=True)


def build_dashboard_snapshot(filtered_df: pd.DataFrame) -> dict:
    active_df = filtered_df[filtered_df["Attrition_Flag"] == 0]
    churned_df = filtered_df[filtered_df["Attrition_Flag"] == 1]

    avg_credit_limit_active = round(float(active_df["Credit_Limit"].mean()), 2) if not active_df.empty else 0
    avg_credit_limit_churned = round(float(churned_df["Credit_Limit"].mean()), 2) if not churned_df.empty else 0
    avg_revolving_bal_active = round(float(active_df["Total_Revolving_Bal"].mean()), 2) if not active_df.empty else 0
    avg_revolving_bal_churned = round(float(churned_df["Total_Revolving_Bal"].mean()), 2) if not churned_df.empty else 0
    avg_util_active = round(float(active_df["Avg_Utilization_Ratio"].mean()), 3) if not active_df.empty else 0
    avg_util_churned = round(float(churned_df["Avg_Utilization_Ratio"].mean()), 3) if not churned_df.empty else 0

    return {
        "income_category_counts": filtered_df["Income_Category"].value_counts().to_dict(),
        "gender_counts": filtered_df["Gender"].value_counts().to_dict(),
        "card_category_counts": filtered_df["Card_Category"].value_counts().to_dict(),
        "education_level_counts": filtered_df["Education_Level"].value_counts().to_dict(),
        "avg_credit_limit_by_income": (
            filtered_df.groupby("Income_Category")["Credit_Limit"].mean().round(2).to_dict()
        ),
        "churn_rate_by_card_category": (
            filtered_df.groupby("Card_Category")["Attrition_Flag"].mean().mul(100).round(2).to_dict()
        ),
        "chart_summaries": {
            "credit_usage_vs_limit": {
                "x_axis": "Credit_Limit",
                "y_axis": "Total_Revolving_Bal",
                "summary": (
                    "This scatter plot compares credit limit with revolving balance to show how much of the available credit customers are actively using."
                ),
                "active_customer_pattern": (
                    f"Active customers average a credit limit of {avg_credit_limit_active:,.0f} and a revolving balance of {avg_revolving_bal_active:,.0f}."
                ),
                "churned_customer_pattern": (
                    f"Churned customers average a credit limit of {avg_credit_limit_churned:,.0f} and a revolving balance of {avg_revolving_bal_churned:,.0f}."
                ),
                "business_takeaway": (
                    f"Utilization is lower for churned customers ({avg_util_churned:.3f}) than for retained customers ({avg_util_active:.3f}), which suggests weaker credit engagement is associated with churn risk."
                ),
            },
            "transaction_activity_vs_churn": {
                "summary": "This plot compares total transaction count and total transaction amount to show how customer activity separates active and churned behavior.",
                "business_takeaway": "Higher transaction activity generally aligns with retained customers, while lower transaction volume and weaker engagement are more common among churned customers.",
            },
            "transaction_count_vs_inactive_months": {
                "summary": "This chart compares total transaction count against inactive months to show how customer activity changes as inactivity rises.",
                "business_takeaway": "Customers with more inactive months generally show weaker transaction activity, which supports the broader churn pattern that lower engagement and inactivity move together.",
            },
            "inactivity_vs_credit_utilization": {
                "summary": "This chart shows how credit utilization changes across inactivity levels.",
                "business_takeaway": "Customers with higher inactivity and lower utilization are generally more likely to appear in churn-risk patterns.",
            },
        },
    }


def initialize_chat_state() -> None:
    if "assistant_messages" not in st.session_state:
        st.session_state.assistant_messages = [
            {
                "role": "assistant",
                "content": (
                    "Ask about churn drivers, risky customer groups, or what the filtered charts suggest."
                ),
            }
        ]
    if "assistant_example_question" not in st.session_state:
        st.session_state.assistant_example_question = ""
    if "assistant_manual_question" not in st.session_state:
        st.session_state.assistant_manual_question = ""


def render_ai_assistant(
    filtered_df: pd.DataFrame, insights: dict, behavior: dict, filter_context: dict
) -> None:
    top_drivers = insights.get("top_churn_drivers", [])
    context_payload = build_dashboard_context_payload(
        dataset_size=len(filtered_df),
        churn_rate=float(filtered_df["Attrition_Flag"].mean() * 100),
        churned_customers=int(filtered_df["Attrition_Flag"].sum()),
        best_model=insights.get("best_model", "Random Forest"),
        top_drivers=top_drivers,
        dashboard_pages=DASHBOARD_PAGES,
    )
    context_payload.update(behavior)
    context_payload["current_filters"] = filter_context
    context_payload["dashboard_snapshot"] = build_dashboard_snapshot(filtered_df)
    context_payload["power_bi_report_url"] = os.getenv(
        "POWER_BI_REPORT_URL", DEFAULT_POWER_BI_REPORT_URL
    )

    st.markdown('<div id="ask-ai-anchor"></div>', unsafe_allow_html=True)
    popover_label = "✨ Ask AI"
    popover_context = (
        st.popover(popover_label, use_container_width=False)
        if hasattr(st, "popover")
        else st.expander(popover_label, expanded=False)
    )

    with popover_context:
        st.markdown('<div class="chat-panel">', unsafe_allow_html=True)
        st.markdown(
            '<div class="assistant-copy">The assistant uses the filtered dashboard context plus the ML churn drivers.</div>',
            unsafe_allow_html=True,
        )

        api_key = get_gemini_api_key()
        model_name = get_gemini_model()
        gemini_available = is_gemini_available()
        gemini_debug_status = get_gemini_debug_status()
        if not api_key:
            st.info(
                "Add `GEMINI_API_KEY` in Streamlit **App Settings -> Secrets** or in your local `.env` "
                "file to enable live Gemini responses."
            )
        elif not gemini_available:
            st.info(
                "Gemini is not installed locally, so the assistant will use a dashboard-based fallback explanation."
            )
        else:
            st.caption(f"Gemini assistant ready. Primary model candidate: `{model_name}`.")

        with st.expander("Gemini connection diagnostics", expanded=not api_key):
            st.caption(
                "This check only shows whether the key source is detected. It does not reveal the actual secret value."
            )
            st.write(
                {
                    "streamlit_secret_found": "Yes" if gemini_debug_status["streamlit_secret_found"] else "No",
                    "environment_variable_found": "Yes"
                    if gemini_debug_status["environment_variable_found"]
                    else "No",
                    "gemini_package_available": "Yes"
                    if gemini_debug_status["gemini_package_available"]
                    else "No",
                    "selected_model": model_name,
                }
            )

        example_queries = [
            "Why are customers churning in the current filtered view?",
            "Which segments in these charts look most at risk?",
            "What actions can reduce churn for this filtered segment?",
        ]
        st.markdown('<div class="assistant-subheading">Quick questions</div>', unsafe_allow_html=True)
        example_columns = st.columns(len(example_queries))
        for column, example_query in zip(example_columns, example_queries):
            with column:
                if st.button(example_query, key=f"assistant-example-{example_query}"):
                    st.session_state.assistant_example_question = example_query

        st.markdown('<div class="assistant-subheading">Conversation</div>', unsafe_allow_html=True)
        st.markdown('<div class="chat-history">', unsafe_allow_html=True)
        for message in st.session_state.assistant_messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        st.markdown("</div>", unsafe_allow_html=True)

        with st.form("assistant-question-form", clear_on_submit=True):
            manual_question = st.text_input(
                "Ask a question about the dashboard",
                key="assistant_manual_question",
                placeholder="Ask a question about the dashboard",
                label_visibility="collapsed",
            )
            submitted = st.form_submit_button("Send", use_container_width=True)

        if submitted:
            question = manual_question.strip()
        else:
            question = st.session_state.assistant_example_question

        if question:
            st.session_state.assistant_example_question = ""
            st.session_state.assistant_messages.append({"role": "user", "content": question})
            if not api_key:
                answer = generate_fallback_response(question, context_payload)
            else:
                answer = generate_llm_response(
                    question=question,
                    chat_history=st.session_state.assistant_messages[:-1],
                    context_payload=context_payload,
                    api_key=api_key,
                    model=model_name,
                )
            st.session_state.assistant_messages.append(
                {"role": "assistant", "content": answer}
            )
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: #0f131a;
            color: #f3f6fb;
        }
        div[data-testid="stPopover"] {
            position: fixed !important;
            right: 24px !important;
            bottom: 24px !important;
            left: auto !important;
            top: auto !important;
            z-index: 1000 !important;
            width: auto !important;
        }
        div[data-testid="stPopover"] > button {
            border-radius: 999px;
            background: #ff5563;
            color: white;
            border: 0;
            box-shadow: 0 12px 26px rgba(255, 85, 99, 0.30);
            padding: 0.65rem 0.95rem;
            min-width: 110px;
        }
        div[data-testid="stPopoverContent"] {
            position: fixed !important;
            right: 24px !important;
            left: auto !important;
            bottom: 84px !important;
            top: auto !important;
            width: 430px !important;
            max-width: calc(100vw - 40px) !important;
            max-height: 78vh !important;
            overflow: auto !important;
            border-radius: 22px !important;
        }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #131924 0%, #0d1219 100%);
            border-right: 1px solid #273041;
        }
        section[data-testid="stSidebar"] * {
            color: #f3f6fb !important;
        }
        h1, h2, h3, p, label, .stMarkdown, .stCaption {
            color: #f3f6fb !important;
        }
        .sidebar-title,
        .page-heading,
        .section-heading,
        .assistant-subheading {
            color: #f7f9fc !important;
            font-weight: 800;
        }
        .sidebar-title {
            font-size: 1.4rem;
            margin-bottom: 0.85rem;
        }
        .page-heading {
            font-size: 1.8rem;
            margin: 1.4rem 0 0.85rem 0;
        }
        .section-heading,
        .assistant-subheading {
            font-size: 1.35rem;
            margin-bottom: 0.7rem;
        }
        .title-card,
        .metric-card,
        .section-card,
        .chart-card,
        .chat-panel {
            background: #171b23;
            border: 1px solid #273041;
            border-radius: 22px;
            box-shadow: 0 16px 30px rgba(0, 0, 0, 0.24);
            padding: 1rem 1.1rem;
        }
        .title-card {
            min-height: 150px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .title-card-text {
            text-align: center;
            color: #d4dae4;
            font-size: 1.65rem;
            font-weight: 800;
            letter-spacing: 0.04em;
            line-height: 1.22;
        }
        .metric-card {
            min-height: 150px;
        }
        .metric-label {
            color: #d7dde7;
            font-size: 1rem;
            font-weight: 700;
            margin-bottom: 1rem;
        }
        .metric-value {
            background: #10161d;
            border: 1px solid #2a3446;
            border-radius: 12px;
            color: #ff5f6b;
            font-size: 2.2rem;
            font-weight: 800;
            text-align: center;
            padding: 1rem 0.6rem;
        }
        .hero-note {
            color: #c8d0db;
            font-size: 0.95rem;
            margin: 0.8rem 0 1.3rem 0.15rem;
        }
        .model-card {
            min-height: 180px;
        }
        .model-card-title {
            color: #f7f9fc;
            font-weight: 800;
            font-size: 1.05rem;
            margin-bottom: 0.75rem;
        }
        .model-card-content {
            color: #d0d7e1;
            line-height: 1.6;
        }
        .chart-card {
            margin-bottom: 1rem;
            padding: 0.35rem 0.5rem 0.2rem 0.5rem;
        }
        .chat-panel {
            margin-top: 0.5rem;
        }
        .chat-history {
            max-height: 340px;
            overflow-y: auto;
            padding-right: 0.2rem;
            margin-bottom: 0.9rem;
        }
        .assistant-copy {
            color: #d0d7e1 !important;
            margin-bottom: 0.7rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_styles()
full_df = load_data()
insights = load_insights()
initialize_chat_state()
filtered_df, filter_context = apply_filters(full_df)
embed_url, report_url = get_power_bi_links()

if filtered_df.empty:
    st.warning("No records match the current filters. Please widen the filter selections.")
    st.stop()

render_header(filtered_df, full_df, insights)
render_dataset_summary(filtered_df)
st.markdown('<div class="page-heading">Model Insights</div>', unsafe_allow_html=True)
behavior_summary = render_model_insights(filtered_df, insights)
figures = create_visualizations(filtered_df)
render_charts(figures)

st.markdown('<div class="page-heading">Power BI Report Link</div>', unsafe_allow_html=True)
if embed_url:
    st.markdown(f"[Open embedded Power BI report]({embed_url})")
else:
    st.markdown(f"[Open Power BI report]({report_url})")

render_ai_assistant(filtered_df, insights, behavior_summary, filter_context)
