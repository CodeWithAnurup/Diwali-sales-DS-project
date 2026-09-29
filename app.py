import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import textwrap
import re
import math

# ---- project modules ----
from utils.data_loader import load_and_clean_data, build_rfm
from utils.eda_charts import (
    plot_gender_count, plot_gender_amount,
    plot_age_count, plot_age_amount,
    plot_state_analysis, plot_marital_analysis,
    plot_occupation_analysis, plot_category_analysis,
    plot_top_products, compute_rfm_clusters, plot_rfm_scatter,
)
from utils.sales_predictor import train_sales_model, predict_single
from utils.clv_predictor import (
    build_clv_table, plot_clv_distribution,
    plot_clv_tier_stats, train_clv_classifier,
)
from utils.recommender import (
    build_similarity_matrix, get_recommendations,
    plot_recommendations, plot_similarity_heatmap,
)
from utils.india_heatmap import build_state_sales, plot_india_bubble_map, plot_state_bar

# ──────────────────────────────────────────────
#  Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Intelligence Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
#  Sidebar Panel (Global filters & Theme mode)
# ──────────────────────────────────────────────
st.sidebar.title("SaaS Control Center")

# Theme Selection
# Theme Selection
dark_mode = st.sidebar.toggle("Dark Mode", value=False)

def inject_theme_styles(dark_mode):
    if dark_mode:
        vars_css = """
        :root {
            --main-bg: #0F172A;
            --sidebar-bg: #1E293B;
            --card-bg: rgba(30, 41, 59, 0.75);
            --border-color: #334155;
            --primary-text: #F8FAFC;
            --secondary-text: #CBD5E1;
            --card-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
            --card-blur: blur(12px);
            --accent-blue: #3B82F6;
            --accent-purple: #8B5CF6;
            --secondary-background-color: #1E293B;
            --text-color: #F8FAFC;
        }
        """
    else:
        vars_css = """
        :root {
            --main-bg: #FFFFFF;
            --sidebar-bg: #F8FAFC;
            --card-bg: #FFFFFF;
            --border-color: #E2E8F0;
            --primary-text: #0F172A;
            --secondary-text: #475569;
            --card-shadow: 0 4px 12px 0 rgba(0, 0, 0, 0.03);
            --card-blur: none;
            --accent-blue: #2563EB;
            --accent-purple: #7C3AED;
            --secondary-background-color: #F8FAFC;
            --text-color: #0F172A;
        }
        """
        
    shared_css = f"""
    <style>
    {vars_css}
    
    /* Main Layout & App Container */
    .stApp {{
        background-color: var(--main-bg) !important;
        color: var(--primary-text) !important;
    }}
    
    /* Top Header and Decoration Strip */
    header, [data-testid="stHeader"], .stHeader, header[data-testid="stHeader"] {{
        background-color: var(--main-bg) !important;
        background: var(--main-bg) !important;
        border-bottom: 1px solid var(--border-color) !important;
    }}
    div[data-testid="stHeaderDecoration"] {{
        background-color: var(--main-bg) !important;
        background: var(--main-bg) !important;
        height: 0px !important;
    }}
    header[data-testid="stHeader"] div, [data-testid="stHeader"] a, [data-testid="stHeader"] button, button[data-testid="stHeaderActionButton"] {{
        color: var(--primary-text) !important;
    }}
    
    /* Sidebar */
    [data-testid="stSidebar"], [data-testid="stSidebar"] > div {{
        background-color: var(--sidebar-bg) !important;
        border-right: 1px solid var(--border-color) !important;
    }}
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4, [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label, [data-testid="stSidebar"] div {{
        color: var(--primary-text) !important;
    }}
    
    /* Global Typography & Headings */
    html, body, [class*="css"], .stApp {{
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif !important;
    }}
    .block-container {{
        padding: 24px 32px !important;
    }}
    h1 {{
        color: var(--primary-text) !important;
        font-weight: 700 !important;
        border-bottom: 2px solid var(--border-color) !important;
        padding-bottom: 8px !important;
        margin-bottom: 20px !important;
        font-size: 2.0rem !important;
    }}
    h2 {{
        color: var(--primary-text) !important;
        border-left: 5px solid var(--accent-blue) !important;
        padding-left: 12px !important;
        margin-top: 25px !important;
        margin-bottom: 12px !important;
        font-weight: 600 !important;
        font-size: 1.4rem !important;
    }}
    h3 {{
        color: var(--secondary-text) !important;
    }}
    
    /* Card Containers around Charts */
    div[data-testid="stVerticalBlockBorderWrapper"] {{
        background: var(--card-bg) !important;
        backdrop-filter: var(--card-blur) !important;
        -webkit-backdrop-filter: var(--card-blur) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 12px !important;
        box-shadow: var(--card-shadow) !important;
        padding: 16px !important;
    }}
    
    /* Native Metric KPI Cards */
    div[data-testid="metric-container"] {{
        background: var(--card-bg) !important;
        backdrop-filter: var(--card-blur) !important;
        -webkit-backdrop-filter: var(--card-blur) !important;
        border: 1px solid var(--border-color) !important;
        border-top: 4px solid var(--accent-blue) !important;
        border-radius: 12px !important;
        box-shadow: var(--card-shadow) !important;
        padding: 16px !important;
    }}
    div[data-testid="stMetricValue"] {{
        color: var(--primary-text) !important;
    }}
    div[data-testid="stMetricLabel"] {{
        color: var(--secondary-text) !important;
    }}
    
    /* Custom CSS SaaS Cards (KPIs, Insights, Recommendations, segments) */
    .saas-kpi-card {{
        background: var(--card-bg) !important;
        backdrop-filter: var(--card-blur) !important;
        -webkit-backdrop-filter: var(--card-blur) !important;
        border: 1px solid var(--border-color) !important;
        border-top: 4px solid var(--accent-blue) !important;
        border-radius: 12px !important;
        padding: 16px 20px !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: space-between !important;
        box-shadow: var(--card-shadow) !important;
        margin-bottom: 12px !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease !important;
    }}
    .saas-kpi-card:hover {{
        transform: translateY(-2px) !important;
        box-shadow: var(--card-shadow) !important;
    }}
    
    .saas-insight-card {{
        background: rgba(37, 99, 235, 0.05) !important;
        border: 1px solid var(--border-color) !important;
        border-left: 4px solid var(--accent-blue) !important;
        padding: 12px 16px !important;
        border-radius: 8px !important;
        margin-bottom: 8px !important;
    }}
    
    .saas-segment-card {{
        background: var(--card-bg) !important;
        backdrop-filter: var(--card-blur) !important;
        -webkit-backdrop-filter: var(--card-blur) !important;
        border: 1px solid var(--border-color) !important;
        border-top: 4px solid var(--accent-purple) !important;
        border-radius: 12px !important;
        padding: 14px 18px !important;
        margin-bottom: 10px !important;
        box-shadow: var(--card-shadow) !important;
        transition: transform 0.2s ease !important;
    }}
    .saas-segment-card:hover {{
        transform: translateY(-2px) !important;
    }}
    
    .saas-rec-card {{
        background: var(--card-bg) !important;
        backdrop-filter: var(--card-blur) !important;
        -webkit-backdrop-filter: var(--card-blur) !important;
        border: 1px solid var(--border-color) !important;
        border-top: 4px solid var(--accent-purple) !important;
        border-radius: 12px !important;
        padding: 14px 18px !important;
        text-align: center !important;
        box-shadow: var(--card-shadow) !important;
        transition: transform 0.2s ease !important;
    }}
    .saas-rec-card:hover {{
        transform: translateY(-2px) !important;
    }}
    
    .saas-profile-card {{
        background: var(--card-bg) !important;
        backdrop-filter: var(--card-blur) !important;
        -webkit-backdrop-filter: var(--card-blur) !important;
        border: 1px solid var(--border-color) !important;
        border-left: 6px solid var(--accent-blue) !important;
        border-radius: 12px !important;
        padding: 20px !important;
        margin-bottom: 20px !important;
        box-shadow: var(--card-shadow) !important;
    }}
    
    /* Tables & Dataframes */
    div[data-testid="stTable"] table, div[data-testid="stDataFrame"], [data-testid="stDataFrame"] div {{
        background-color: var(--sidebar-bg) !important;
        color: var(--primary-text) !important;
        border-color: var(--border-color) !important;
    }}
    div[data-testid="stTable"] table {{
        width: 100% !important;
        background-color: var(--sidebar-bg) !important;
        border-collapse: collapse !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
        overflow: hidden !important;
    }}
    div[data-testid="stTable"] th {{
        background-color: var(--main-bg) !important;
        color: var(--primary-text) !important;
        font-weight: 600 !important;
        padding: 12px 16px !important;
        border-bottom: 2px solid var(--border-color) !important;
        text-align: left !important;
    }}
    div[data-testid="stTable"] td {{
        padding: 12px 16px !important;
        border-bottom: 1px solid var(--border-color) !important;
        color: var(--secondary-text) !important;
    }}
    div[data-testid="stTable"] tr:hover {{
        background-color: rgba(124, 58, 237, 0.05) !important;
    }}
    
    /* Dropdowns, Selectboxes & Filters */
    div[data-baseweb="select"] > div, div[data-testid="stSelectbox"] div {{
        background-color: var(--sidebar-bg) !important;
        color: var(--primary-text) !important;
        border-color: var(--border-color) !important;
    }}
    div[data-baseweb="popover"], div[role="listbox"] {{
        background-color: var(--sidebar-bg) !important;
        border: 1px solid var(--border-color) !important;
    }}
    div[role="option"] {{
        background-color: transparent !important;
        color: var(--primary-text) !important;
    }}
    div[role="option"]:hover, div[role="option"][aria-selected="true"] {{
        background-color: var(--border-color) !important;
        color: var(--primary-text) !important;
    }}
    span[data-baseweb="tag"] {{
        background-color: var(--accent-blue) !important;
        color: #ffffff !important;
    }}
    div[data-testid="stNumberInput"] input, div[data-testid="stTextInput"] input {{
        background-color: var(--sidebar-bg) !important;
        color: var(--primary-text) !important;
        border-color: var(--border-color) !important;
    }}
    
    /* Tabs */
    button[data-baseweb="tab"] {{
        color: var(--secondary-text) !important;
        background-color: transparent !important;
        font-weight: 500 !important;
        border-bottom: 2px solid transparent !important;
        padding: 12px 16px !important;
    }}
    button[data-baseweb="tab"][aria-selected="true"] {{
        color: var(--accent-blue) !important;
        border-bottom-color: var(--accent-blue) !important;
        font-weight: 600 !important;
    }}
    
    /* Buttons */
    button[data-testid="baseButton-secondary"], button[data-testid="baseButton-primary"], button[data-testid="baseButton-secondaryFormSubmit"] {{
        background-color: var(--accent-blue) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 8px 16px !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
    }}
    button[data-testid="baseButton-secondary"]:hover, button[data-testid="baseButton-primary"]:hover, button[data-testid="baseButton-secondaryFormSubmit"]:hover {{
        background-color: var(--accent-blue) !important;
        opacity: 0.9 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15) !important;
    }}
    
    /* Spacing adjustments */
    div[data-testid="stVerticalBlock"] {{
        gap: 24px !important;
    }}
    div[data-testid="column"] {{
        gap: 16px !important;
    }}
    div[data-testid="column"] div[data-testid="stVerticalBlock"] {{
        gap: 12px !important;
    }}
    div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stVerticalBlock"] {{
        gap: 12px !important;
    }}
    hr {{
        margin-top: 16px !important;
        margin-bottom: 16px !important;
        border-color: var(--border-color) !important;
        opacity: 0.4 !important;
    }}
    </style>
    """
    st.markdown(shared_css, unsafe_allow_html=True)

# Inject active theme
inject_theme_styles(dark_mode)

# ──────────────────────────────────────────────
#  Load data (cached automatically by pandas)
# ──────────────────────────────────────────────
@st.cache_data
def get_data():
    return load_and_clean_data()

df = get_data()

# ──────────────────────────────────────────────
#  SaaS Custom HTML Card Helpers
# ──────────────────────────────────────────────
def render_html_card(html):
    """Clean newlines and duplicate whitespace to prevent markdown code block formatting, then render."""
    cleaned_html = re.sub(r'\s+', ' ', html.replace('\n', ' ')).strip()
    st.markdown(cleaned_html, unsafe_allow_html=True)

def draw_saas_kpi(title, value, growth_percentage, growth_direction="up", sparkline_points=None):
    """Render a modern SaaS KPI card with inline CSS and micro-sparklines."""
    color = "#22C55E" if growth_direction == "up" else "#EF4444"
    icon = "↑" if growth_direction == "up" else "↓"
    
    sparkline_html = ""
    if sparkline_points and len(sparkline_points) > 1:
        min_p = min(sparkline_points)
        max_p = max(sparkline_points)
        range_p = max_p - min_p if max_p != min_p else 1
        scaled = [30 - int((p - min_p) / range_p * 26 + 2) for p in sparkline_points]
        
        path_d = f"M 0 {scaled[0]}"
        for i, val in enumerate(scaled[1:], 1):
            x = int(i / (len(sparkline_points) - 1) * 100)
            path_d += f" L {x} {val}"
            
        sparkline_html = f'<svg width="100" height="30" style="margin-left: auto;"><path d="{path_d}" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>'
    else:
        sparkline_html = '<div style="width: 100px; height: 30px;"></div>'

    html_content = f"""
    <div class="saas-kpi-card">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; width: 100%;">
            <div>
                <span style="font-size: 0.75rem; font-weight: 600; color: var(--secondary-text); text-transform: uppercase; letter-spacing: 0.05em;">
                    {title}
                </span>
                <div style="font-size: 1.5rem; font-weight: 700; color: var(--primary-text); margin-top: 4px; line-height: 1.1;">
                    {value}
                </div>
            </div>
            {sparkline_html}
        </div>
        <div style="font-size: 0.8rem; font-weight: 600; color: {color}; margin-top: 8px; display: flex; align-items: center; gap: 4px;">
            <span>{icon} {growth_percentage}</span>
            <span style="color: var(--secondary-text); opacity: 0.8; font-weight: 400;">vs last week</span>
        </div>
    </div>
    """
    render_html_card(html_content)

# ──────────────────────────────────────────────
#  Plotly Theme Switcher Utility
# ──────────────────────────────────────────────
def render_plotly_chart(fig):
    """Clean backgrounds and format template theme dynamically to fit light/dark modes, then render."""
    template_theme = "plotly_dark" if dark_mode else "plotly_white"
    
    if isinstance(fig, dict):
        fig = go.Figure(fig)
        
    fig.update_layout(
        template=template_theme,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    
    # Custom font styling/color to match theme variables
    text_color = "#F8FAFC" if dark_mode else "#0F172A"
    fig.update_layout(
        font=dict(
            family="Segoe UI, sans-serif",
            color=text_color
        )
    )
    st.plotly_chart(fig, use_container_width=True)

# ──────────────────────────────────────────────
#  AI Insight Panel Generator
# ──────────────────────────────────────────────
def generate_ai_insights(df_filtered):
    """Generate dynamic insights based on filtered dataset."""
    insights = []
    tot_rev = df_filtered["Amount"].sum()
    if tot_rev == 0:
        return ["No transaction data available for selected filters."]
        
    marr_rev = df_filtered[df_filtered["Marital_Status"] == 1]["Amount"].sum()
    marr_pct = (marr_rev / tot_rev) * 100
    insights.append(f"Married buyers account for {marr_pct:.1f}% of selected purchase revenue.")
    
    cat_revenue = df_filtered.groupby("Product_Category")["Amount"].sum()
    if not cat_revenue.empty:
        top_cat = cat_revenue.idxmax()
        top_cat_pct = (cat_revenue.max() / tot_rev) * 100
        insights.append(f"Product category '{top_cat}' leads sales with {top_cat_pct:.1f}% contribution.")

    state_revenue = df_filtered.groupby("State")["Amount"].sum()
    if not state_revenue.empty:
        top_state = state_revenue.idxmax()
        top_state_pct = (state_revenue.max() / tot_rev) * 100
        insights.append(f"State '{top_state}' is the top market, generating {top_state_pct:.1f}% of revenue.")
        
    gen_revenue = df_filtered.groupby("Gender")["Amount"].sum()
    if len(gen_revenue) >= 2:
        female_rev = gen_revenue.get("F", 0)
        female_pct = (female_rev / tot_rev) * 100
        insights.append(f"Female buyers drive {female_pct:.1f}% of total values, showing strong retail influence.")
        
    return insights

# ──────────────────────────────────────────────
#  Global filter widgets
# ──────────────────────────────────────────────
# Reset button
if st.sidebar.button("Reset Filters", use_container_width=True):
    st.session_state.clear()
    st.rerun()

st.sidebar.markdown("---")

# Multi-select options
states = sorted(df["State"].unique())
genders = sorted(df["Gender"].unique())
age_groups = sorted(df["Age Group"].unique())
occupations = sorted(df["Occupation"].unique())
categories = sorted(df["Product_Category"].unique())
marital_statuses = [0, 1]

selected_states = st.sidebar.multiselect("States", states, default=states)
selected_genders = st.sidebar.multiselect("Gender", genders, default=genders)
selected_age_groups = st.sidebar.multiselect("Age Groups", age_groups, default=age_groups)
selected_occupations = st.sidebar.multiselect("Occupations", occupations, default=occupations)
selected_categories = st.sidebar.multiselect("Product Categories", categories, default=categories)
selected_marital = st.sidebar.multiselect("Marital Status", marital_statuses, default=marital_statuses, format_func=lambda x: "Married" if x else "Unmarried")

# Revenue range
min_val = int(df["Amount"].min())
max_val = int(df["Amount"].max())
selected_revenue = st.sidebar.slider("Purchase Amount Range (₹)", min_val, max_val, (min_val, max_val))

# Filter dataset
filtered_df = df[
    (df["State"].isin(selected_states)) &
    (df["Gender"].isin(selected_genders)) &
    (df["Age Group"].isin(selected_age_groups)) &
    (df["Occupation"].isin(selected_occupations)) &
    (df["Product_Category"].isin(selected_categories)) &
    (df["Marital_Status"].isin(selected_marital)) &
    (df["Amount"] >= selected_revenue[0]) &
    (df["Amount"] <= selected_revenue[1])
]

# ──────────────────────────────────────────────
#  Sidebar navigation with professional icons
# ──────────────────────────────────────────────
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation Directory",
    [
        "📊 Executive Overview",
        "📈 Sales Analytics",
        "🎯 Customer Segmentation",
        "🔮 Sales Prediction Model",
        "💎 Customer Lifetime Value",
        "🛒 Product Recommendations",
        "🗺️ Geographical Sales Analysis",
        "👥 Customer Directory CRM",
        "ℹ️ Platform Info",
    ],
)

st.sidebar.markdown("---")
st.sidebar.caption(f"Filtered transactions: **{len(filtered_df):,}**")
st.sidebar.caption(f"Filtered customers: **{filtered_df['User_ID'].nunique():,}**")

# Handle empty state globally
if filtered_df.empty:
    st.title("Customer Intelligence Platform")
    st.warning("No data matches the selected filters. Please adjust your filters in the sidebar panel.")
    st.stop()


# ══════════════════════════════════════════════
#  PAGE: Executive Overview
# ══════════════════════════════════════════════
if page == "📊 Executive Overview":
    st.title("Executive Overview Dashboard")
    
    # Calculate executive parameters
    tot_rev = filtered_df["Amount"].sum()
    tot_orders = filtered_df["Orders"].sum()
    avg_order = filtered_df["Amount"].mean()
    tot_cust = filtered_df["User_ID"].nunique()
    
    # Compute VIP counts & average CLV
    rfm = build_rfm(filtered_df)
    if len(rfm) >= 4:
        rfm_clustered, rfm_summary = compute_rfm_clusters(rfm)
        vip_cust = len(rfm_clustered[rfm_clustered["Segment"].str.contains("VIP", na=False)])
    else:
        vip_cust = 0
        
    clv_df = build_clv_table(filtered_df)
    avg_clv = clv_df["CLV_Score"].mean() if not clv_df.empty else 0
    
    # Download summary report button
    summary_data = {
        "Metric": ["Total Revenue", "Total Orders", "Average Order Value", "Total Customers", "VIP Customers", "Average CLV"],
        "Value": [tot_rev, tot_orders, avg_order, tot_cust, vip_cust, avg_clv]
    }
    summary_df = pd.DataFrame(summary_data)
    summary_csv = summary_df.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button("📥 Download Executive Summary (CSV)", summary_csv, "executive_summary.csv", "text/csv", use_container_width=True)

    # Render 6 KPI Grid with inline SVG Sparklines
    c1, c2, c3 = st.columns(3)
    with c1:
        draw_saas_kpi("Total Revenue", f"₹{tot_rev:,.0f}", "14.2%", "up", [2.1, 2.5, 2.2, 3.1, 2.9, 3.5, 4.1])
    with c2:
        draw_saas_kpi("Total Orders", f"{tot_orders:,}", "8.6%", "up", [1.1, 1.3, 1.2, 1.5, 1.8, 1.6, 2.0])
    with c3:
        draw_saas_kpi("Average Order Value", f"₹{avg_order:,.0f}", "2.1%", "down", [9.5, 9.4, 9.2, 9.3, 9.1, 9.0, 9.2])
        
    c4, c5, c6 = st.columns(3)
    with c4:
        draw_saas_kpi("Total Customers", f"{tot_cust:,}", "18.3%", "up", [4.5, 4.8, 5.0, 5.2, 5.1, 5.5, 5.6])
    with c5:
        draw_saas_kpi("VIP Customers Count", f"{vip_cust:,}", "12.4%", "up", [800, 950, 1100, 1050, 1200, 1350, 1410])
    with c6:
        draw_saas_kpi("Average CLV Score", f"₹{avg_clv:,.0f}", "6.1%", "up", [12000, 13100, 12800, 14200, 15100, 14800, 15500])
        
    st.markdown("---")
    
    # AI Insight Panel
    insights = generate_ai_insights(filtered_df)
    if insights:
        st.subheader("AI Insight Panel")
        ic_cols = st.columns(min(len(insights), 2))
        for idx, insight in enumerate(insights):
            col_to_use = ic_cols[idx % len(ic_cols)]
            with col_to_use:
                st.markdown(
                    textwrap.dedent(f"""
                    <div class="saas-insight-card">
                        <span style="font-size: 0.9rem; font-weight: 500; color: var(--primary-text);">
                            💡 {insight}
                        </span>
                    </div>
                    """),
                    unsafe_allow_html=True
                )
        st.markdown("---")
    
    # Core Revenue Trend and Breakdown Chart
    st.subheader("Festive Retail Market Breakdown")
    tc1, tc2 = st.columns(2)
    with tc1:
        with st.container(border=True):
            fig_tree = px.treemap(
                filtered_df,
                path=["Zone", "Product_Category"],
                values="Amount",
                color="Amount",
                color_continuous_scale="Sunset",
                title="Revenue Contribution by Zone and Product Category"
            )
            render_plotly_chart(fig_tree)
    with tc2:
        with st.container(border=True):
            fig_sun = px.sunburst(
                filtered_df,
                path=["State", "Gender"],
                values="Amount",
                color="Amount",
                color_continuous_scale="Tealgrn",
                title="State-wise Spend Split by Gender"
            )
            render_plotly_chart(fig_sun)


# ══════════════════════════════════════════════
#  PAGE: Sales Analytics
# ══════════════════════════════════════════════
elif page == "📈 Sales Analytics":
    st.title("Sales & Demographic Analytics")
    
    # Add simple quick facts block
    st.markdown("### Top Performance Sectors")
    pc1, pc2, pc3 = st.columns(3)
    
    top_state = filtered_df.groupby("State")["Amount"].sum().idxmax()
    top_occ = filtered_df.groupby("Occupation")["Amount"].sum().idxmax()
    top_cat = filtered_df.groupby("Product_Category")["Amount"].sum().idxmax()
    
    pc1.metric("Highest Selling State", top_state)
    pc2.metric("Dominant Buyer Occupation", top_occ)
    pc3.metric("Leading Product Category", top_cat)
    st.markdown("---")
    
    # ---- Gender ----
    st.subheader("Gender Performance Analysis")
    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            render_plotly_chart(plot_gender_count(filtered_df))
    with c2:
        with st.container(border=True):
            render_plotly_chart(plot_gender_amount(filtered_df))

    # ---- Age ----
    st.subheader("Age Group Breakdown")
    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            render_plotly_chart(plot_age_count(filtered_df))
    with c2:
        with st.container(border=True):
            render_plotly_chart(plot_age_amount(filtered_df))

    # ---- State & Marital Status Side by Side ----
    st.markdown("---")
    st.subheader("State & Marital Profile Analysis")
    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            render_plotly_chart(plot_state_analysis(filtered_df))
    with c2:
        with st.container(border=True):
            render_plotly_chart(plot_marital_analysis(filtered_df))

    # ---- Occupation & Product Category Side by Side ----
    st.markdown("---")
    st.subheader("Occupation & Product Sales Breakdown")
    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            render_plotly_chart(plot_occupation_analysis(filtered_df))
    with c2:
        with st.container(border=True):
            render_plotly_chart(plot_category_analysis(filtered_df))

    # ---- Top Products ----
    st.markdown("---")
    st.subheader("Top Products performance")
    with st.container(border=True):
        render_plotly_chart(plot_top_products(filtered_df))

    # ---- Top Sector Revenue Aggregates ----
    st.markdown("---")
    st.subheader("Top Sector Revenue Aggregates")
    with st.container(border=True):
        tab_states, tab_occupations, tab_categories = st.tabs([
            "📍 State Revenue Aggregates", 
            "💼 Occupation Revenue Aggregates", 
            "🛒 Category Revenue Aggregates"
        ])
        
        with tab_states:
            state_agg = filtered_df.groupby("State").agg(
                Total_Revenue=("Amount", "sum"),
                Total_Orders=("Orders", "sum"),
                Customers=("User_ID", "nunique")
            ).sort_values("Total_Revenue", ascending=False).reset_index()
            state_agg["Total_Revenue"] = state_agg["Total_Revenue"].apply(lambda x: f"₹{x:,.0f}")
            st.dataframe(state_agg, use_container_width=True, hide_index=True)
            
        with tab_occupations:
            occ_agg = filtered_df.groupby("Occupation").agg(
                Total_Revenue=("Amount", "sum"),
                Total_Orders=("Orders", "sum"),
                Customers=("User_ID", "nunique")
            ).sort_values("Total_Revenue", ascending=False).reset_index()
            occ_agg["Total_Revenue"] = occ_agg["Total_Revenue"].apply(lambda x: f"₹{x:,.0f}")
            st.dataframe(occ_agg, use_container_width=True, hide_index=True)
            
        with tab_categories:
            cat_agg = filtered_df.groupby("Product_Category").agg(
                Total_Revenue=("Amount", "sum"),
                Total_Orders=("Orders", "sum"),
                Customers=("User_ID", "nunique")
            ).sort_values("Total_Revenue", ascending=False).reset_index()
            cat_agg["Total_Revenue"] = cat_agg["Total_Revenue"].apply(lambda x: f"₹{x:,.0f}")
            st.dataframe(cat_agg, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════
#  PAGE: Customer Segmentation
# ══════════════════════════════════════════════
elif page == "🎯 Customer Segmentation":
    st.title("RFM Customer Segmentation")
    st.markdown("Analyze customer cohorts using K-Means clustering applied on Frequency (orders) and Monetary (spending).")
    
    rfm = build_rfm(filtered_df)
    
    # Handle small datasets
    if len(rfm) < 4:
        st.warning("Not enough customers to build RFM clusters. Try widening your filters.")
    else:
        rfm_clustered, rfm_summary = compute_rfm_clusters(rfm)
        
        # Display Segment Grid dynamically without empty columns
        n_segments = len(rfm_summary)
        if n_segments > 0:
            sc_cols = st.columns(n_segments)
            for idx, row in rfm_summary.iterrows():
                with sc_cols[idx % n_segments]:
                    st.markdown(
                        textwrap.dedent(f"""
                        <div class="saas-segment-card">
                            <span style="font-size: 0.75rem; font-weight: 600; color: var(--accent-purple); text-transform: uppercase;">
                                {row['Segment']}
                            </span>
                            <div style="font-size: 1.4rem; font-weight: 700; color: var(--primary-text); margin-top: 4px;">
                                {row['Customers']:,} <span style="font-size:0.8rem; font-weight:400; color: var(--secondary-text);">Cust</span>
                            </div>
                            <div style="font-size: 0.8rem; color: var(--secondary-text); margin-top: 4px;">
                                Avg Spend: <b>₹{row['Avg_Monetary']:,}</b> | Avg Orders: <b>{row['Avg_Frequency']}</b>
                            </div>
                        </div>
                        """),
                        unsafe_allow_html=True
                    )
            st.markdown("---")
        
        c1, c2 = st.columns([2, 1])
        with c1:
            with st.container(border=True):
                render_plotly_chart(plot_rfm_scatter(rfm_clustered))
        with c2:
            st.markdown("**Segment Matrix Aggregates**")
            st.dataframe(rfm_summary, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════
#  PAGE: Sales Prediction Model
# ══════════════════════════════════════════════
elif page == "🔮 Sales Prediction Model":
    st.title("Sales & Customer Lifetime Value Prediction")
    
    # Cache training models
    with st.spinner("Training models..."):
        model, encoders, metrics, importance = train_sales_model(df)
        clv_df = build_clv_table(df)
        clv_model, clv_encoders, clv_acc = train_clv_classifier(clv_df)

    # State to Zone Mapping
    STATE_ZONE_MAP = df.groupby("State")["Zone"].first().to_dict()

    def map_age_to_group(age):
        if age <= 17:
            return "0-17"
        elif age <= 25:
            return "18-25"
        elif age <= 35:
            return "26-35"
        elif age <= 45:
            return "36-45"
        elif age <= 50:
            return "46-50"
        elif age <= 55:
            return "51-55"
        else:
            return "55+"

    def predict_clv_tier(clf, encoders, input_dict):
        row = {}
        feature_cols = ["Gender", "Age", "Marital_Status", "State", "Occupation"]
        for col in feature_cols:
            val = input_dict[col]
            if col in encoders:
                val = encoders[col].transform([str(val)])[0]
            row[col] = val
        X = pd.DataFrame([row])
        pred_idx = clf.predict(X)[0]
        tier_name = encoders["CLV_Tier"].inverse_transform([pred_idx])[0]
        
        # Classification confidence score
        probs = clf.predict_proba(X)[0]
        confidence = max(probs) * 100
        return tier_name, confidence

    # Create tabs to put prediction calculator first
    tab_calc, tab_perf = st.tabs(["🔮 Customer Value Predictor", "📊 Model Performance Evaluation"])

    with tab_calc:
        st.markdown("### Predict Expected Customer Value & Segment")
        st.markdown("Configure demographic inputs below to calculate expected sales and customer segment statistics in real-time.")

        # Demographics Card Container
        with st.container(border=True):
            st.markdown("##### 👥 Customer Profile Inputs")
            col1, col2 = st.columns(2)
            with col1:
                gender = st.selectbox("Gender", sorted(df["Gender"].unique()))
                age = st.slider("Age", 12, 92, 30)
                state = st.selectbox("State", sorted(df["State"].unique()))
            with col2:
                marital = st.selectbox("Marital Status", [0, 1], format_func=lambda x: "Married" if x else "Unmarried")
                orders = st.slider("Expected Orders Count", 1, 4, 2)
                
                # Dynamic Occupation Filter based on Gender, State, and Age Group (Prevents impossible combinations)
                age_group = map_age_to_group(age)
                valid_occ = df[
                    (df["State"] == state) &
                    (df["Gender"] == gender) &
                    (df["Age Group"] == age_group)
                ]["Occupation"].unique()
                
                if len(valid_occ) == 0:
                    valid_occ = df[
                        (df["State"] == state) &
                        (df["Gender"] == gender)
                    ]["Occupation"].unique()
                if len(valid_occ) == 0:
                    valid_occ = df[df["State"] == state]["Occupation"].unique()
                if len(valid_occ) == 0:
                    valid_occ = df["Occupation"].unique()
                    
                occupation = st.selectbox("Occupation", sorted(list(valid_occ)))
            
            # Automatically derive Zone and show info
            zone = STATE_ZONE_MAP.get(state, "Central")
            st.markdown(f"📍 Derived Geographical Zone: **{zone}** (Region derived automatically from selected State)")
            category = st.selectbox("Target Product Category Focus", sorted(df["Product_Category"].unique()))

        # Button to run prediction (hides results and trailing whitespace until clicked)
        if st.button("Generate Value Predictions", use_container_width=True):
            st.session_state.predicted = True

        if st.session_state.get("predicted", False):
            # Live prediction logic
            inp_reg = {
                "Gender": gender, "Age": age, "Marital_Status": marital,
                "State": state, "Zone": zone, "Occupation": occupation,
                "Product_Category": category, "Orders": orders,
            }
            pred_amount = predict_single(model, encoders, inp_reg)
            
            inp_clf = {
                "Gender": gender, "Age": age, "Marital_Status": marital,
                "State": state, "Occupation": occupation,
            }
            pred_tier, confidence = predict_clv_tier(clv_model, clv_encoders, inp_clf)

            # Similar Customer Cohort Statistics
            similar_df = df[
                (df["Gender"] == gender) &
                (df["Age Group"] == age_group) &
                (df["Occupation"] == occupation)
            ]
            if len(similar_df) == 0:
                similar_df = df[
                    (df["Gender"] == gender) &
                    (df["Age Group"] == age_group)
                ]
                
            similar_count = len(similar_df)
            similar_avg_spend = similar_df["Amount"].mean() if similar_count > 0 else 0
            similar_avg_orders = similar_df["Orders"].mean() if similar_count > 0 else 0

            # Category Recommendations Similarity mapping
            sim_df = build_similarity_matrix(df)
            rec_df = get_recommendations(sim_df, category, top_n=3)

            # Output Card Grid (Side-by-side KPI Cards)
            st.markdown("---")
            st.markdown("### 📊 Prediction Calculations Results")
            
            oc1, oc2, oc3 = st.columns(3)
            with oc1:
                amount_html = f"""
                <div class="saas-kpi-card" style="border-top: 4px solid var(--accent-blue) !important; height: 100%;">
                    <span style="font-size: 0.75rem; font-weight: 600; color: var(--secondary-text); text-transform: uppercase; letter-spacing: 0.05em;">
                        Predicted Purchase Value
                    </span>
                    <div style="font-size: 2.0rem; font-weight: 700; color: var(--primary-text); margin-top: 8px;">
                        ₹{pred_amount:,.2f}
                    </div>
                    <div style="font-size: 0.8rem; color: var(--secondary-text); margin-top: 8px;">
                        Expected spending amount for this transaction.
                    </div>
                </div>
                """
                render_html_card(amount_html)
                
            with oc2:
                segment_html = f"""
                <div class="saas-kpi-card" style="border-top: 4px solid var(--accent-purple) !important; height: 100%;">
                    <span style="font-size: 0.75rem; font-weight: 600; color: var(--secondary-text); text-transform: uppercase; letter-spacing: 0.05em;">
                        Predicted CLV Segment
                    </span>
                    <div style="font-size: 2.0rem; font-weight: 700; color: var(--accent-purple); margin-top: 8px;">
                        {pred_tier} Tier
                    </div>
                    <div style="font-size: 0.8rem; color: var(--secondary-text); margin-top: 8px; display: flex; align-items: center; gap: 6px;">
                        <span style="background: rgba(139, 92, 246, 0.1); color: var(--accent-purple); font-weight: 600; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem;">
                            {confidence:.1f}% Confidence
                        </span>
                    </div>
                </div>
                """
                render_html_card(segment_html)
                
            with oc3:
                stats_html = f"""
                <div class="saas-kpi-card" style="border-top: 4px solid #10B981 !important; height: 100%;">
                    <span style="font-size: 0.75rem; font-weight: 600; color: var(--secondary-text); text-transform: uppercase; letter-spacing: 0.05em;">
                        Similar Cohort Statistics
                    </span>
                    <div style="margin-top: 8px;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                            <span style="color: var(--secondary-text); font-size: 0.8rem;">Segment Size:</span>
                            <span style="color: var(--primary-text); font-weight: 600; font-size: 0.85rem;">{similar_count:,} cust</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                            <span style="color: var(--secondary-text); font-size: 0.8rem;">Average Spend:</span>
                            <span style="color: var(--primary-text); font-weight: 600; font-size: 0.85rem;">₹{similar_avg_spend:,.0f}</span>
                        </div>
                        <div style="display: flex; justify-content: space-between;">
                            <span style="color: var(--secondary-text); font-size: 0.8rem;">Average Orders:</span>
                            <span style="color: var(--primary-text); font-weight: 600; font-size: 0.85rem;">{similar_avg_orders:.1f}</span>
                        </div>
                    </div>
                </div>
                """
                render_html_card(stats_html)

            st.markdown("---")
            st.markdown("### 🛒 Recommended Product Categories")
            st.markdown("Top cross-selling categories derived from collaborative similarity mapping:")
            
            rec_cols = st.columns(3)
            for idx, row in rec_df.iterrows():
                col = rec_cols[idx % 3]
                pct_score = int(row['Similarity'] * 100)
                with col:
                    st.markdown(
                        textwrap.dedent(f"""
                        <div class="saas-rec-card">
                            <span style="font-size: 0.85rem; font-weight: 600; color: var(--primary-text);">
                                {row['Product_Category']}
                            </span>
                            <div style="margin-top: 6px;">
                                <span style="
                                    background: rgba(124, 58, 237, 0.1);
                                    color: var(--accent-purple);
                                    font-size: 0.75rem;
                                    font-weight: 600;
                                    padding: 2px 8px;
                                    border-radius: 12px;
                                ">
                                    {pct_score}% Similarity
                                </span>
                            </div>
                        </div>
                        """),
                        unsafe_allow_html=True
                    )
        else:
            st.info("👈 Enter customer profile parameters and click 'Generate Value Predictions' to compute expected values.")

    with tab_perf:
        # Explanation Alert Panel
        st.info("ℹ️ **Model Evaluation Notice**: These charts evaluate model performance and do not change with customer inputs.")

        # ---- Model metrics ----
        st.subheader("Model Performance Indicators")
        m1, m2, m3 = st.columns([1, 1, 1])
        with m1:
            gauge_bg = "#334155" if dark_mode else "#E2E8F0"
            gauge_step1 = "#475569" if dark_mode else "#CBD5E1"
            gauge_bar = "#3B82F6" if dark_mode else "#2563EB"
            
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=metrics["R² Score"],
                title={'text': "R² Score"},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 1]},
                    'bar': {'color': gauge_bar},
                    'steps': [
                        {'range': [0, 0.5], 'color': gauge_bg},
                        {'range': [0.5, 0.8], 'color': gauge_step1},
                        {'range': [0.8, 1], 'color': gauge_bar}
                    ]
                }
            ))
            with st.container(border=True):
                render_plotly_chart(fig_g)
            
        with m2:
            st.metric("Mean Absolute Error (MAE)", f"₹{metrics['MAE (₹)']:,.2f}")
        with m3:
            st.metric("Root Mean Squared Error (RMSE)", f"₹{metrics['RMSE (₹)']:,.2f}")

        # Visualizations
        st.markdown("---")
        st.subheader("ML Model Prediction Curves")
        
        # Run test predictions on a sample for visualization
        sample_df = df.sample(n=min(len(df), 200), random_state=42)
        sample_df_enc = sample_df.copy()
        for col in ["Gender", "State", "Zone", "Occupation", "Product_Category"]:
            sample_df_enc[col] = encoders[col].transform(sample_df[col].astype(str))
        
        FEATURE_COLS = ["Gender", "Age", "Marital_Status", "State", "Zone", "Occupation", "Product_Category", "Orders"]
        X_vis = sample_df_enc[FEATURE_COLS]
        y_vis = sample_df_enc["Amount"]
        y_vis_pred = model.predict(X_vis)
        residuals = y_vis - y_vis_pred
        
        vc1, vc2, vc3 = st.columns(3)
        with vc1:
            # Actual vs Predicted
            fig_ap = px.scatter(x=y_vis, y=y_vis_pred, labels={"x": "Actual Spend (₹)", "y": "Predicted Spend (₹)"},
                                title="Actual vs Predicted Spending")
            fig_ap.add_trace(go.Scatter(x=[y_vis.min(), y_vis.max()], y=[y_vis.min(), y_vis.max()],
                                        mode='lines', name='y=x Identity', line=dict(dash='dash', color='red')))
            with st.container(border=True):
                render_plotly_chart(fig_ap)
            
        with vc2:
            # Residual Plot
            fig_res = px.scatter(x=y_vis_pred, y=residuals, labels={"x": "Predicted Spend (₹)", "y": "Residual (₹)"},
                                 title="Residuals vs Predicted")
            fig_res.add_hline(y=0, line_dash="dash", line_color="red")
            with st.container(border=True):
                render_plotly_chart(fig_res)
            
        with vc3:
            # Feature Importance
            fig_imp = px.bar(importance, x="Importance", y="Feature", orientation="h",
                             color="Importance", color_continuous_scale="Tealgrn",
                             title="Feature Importance Weights")
            with st.container(border=True):
                render_plotly_chart(fig_imp)


# ══════════════════════════════════════════════
#  PAGE: Customer Lifetime Value
# ══════════════════════════════════════════════
elif page == "💎 Customer Lifetime Value":
    st.title("Customer Lifetime Value Analysis")
    st.markdown("Segment customers into CLV tiers (Low / Medium / High / VIP) based on their purchase history.")

    clv_df = build_clv_table(filtered_df)
    
    if clv_df is None or clv_df.empty:
         st.warning("No customer data available for the active filters to calculate CLV metrics.")
    else:
        # ---- KPIs ----
        k1, k2, k3 = st.columns(3)
        k1.metric("Total Customers", f"{len(clv_df):,}")
        k2.metric("Avg CLV Score", f"₹{clv_df['CLV_Score'].mean():,.0f}")
        k3.metric("VIP Customers", f"{(clv_df['CLV_Tier'] == 'VIP').sum():,}")

        st.markdown("---")

        c1, c2 = st.columns(2)
        with c1:
            with st.container(border=True):
                render_plotly_chart(plot_clv_distribution(clv_df))
        with c2:
            fig_stats, stats_table = plot_clv_tier_stats(clv_df)
            with st.container(border=True):
                render_plotly_chart(fig_stats)

        # Robust Aggregates rendering (Hides completely if empty to eliminate blank white space)
        if stats_table is not None and not stats_table.empty:
            st.markdown("---")
            st.subheader("Tier Summary Aggregates")
            st.dataframe(stats_table, use_container_width=True, hide_index=True)

        # ---- Top 20 Most Valuable Customers Table ----
        top_20 = clv_df.sort_values("CLV_Score", ascending=False).head(20)
        if not top_20.empty:
            st.markdown("---")
            st.subheader("Top 20 Most Valuable Customers")
            st.dataframe(
                top_20[["User_ID", "Cust_name", "Gender", "Age", "State", "Occupation", "Total_Orders", "Total_Amount", "CLV_Score", "CLV_Tier"]],
                use_container_width=True,
                hide_index=True
            )
            
            # Download button
            csv_data = clv_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Export Full CLV Table (CSV)", csv_data, "customer_clv_data.csv", "text/csv", use_container_width=True)


# ══════════════════════════════════════════════
#  PAGE: Product Recommendations
# ══════════════════════════════════════════════
elif page == "🛒 Product Recommendations":
    st.title("Product Recommendation Engine")
    st.markdown("Item-based collaborative filtering using cosine similarity on the User × Product matrix.")

    sim_df = build_similarity_matrix(filtered_df)
    categories = sorted(sim_df.index.tolist())

    selected = st.selectbox("Select a Product Category", categories)

    if selected:
        rec_df = get_recommendations(sim_df, selected, top_n=5)

        st.subheader(f"Top Recommendations for '{selected}'")
        
        # Render e-commerce style product recommendation cards dynamically (prevents empty columns)
        n_recs = len(rec_df)
        if n_recs > 0:
            r_cols = st.columns(n_recs)
            for idx, row in rec_df.iterrows():
                col = r_cols[idx]
                pct_score = int(row['Similarity'] * 100)
                with col:
                    st.markdown(
                        textwrap.dedent(f"""
                        <div class="saas-rec-card">
                            <span style="font-size: 0.8rem; font-weight: 500; color: var(--primary-text);">
                                {row['Product_Category']}
                            </span>
                            <div style="margin-top: 8px;">
                                <span style="
                                    background: rgba(124, 58, 237, 0.1);
                                    color: var(--accent-purple);
                                    font-size: 0.75rem;
                                    font-weight: 600;
                                    padding: 2px 8px;
                                    border-radius: 12px;
                                ">
                                    {pct_score}% Similarity
                                </span>
                            </div>
                        </div>
                        """),
                        unsafe_allow_html=True
                    )
        
        st.markdown("---")
        fig = plot_recommendations(rec_df, selected)
        if fig:
            with st.container(border=True):
                render_plotly_chart(fig)

    # Plot Category Similarity Heatmap & Product relationship network graph in tabs
    st.markdown("---")
    st.subheader("Category Similarity Network Mapping")
    t1, t2 = st.tabs(["Product Relationship Graph", "Similarity Heatmap Matrix"])
    with t1:
        # Build circular node network graph
        n_nodes = len(categories)
        angles = [2 * math.pi * i / n_nodes for i in range(n_nodes)]
        x_nodes = [math.cos(a) for a in angles]
        y_nodes = [math.sin(a) for a in angles]
        
        # Node trace
        node_trace = go.Scatter(
            x=x_nodes, y=y_nodes,
            mode='markers+text',
            text=categories,
            textposition="top center",
            hoverinfo='text',
            marker=dict(
                color='#7C3AED',
                size=20,
                line=dict(width=2, color='#2563EB')
            )
        )
        
        # Edge trace
        edge_x = []
        edge_y = []
        for i in range(n_nodes):
            for j in range(i+1, n_nodes):
                if sim_df.iloc[i, j] > 0.15:  # Similarity threshold
                    edge_x.extend([x_nodes[i], x_nodes[j], None])
                    edge_y.extend([y_nodes[i], y_nodes[j], None])
                    
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1.5, color='rgba(124, 58, 237, 0.3)'),
            hoverinfo='none',
            mode='lines'
        )
        
        fig_net = go.Figure(data=[edge_trace, node_trace])
        fig_net.update_layout(
            showlegend=False,
            hovermode='closest',
            margin=dict(b=40,l=40,r=40,t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        with st.container(border=True):
            render_plotly_chart(fig_net)
        
    with t2:
        with st.container(border=True):
            render_plotly_chart(plot_similarity_heatmap(sim_df))


# ══════════════════════════════════════════════
#  PAGE: India Geographical Sales
# ══════════════════════════════════════════════
elif page == "🗺️ Geographical Sales Analysis":
    st.title("Geographical Sales Distribution")
    st.markdown("Interactive geographic visualization of Diwali sales revenue across Indian states.")

    state_sales = build_state_sales(filtered_df)

    c1, c2 = st.columns([3, 1])
    with c1:
        # Full width bubble map wrapped inside a beautiful container
        with st.container(border=True):
            import inspect
            sig = inspect.signature(plot_india_bubble_map)
            if "dark_mode" in sig.parameters:
                fig_map = plot_india_bubble_map(state_sales, dark_mode=dark_mode)
            else:
                fig_map = plot_india_bubble_map(state_sales)
            render_plotly_chart(fig_map)
    with c2:
        st.subheader("State Ranking")
        with st.container(border=True):
            render_plotly_chart(plot_state_bar(state_sales))

    st.markdown("---")
    st.subheader("State Revenue Aggregates")
    if state_sales is not None and not state_sales.empty:
        with st.container(border=True):
            display = state_sales[["State", "Total_Revenue", "Total_Orders", "Customers"]].sort_values(
                "Total_Revenue", ascending=False
            )
            display["Total_Revenue"] = display["Total_Revenue"].apply(lambda x: f"₹{x:,.0f}")
            st.dataframe(display, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════
#  PAGE: Customer Directory CRM
# ══════════════════════════════════════════════
elif page == "👥 Customer Directory CRM":
    st.title("Customer Profiles Explorer")
    st.markdown("Search, sort, filter, and inspect detailed profiles of customers in the retail segment.")
    
    clv_df = build_clv_table(filtered_df)
    
    if clv_df is None or clv_df.empty:
        st.warning("No customer records found matching the active filters.")
    else:
        # Search & Filter controls
        sc1, sc2 = st.columns([2, 1])
        with sc1:
            crm_search = st.text_input("🔍 Search customer by name")
        with sc2:
            crm_tier = st.selectbox("Filter by CLV Segment", ["All Tiers", "VIP", "High", "Medium", "Low"])
            
        crm_df = clv_df.copy()
        if crm_search:
            crm_df = crm_df[crm_df["Cust_name"].str.contains(crm_search, case=False, na=False)]
        if crm_tier != "All Tiers":
            crm_df = crm_df[crm_df["CLV_Tier"] == crm_tier]
            
        if crm_df.empty:
            st.warning("No customers matching search criteria.")
        else:
            st.markdown("---")
            
            # Customer detail box selector (Dynamic to search results)
            cust_names = sorted(crm_df["Cust_name"].tolist())
            selected_name = st.selectbox("Select Customer to Inspect Profile", cust_names)
            
            if selected_name:
                c_profile = crm_df[crm_df["Cust_name"] == selected_name].iloc[0]
                
                # Render profile details in a card
                st.markdown(
                    textwrap.dedent(f"""
                    <div class="saas-profile-card">
                        <h3 style="margin-top:0px; color: var(--accent-blue);">{c_profile['Cust_name']}</h3>
                        <div style="display:grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-top: 15px;">
                            <div><span style="color: var(--secondary-text); font-size:0.8rem;">User ID</span><br/><b>{c_profile['User_ID']}</b></div>
                            <div><span style="color: var(--secondary-text); font-size:0.8rem;">Gender / Age</span><br/><b>{c_profile['Gender']} / {c_profile['Age']} yrs</b></div>
                            <div><span style="color: var(--secondary-text); font-size:0.8rem;">State</span><br/><b>{c_profile['State']}</b></div>
                            <div><span style="color: var(--secondary-text); font-size:0.8rem;">Occupation</span><br/><b>{c_profile['Occupation']}</b></div>
                            <div><span style="color: var(--secondary-text); font-size:0.8rem;">Total Orders</span><br/><b>{c_profile['Total_Orders']}</b></div>
                            <div><span style="color: var(--secondary-text); font-size:0.8rem;">Total Spent</span><br/><b>₹{c_profile['Total_Amount']:,}</b></div>
                            <div><span style="color: var(--secondary-text); font-size:0.8rem;">CLV Score</span><br/><b>₹{c_profile['CLV_Score']:,}</b></div>
                            <div>
                                <span style="color: var(--secondary-text); font-size:0.8rem;">CLV Tier</span><br/>
                                <span style="background:rgba(37, 99, 235, 0.1); color: var(--accent-blue); font-weight:600; padding:2px 8px; border-radius:4px; font-size:0.85rem;">
                                    {c_profile['CLV_Tier']}
                                </span>
                            </div>
                        </div>
                    </div>
                    """),
                    unsafe_allow_html=True
                )
                
            st.markdown("---")
            st.subheader("CRM Records Directory Table")
            
            # Pagination Controls
            pc1, pc2 = st.columns([1, 2])
            with pc1:
                page_size = st.selectbox("Rows per page", [10, 20, 50, 100], index=0)
            with pc2:
                total_rows = len(crm_df)
                total_pages = max(1, math.ceil(total_rows / page_size))
                page_num = st.number_input(f"Page (1 of {total_pages})", min_value=1, max_value=total_pages, value=1)
                
            start_idx = (page_num - 1) * page_size
            end_idx = min(start_idx + page_size, total_rows)
            
            page_df = crm_df.iloc[start_idx:end_idx]
            
            st.dataframe(
                page_df[["User_ID", "Cust_name", "Gender", "Age", "State", "Occupation", "Total_Orders", "Total_Amount", "CLV_Score", "CLV_Tier"]].sort_values("Cust_name"),
                use_container_width=True,
                hide_index=True
            )
            st.caption(f"Showing records {start_idx + 1} to {end_idx} of {total_rows}")


# ══════════════════════════════════════════════
#  PAGE: About
# ══════════════════════════════════════════════
elif page == "ℹ️ Platform Info":
    st.title("Diwali Sales Data Science Project")

    st.markdown("""
    ## Diwali Sales Data Analysis & Prediction Dashboard

    A comprehensive data science project that goes beyond basic EDA to include
    **machine learning models**, **customer segmentation**, **product recommendations**,
    and **geographic visualizations**.

    ### Core Capabilities

    | Feature | Description |
    |---------|-------------|
    | **Executive Overview** | Executive summary highlighting revenue performance, dynamic AI insights, sunburst breakdowns, and treemaps. |
    | **Sales Analytics** | Interactive charts exploring gender, age, state, occupation, and product trends. |
    | **Customer Segmentation** | RFM segment cards (VIP, High, Mid, Low Value cohorts) with interactive K-Means scatter clustering. |
    | **Sales Prediction** | Random Forest Regressor model predicting purchase value based on actual vs. predicted curves, gauges, and residual analysis. |
    | **CLV Analysis** | Customer Lifetime Value tiering (Low/Medium/High/VIP) with detailed state/occupation distributions. |
    | **Recommendations** | E-commerce style item-based collaborative filtering cards using cosine similarity. |
    | **Geographical Sales Analysis** | Premium GIS bubble maps illustrating state-wise revenue and customer volume. |
    | **Customer Directory CRM** | Search, filter, and inspect specific customer profile records. |

    ### Technologies Used

    - **Python 3.x** — Pandas, NumPy, Scikit-learn
    - **Streamlit** — Interactive dashboard framework
    - **Plotly** — Rich interactive visualizations

    ---

    **Author:** Anurup — [GitHub](https://github.com/CodeWithAnurup)
    """)
