
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px



st.set_page_config(
    page_title="Brazilian Ecommerce Retail Dashboard",
    layout="wide"
)



# Load dataset

@st.cache_data
def load_data():
    df = pd.read_csv("master_retail_dataset.csv")

    df["order_purchase_timestamp"] = pd.to_datetime(
        df["order_purchase_timestamp"]
    )

    return df


df = load_data()



# Dashboard title


st.title("Brazilian E-Commerce Retail Dashboard")

st.markdown(
    """
    This dashboard summarises key retail insights from the merged Olist e-commerce dataset.
    It is designed with clear charts, large labels, simple filters, and minimal clutter to support adults aged 65+.
    """
)



# Sidebar filters


st.sidebar.header("Dashboard Filters")

selected_state = st.sidebar.selectbox(
    "Select Customer State",
    ["All"] + sorted(df["customer_state"].dropna().unique().tolist())
)

selected_payment = st.sidebar.selectbox(
    "Select Payment Type",
    ["All"] + sorted(df["payment_type"].dropna().unique().tolist())
)


filtered_df = df.copy()

if selected_state != "All":
    filtered_df = filtered_df[
        filtered_df["customer_state"] == selected_state
    ]

if selected_payment != "All":
    filtered_df = filtered_df[
        filtered_df["payment_type"] == selected_payment
    ]



# KPI cards


total_orders = filtered_df["order_id"].nunique()
total_customers = filtered_df["customer_unique_id"].nunique()
total_products = filtered_df["product_id"].nunique()
total_sales = filtered_df["payment_value"].sum()

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Orders", f"{total_orders:,}")
col2.metric("Total Customers", f"{total_customers:,}")
col3.metric("Total Products", f"{total_products:,}")
col4.metric("Total Sales", f"{total_sales:,.2f}")



# Monthly sales trend


st.subheader("Monthly Sales Trend")

monthly_sales = (
    filtered_df
    .groupby(filtered_df["order_purchase_timestamp"].dt.to_period("M"))["payment_value"]
    .sum()
    .reset_index()
)

monthly_sales["order_purchase_timestamp"] = (
    monthly_sales["order_purchase_timestamp"].astype(str)
)

monthly_sales.columns = ["Month", "Total Sales"]

fig_monthly = px.line(
    monthly_sales,
    x="Month",
    y="Total Sales",
    markers=True,
    title="Monthly Sales Trend"
)

fig_monthly.update_layout(
    title_font_size=26,
    font=dict(size=18),
    xaxis_title="Month",
    yaxis_title="Total Sales"
)

st.plotly_chart(fig_monthly, use_container_width=True)



# Top product categories


st.subheader("Top Product Categories")

top_categories = (
    filtered_df
    .groupby("product_category_name_english")["order_id"]
    .count()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)

top_categories.columns = ["Product Category", "Order Count"]

fig_categories = px.bar(
    top_categories,
    x="Product Category",
    y="Order Count",
    title="Top 10 Product Categories"
)

fig_categories.update_layout(
    title_font_size=26,
    font=dict(size=18),
    xaxis_title="Product Category",
    yaxis_title="Order Count"
)

st.plotly_chart(fig_categories, use_container_width=True)



# Orders by state


st.subheader("Orders by Customer State")

state_orders = (
    filtered_df
    .groupby("customer_state")["order_id"]
    .count()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)

state_orders.columns = ["Customer State", "Order Count"]

fig_states = px.bar(
    state_orders,
    x="Customer State",
    y="Order Count",
    title="Top 10 States by Orders"
)

fig_states.update_layout(
    title_font_size=26,
    font=dict(size=18),
    xaxis_title="Customer State",
    yaxis_title="Order Count"
)

st.plotly_chart(fig_states, use_container_width=True)



# Payment type analysis


st.subheader("Payment Type Revenue Distribution")

payment_summary = (
    filtered_df
    .groupby("payment_type")["payment_value"]
    .sum()
    .sort_values(ascending=False)
    .reset_index()
)

payment_summary.columns = ["Payment Type", "Total Revenue"]

fig_payment = px.pie(
    payment_summary,
    names="Payment Type",
    values="Total Revenue",
    title="Revenue by Payment Type"
)

fig_payment.update_layout(
    title_font_size=26,
    font=dict(size=18)
)

st.plotly_chart(fig_payment, use_container_width=True)



# Dataset suitability for ML


st.subheader("Why this Dataset is Suitable for Machine Learning")

st.markdown(
    """
    The dataset is suitable for machine learning because it contains customer IDs, order IDs,
    product IDs, product categories, payment values, timestamps, and transaction behaviour.
    These fields support recommendation systems, customer behaviour analysis, market basket analysis,
    and dashboard-based business intelligence.
    """
)



# Data preview


st.subheader("Filtered Data Preview")

st.dataframe(
    filtered_df.head(50),
    use_container_width=True
)
