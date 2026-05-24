import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px


# Page title
st.title("Brazilian E-Commerce Retail Dashboard")


# Load dataset
@st.cache_data
def load_data():
    data = pd.read_csv("master_retail_dataset.csv")
    data["order_purchase_timestamp"] = pd.to_datetime(
        data["order_purchase_timestamp"],
        errors="coerce"
    )
    return data


data_load_state = st.text("Loading data...")
df = load_data()
data_load_state.text("Done! Dataset loaded successfully.")


# Dashboard description
st.write(
    """
    This dashboard summarises the most important aspects of the Brazilian Olist retail dataset.
    """
)


# Show raw data option similar to Streamlit sample
if st.checkbox("Show raw data"):
    st.subheader("Raw Data")
    st.write(df.head(100))


# Sidebar filters
st.sidebar.header("Filters")

selected_state = st.sidebar.selectbox(
    "Choose Customer State",
    ["All"] + sorted(df["customer_state"].dropna().unique().tolist())
)

selected_payment = st.sidebar.selectbox(
    "Choose Payment Type",
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


# KPI summary
st.subheader("Dashboard Summary")

total_orders = filtered_df["order_id"].nunique()
total_customers = filtered_df["customer_unique_id"].nunique()
total_products = filtered_df["product_id"].nunique()
total_sales = filtered_df["payment_value"].sum()

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Orders", f"{total_orders:,}")
col2.metric("Total Customers", f"{total_customers:,}")
col3.metric("Total Products", f"{total_products:,}")
col4.metric("Total Sales", f"{total_sales:,.2f}")


# Monthly sales chart
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
    title_font_size=24,
    font=dict(size=16),
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
    title_font_size=24,
    font=dict(size=16),
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
    title_font_size=24,
    font=dict(size=16),
    xaxis_title="Customer State",
    yaxis_title="Order Count"
)

st.plotly_chart(fig_states, use_container_width=True)


# Payment type distribution
st.subheader("Payment Type Distribution")

payment_summary = (
    filtered_df
    .groupby("payment_type")["order_id"]
    .count()
    .sort_values(ascending=False)
    .reset_index()
)

payment_summary.columns = ["Payment Type", "Number of Orders"]

fig_payment = px.bar(
    payment_summary,
    x="Payment Type",
    y="Number of Orders",
    title="Payment Type Distribution"
)

fig_payment.update_layout(
    title_font_size=24,
    font=dict(size=16),
    xaxis_title="Payment Type",
    yaxis_title="Number of Orders"
)

st.plotly_chart(fig_payment, use_container_width=True)


# Price vs freight
st.subheader("Price vs Freight Value")

scatter_sample = filtered_df.sample(
    min(5000, len(filtered_df)),
    random_state=42
) if len(filtered_df) > 0 else filtered_df

fig_scatter = px.scatter(
    scatter_sample,
    x="price",
    y="freight_value",
    title="Price vs Freight Value",
    opacity=0.6
)

fig_scatter.update_layout(
    title_font_size=24,
    font=dict(size=16),
    xaxis_title="Price",
    yaxis_title="Freight Value"
)

st.plotly_chart(fig_scatter, use_container_width=True)




# Revenue by payment type pie chart
st.subheader("Revenue Distribution by Payment Type")

payment_revenue = (
    filtered_df
    .groupby("payment_type")["payment_value"]
    .sum()
    .reset_index()
)

payment_revenue.columns = ["Payment Type", "Total Revenue"]

fig_pie = px.pie(
    payment_revenue,
    names="Payment Type",
    values="Total Revenue",
    title="Revenue Distribution by Payment Type"
)

fig_pie.update_layout(
    title_font_size=24,
    font=dict(size=16)
)

st.plotly_chart(fig_pie, use_container_width=True)


# Average price by category
st.subheader("Average Product Price by Category")

avg_price_category = (
    filtered_df
    .groupby("product_category_name_english")["price"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)

avg_price_category.columns = ["Product Category", "Average Price"]

fig_avg_price = px.bar(
    avg_price_category,
    x="Product Category",
    y="Average Price",
    title="Top Categories by Average Product Price"
)

fig_avg_price.update_layout(
    title_font_size=24,
    font=dict(size=16),
    xaxis_title="Product Category",
    yaxis_title="Average Price"
)

st.plotly_chart(fig_avg_price, use_container_width=True)
# Plot Explanation
st.subheader("Why this Dataset is Suitable for Machine Learning")

st.write(
    """
    The dataset contains customer IDs, product IDs, order IDs, product categories,
    payment values, timestamps, price, and freight values. These fields support
    recommendation systems, market basket analysis, customer behaviour analysis,
    and dashboard-based business intelligence.
    """
)
