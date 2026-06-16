# 🎆 Diwali Sales Data Analysis

A comprehensive **Exploratory Data Analysis (EDA)** project on Diwali season sales data to uncover customer buying patterns, identify high-value segments, and generate actionable business insights for targeted marketing and inventory planning.

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Analysis Workflow](#-analysis-workflow)
- [Key Business Insights](#-key-business-insights)
- [Customer Segmentation (RFM)](#-customer-segmentation-rfm)
- [Technologies Used](#-technologies-used)
- [Project Structure](#-project-structure)
- [How to Run](#-how-to-run)
- [Sample Visualizations](#-sample-visualizations)
- [Future Scope](#-future-scope)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Project Overview

This project analyzes Diwali sales transaction data to:

- **Identify** top-selling products and high-revenue product categories
- **Analyze** customer demographics — gender, age group, state, zone, occupation, and marital status
- **Understand** spending behavior across different customer segments
- **Segment** customers using RFM (Recency, Frequency, Monetary) analysis with K-Means clustering
- **Generate** data-driven insights for revenue growth, marketing strategies, and inventory optimization

---

## 📊 Dataset

| Property | Details |
|----------|---------|
| **File** | `Diwali Sales Data.csv` |
| **Records** | 11,251 transactions |
| **Features** | 15 columns |
| **Encoding** | `latin1` |

### Feature Description

| Column | Type | Description |
|--------|------|-------------|
| `User_ID` | int | Unique customer identifier |
| `Cust_name` | object | Customer name |
| `Product_ID` | object | Unique product identifier |
| `Gender` | object | Customer gender (M/F) |
| `Age Group` | object | Age bracket (0-17, 18-25, 26-35, 36-45, 46-50, 51-55, 55+) |
| `Age` | int | Customer age |
| `Marital_Status` | int | 0 = Unmarried, 1 = Married |
| `State` | object | Indian state of the customer |
| `Zone` | object | Geographic zone (Western, Southern, Central, Northern) |
| `Occupation` | object | Customer's occupation |
| `Product_Category` | object | Product category (Auto, Food, Clothing, Electronics, etc.) |
| `Orders` | int | Number of orders placed |
| `Amount` | float | Total purchase amount (₹) |
| `Status` | float | *Empty — dropped during cleaning* |
| `unnamed1` | float | *Empty — dropped during cleaning* |

---

## 🔬 Analysis Workflow

The analysis follows a structured data science workflow:

### 1. Data Loading & Inspection
- Loaded dataset with `pandas` using `latin1` encoding
- Inspected shape, data types, and first/last records with `df.shape`, `df.info()`, `df.head()`

### 2. Data Cleaning
- ✅ Dropped irrelevant empty columns (`Status`, `unnamed1`)
- ✅ Identified 12 null values in `Amount` column (~0.1% of data)
- ✅ Removed rows with missing values using `dropna()`
- ✅ Converted `Amount` from `float64` to `int64` for cleaner analysis

### 3. Exploratory Data Analysis (EDA)
- Generated descriptive statistics with `df.describe()`
- Performed multi-dimensional analysis across:

| Dimension | Analysis Performed |
|-----------|--------------------|
| **Gender** | Count distribution + Total spending by gender |
| **Age Group** | Transaction count + Spending patterns per age bracket |
| **State** | State-wise total sales and order volume |
| **Marital Status** | Spending comparison: Married vs. Unmarried |
| **Occupation** | Revenue contribution by profession |
| **Product Category** | Category-wise sales performance |
| **Top Products** | Top 10 best-selling products by order volume |

### 4. Customer Segmentation (RFM + K-Means Clustering)
- Computed **Frequency** and **Monetary** metrics per customer
- Applied **K-Means clustering** (k=4) to segment customers
- Labeled clusters with business-meaningful descriptions

### 5. Business Insights & Conclusions
- Synthesized findings into actionable recommendations

---

## 💡 Key Business Insights

Based on the analysis, the **ideal Diwali customer profile** is:

> **Married women, aged 26-35, from Uttar Pradesh, Maharashtra, and Karnataka, working in IT, Healthcare, or Aviation sectors, purchasing Food, Clothing, and Electronics.**

### Detailed Findings

| Insight | Detail |
|---------|--------|
| 🚺 **Gender** | Female customers outnumber males and contribute higher total spending |
| 🎂 **Age Group** | The 26-35 age group is the most active buyer segment |
| 📍 **Geography** | Uttar Pradesh, Maharashtra, and Karnataka are the top 3 revenue-generating states |
| 💍 **Marital Status** | Married customers show higher spending patterns |
| 💼 **Occupation** | IT, Healthcare, and Aviation professionals are the top spenders |
| 🛒 **Product Categories** | Food, Clothing & Apparel, and Electronics & Gadgets dominate sales |
| 📈 **Avg. Order Value** | Mean spending per transaction ≈ ₹9,454 |
| 📦 **Orders** | Average orders per transaction ≈ 2.5 |

---

## 👥 Customer Segmentation (RFM)

Using K-Means clustering on Frequency and Monetary values, customers were segmented into 4 groups:

| Cluster | Avg. Frequency | Avg. Monetary (₹) | Segment Label |
|---------|---------------:|-------------------:|---------------|
| 0 | 3.5 | ₹13,154 | 🟢 Low-value / Infrequent Buyers |
| 1 | 35.5 | ₹1,43,286 | 🔴 VIP Customers — Highest Value |
| 2 | 20.2 | ₹77,111 | 🟠 High-frequency & High-spending |
| 3 | 10.1 | ₹38,224 | 🟡 Mid-value Customers |

### Recommended Actions per Segment

- **Cluster 0 (Low-value):** Send re-engagement campaigns, offer first-purchase discounts
- **Cluster 1 (VIP):** Provide loyalty rewards, early access to deals, personalized recommendations
- **Cluster 2 (High-value):** Maintain engagement with exclusive offers, upsell premium products
- **Cluster 3 (Mid-value):** Nurture to upgrade — targeted cross-sell campaigns, bundle offers

---

## 🧰 Technologies Used

| Technology | Purpose |
|------------|---------|
| **Python 3.x** | Core programming language |
| **Jupyter Notebook** | Interactive analysis environment |
| **Pandas** | Data manipulation and analysis |
| **NumPy** | Numerical computations |
| **Matplotlib** | Data visualization (base plotting) |
| **Seaborn** | Statistical visualizations (bar plots, count plots) |
| **Scikit-learn** | K-Means clustering for RFM segmentation |

---

## 📁 Project Structure

```
Diwali-sales-DS-project/
│
├── Dewali sales/
│   ├── Diwali Sales Data.csv        # Raw dataset (11,251 records)
│   └── Diwali Sales Project.ipynb   # Main analysis notebook
│
└── README.md                        # Project documentation (this file)
```

---

## 🚀 How to Run

### Prerequisites
- Python 3.8+
- Jupyter Notebook or JupyterLab

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/CodeWithAnurup/Diwali-sales-DS-project.git
   cd Diwali-sales-DS-project
   ```

2. **Install dependencies**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter
   ```

3. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

4. **Open and run** `Dewali sales/Diwali Sales Project.ipynb`

> **Note:** You may need to update the file path in the first code cell to match your local directory structure.

---

## 📸 Sample Visualizations

The notebook generates the following types of visualizations:

- 📊 **Bar Charts** — Gender distribution, age group spending, state-wise sales
- 📈 **Grouped Bar Charts** — Marital status × gender spending
- 🏆 **Top-N Charts** — Top 10 products by order volume
- 🗺️ **Horizontal Bar Charts** — State-wise revenue comparison
- 🧩 **Cluster Summary Tables** — RFM customer segments

---

## 🔮 Future Scope

### Short-term Enhancements
- [ ] **Time-series analysis** — Analyze daily/weekly sales trends if timestamp data becomes available
- [ ] **Market basket analysis** — Discover frequently co-purchased products using Apriori algorithm
- [ ] **Statistical significance testing** — Validate demographic spending differences with t-tests/chi-square tests
- [ ] **Outlier detection** — Use IQR/Z-score methods to identify and handle anomalous transactions

### Medium-term Extensions
- [ ] **Predictive modeling** — Build regression models (Random Forest, XGBoost) to predict purchase amount based on customer demographics
- [ ] **Churn prediction** — Classify customers at risk of not returning for next Diwali season
- [ ] **Customer Lifetime Value (CLV)** — Estimate long-term revenue per customer segment
- [ ] **Interactive dashboard** — Build a Streamlit/Dash web app with real-time filters and KPI cards
- [ ] **Advanced clustering** — Experiment with DBSCAN, Hierarchical, or Gaussian Mixture Models

### Long-term / Production Goals
- [ ] **Recommendation engine** — Collaborative filtering or content-based product recommendations
- [ ] **Geospatial visualization** — Plot state-wise sales on an interactive India map using `folium` or `geopandas`
- [ ] **Real-time data pipeline** — Stream sales data with Apache Kafka/Spark for live festive-season monitoring
- [ ] **ML-powered dynamic pricing** — Optimize pricing based on demand patterns and customer segments
- [ ] **Multi-year comparison** — Compare Diwali sales across years to identify growth trends
- [ ] **Automated reporting** — Schedule periodic PDF/email reports with Airflow or cron jobs

---

## 🤝 Contributing

Contributions are welcome! If you'd like to improve this project:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is open source and available for educational and personal use.

---

## 👨‍💻 Author

**Anurup** — [GitHub Profile](https://github.com/CodeWithAnurup)

---

> ⭐ If you found this project helpful, please consider giving it a star!
