import pandas as pd
import plotly.express as px

# ISO 3166-2:IN codes for Indian states (for Plotly choropleth)
# Mapping dataset state names → ISO codes
STATE_ISO_MAP = {
    "Andhra Pradesh": "IN-AP",
    "Bihar": "IN-BR",
    "Delhi": "IN-DL",
    "Gujarat": "IN-GJ",
    "Haryana": "IN-HR",
    "Himachal Pradesh": "IN-HP",
    "Jharkhand": "IN-JH",
    "Karnataka": "IN-KA",
    "Kerala": "IN-KL",
    "Madhya Pradesh": "IN-MP",
    "Maharashtra": "IN-MH",
    "Punjab": "IN-PB",
    "Rajasthan": "IN-RJ",
    "Telangana": "IN-TG",
    "Uttar Pradesh": "IN-UP",
    "Uttarakhand": "IN-UT",
}

# Approximate lat/lon for each state (for scatter_geo fallback)
STATE_COORDS = {
    "Andhra Pradesh": (15.9129, 79.74),
    "Bihar": (25.0961, 85.3131),
    "Delhi": (28.7041, 77.1025),
    "Gujarat": (22.2587, 71.1924),
    "Haryana": (29.0588, 76.0856),
    "Himachal Pradesh": (31.1048, 77.1734),
    "Jharkhand": (23.6102, 85.2799),
    "Karnataka": (15.3173, 75.7139),
    "Kerala": (10.8505, 76.2711),
    "Madhya Pradesh": (22.9734, 78.6569),
    "Maharashtra": (19.7515, 75.7139),
    "Punjab": (31.1471, 75.3412),
    "Rajasthan": (27.0238, 74.2179),
    "Telangana": (18.1124, 79.0193),
    "Uttar Pradesh": (26.8467, 80.9462),
    "Uttarakhand": (30.0668, 79.0193),
}


def build_state_sales(df):
    """Aggregate sales by state and add geographic info."""
    state_sales = df.groupby("State").agg(
        Total_Revenue=("Amount", "sum"),
        Total_Orders=("Orders", "sum"),
        Customers=("User_ID", "nunique"),
    ).reset_index()

    state_sales["ISO"] = state_sales["State"].map(STATE_ISO_MAP)
    state_sales["Lat"] = state_sales["State"].map(lambda s: STATE_COORDS.get(s, (20, 78))[0])
    state_sales["Lon"] = state_sales["State"].map(lambda s: STATE_COORDS.get(s, (20, 78))[1])

    return state_sales


def plot_india_bubble_map(state_sales, dark_mode=False):
    """Interactive bubble map of India showing state-wise revenue."""
    fig = px.scatter_geo(
        state_sales,
        lat="Lat",
        lon="Lon",
        size="Total_Revenue",
        color="Total_Revenue",
        hover_name="State",
        hover_data={"Total_Revenue": ":,.0f", "Total_Orders": True, "Customers": True,
                    "Lat": False, "Lon": False},
        color_continuous_scale="YlOrRd",
        size_max=50,
        title="India — State-wise Diwali Sales Revenue",
    )
    
    # Theme-based geo styling
    if dark_mode:
        land_color = "#1E293B"
        ocean_color = "#0F172A"
        country_color = "#334155"
        subunit_color = "#334155"
    else:
        land_color = "#F1F5F9"
        ocean_color = "#E2E8F0"
        country_color = "#CBD5E1"
        subunit_color = "#E2E8F0"

    fig.update_geos(
        scope="asia",
        center=dict(lat=22, lon=79),
        projection_scale=4,
        showland=True, landcolor=land_color,
        showocean=True, oceancolor=ocean_color,
        showcountries=True, countrycolor=country_color,
        showsubunits=True, subunitcolor=subunit_color,
    )
    fig.update_layout(height=650, margin=dict(l=0, r=0, t=50, b=0),
                      coloraxis_colorbar_title="Revenue (₹)")
    return fig


def plot_state_bar(state_sales):
    """Horizontal bar chart of state revenue."""
    ss = state_sales.sort_values("Total_Revenue", ascending=True)
    fig = px.bar(ss, x="Total_Revenue", y="State", orientation="h",
                 text_auto=".2s", color="Total_Revenue",
                 color_continuous_scale="YlOrRd",
                 title="State-wise Revenue Ranking")
    fig.update_layout(yaxis_title="", xaxis_title="Total Revenue (₹)",
                      coloraxis_showscale=False, height=500)
    return fig
