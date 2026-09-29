import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px


def build_similarity_matrix(df):
    """Build item-based cosine similarity on User x Product_Category matrix."""
    # Pivot: rows=User_ID, cols=Product_Category, values=Amount (sum)
    user_cat = df.pivot_table(index="User_ID", columns="Product_Category",
                              values="Amount", aggfunc="sum", fill_value=0)

    # Transpose so rows=categories, then compute similarity
    cat_matrix = user_cat.T
    sim = cosine_similarity(cat_matrix)
    sim_df = pd.DataFrame(sim, index=cat_matrix.index, columns=cat_matrix.index)
    return sim_df


def get_recommendations(sim_df, category, top_n=5):
    """Return top-N similar product categories for a given category."""
    if category not in sim_df.index:
        return pd.DataFrame(columns=["Product_Category", "Similarity"])
    scores = sim_df[category].drop(category).sort_values(ascending=False).head(top_n)
    result = scores.reset_index()
    result.columns = ["Product_Category", "Similarity"]
    result["Similarity"] = result["Similarity"].round(4)
    return result


def plot_recommendations(rec_df, source_category):
    """Bar chart of recommended categories."""
    if rec_df.empty:
        return None
    fig = px.bar(rec_df, x="Similarity", y="Product_Category", orientation="h",
                 text_auto=".3f",
                 color="Similarity", color_continuous_scale="Sunset",
                 title=f"Top Recommendations for '{source_category}'")
    fig.update_layout(yaxis_title="", xaxis_title="Cosine Similarity",
                      coloraxis_showscale=False, yaxis=dict(autorange="reversed"))
    return fig


def plot_similarity_heatmap(sim_df):
    """Full heatmap of category similarities."""
    fig = px.imshow(sim_df, text_auto=".2f",
                    color_continuous_scale="RdYlGn",
                    title="Product Category Similarity Matrix")
    fig.update_layout(height=700)
    return fig
