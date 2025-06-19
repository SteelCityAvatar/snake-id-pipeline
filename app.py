import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import os

# Set page config
st.set_page_config(
    page_title="Snake Identification Research Project",
    page_icon="🐍",
    layout="wide"
)

# Title and description
st.title("🐍 Snake Identification Research Project")
st.markdown("""
This dashboard presents the results of our snake identification pipeline, which uses GPT-4 and Gemini models
to identify snakes from Reddit posts. The pipeline scrapes images from r/whatsthissnake, processes them
through our models, and compares the results with expert identifications.
""")

# Load data
@st.cache_data
def load_data():
    results_path = Path("results/classification_results.csv")
    metrics_path = Path("results/metrics_summary.csv")
    
    if not results_path.exists() or not metrics_path.exists():
        st.error("Please run the pipeline first to generate results!")
        return None, None
    
    results_df = pd.read_csv(results_path)
    metrics_df = pd.read_csv(metrics_path)
    return results_df, metrics_df

results_df, metrics_df = load_data()

if results_df is not None and metrics_df is not None:
    # Create two columns for metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Performance Metrics")
        metrics = {
            "Precision": metrics_df["precision"].iloc[0],
            "Recall": metrics_df["recall"].iloc[0],
            "Accuracy": metrics_df["accuracy"].iloc[0],
            "F1 Score": metrics_df["f1_score"].iloc[0]
        }
        
        # Display metrics in a nice format
        for metric, value in metrics.items():
            st.metric(metric, f"{value:.3f}")
    
    with col2:
        st.subheader("Dataset Statistics")
        stats = {
            "Total Known Cases": metrics_df["total_known"].iloc[0],
            "Total Unknown Cases": metrics_df["total_unknown"].iloc[0],
            "True Positives": metrics_df["true_positives"].iloc[0],
            "False Positives": metrics_df["false_positives"].iloc[0],
            "False Negatives": metrics_df["false_negatives"].iloc[0]
        }
        
        for stat, value in stats.items():
            st.metric(stat, value)
    
    # Create confusion matrix
    st.subheader("Confusion Matrix")
    conf_matrix = go.Figure(data=go.Heatmap(
        z=[[metrics_df["true_positives"].iloc[0], metrics_df["false_positives"].iloc[0]],
           [metrics_df["false_negatives"].iloc[0], 0]],
        x=["Predicted Positive", "Predicted Negative"],
        y=["Actual Positive", "Actual Negative"],
        colorscale="Blues"
    ))
    conf_matrix.update_layout(title="Confusion Matrix")
    st.plotly_chart(conf_matrix, use_container_width=True)
    
    # Display results table
    st.subheader("Detailed Results")
    
    # Add image preview functionality
    def display_image(image_path):
        if os.path.exists(image_path):
            return st.image(image_path, width=200)
        return st.write("Image not found")
    
    # Create tabs for different views
    tab1, tab2 = st.tabs(["Results Table", "Image Gallery"])
    
    with tab1:
        # Filter results
        st.write("Filter results:")
        col1, col2 = st.columns(2)
        with col1:
            correct_filter = st.selectbox(
                "Correctness",
                ["All", "Correct", "Incorrect"],
                index=0
            )
        with col2:
            ground_truth_filter = st.selectbox(
                "Ground Truth Status",
                ["All", "Known", "Unknown"],
                index=0
            )
        
        # Apply filters
        filtered_df = results_df.copy()
        if correct_filter != "All":
            filtered_df = filtered_df[filtered_df["correct"] == (correct_filter == "Correct")]
        if ground_truth_filter != "All":
            filtered_df = filtered_df[filtered_df["ground_truth"] == ("UNKNOWN" if ground_truth_filter == "Unknown" else "KNOWN")]
        
        st.dataframe(filtered_df)
    
    with tab2:
        # Display images in a grid
        cols = st.columns(3)
        for idx, row in results_df.iterrows():
            with cols[idx % 3]:
                if "image_path" in row and os.path.exists(row["image_path"]):
                    st.image(row["image_path"], width=200)
                    st.write(f"GPT Label: {row['gpt_label']}")
                    st.write(f"Ground Truth: {row['ground_truth']}")
                    st.write(f"Correct: {'✅' if row['correct'] else '❌'}")
                    st.write("---")

# Add footer
st.markdown("---")
st.markdown("""
### About this Project
This project uses GPT-4 and Gemini models to identify snakes from Reddit posts. The pipeline:
1. Scrapes images from r/whatsthissnake
2. Processes them through GPT-4 for identification
3. Uses Gemini to extract ground truth from expert comments
4. Evaluates the model's performance

The results are presented in this interactive dashboard.
""") 