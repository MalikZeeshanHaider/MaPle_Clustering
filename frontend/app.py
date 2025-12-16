"""
MaPle Clustering - Streamlit Frontend Application

This is the main frontend interface for the MaPle (Maximal Pattern-based Clustering) system.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, Dict, List
import time

from utils.api_client import MapleAPIClient


# Page configuration
st.set_page_config(
    page_title="MaPle Clustering",
    page_icon="🍁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful UI
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styling */
    .stApp {
        font-family: 'Inter', sans-serif;
        background-color: #f5f7fa;
    }
    
    /* Ensure all text is visible */
    .stApp, .stApp p, .stApp span, .stApp div {
        color: #2d3748;
    }
    
    /* Main header with maple leaf color */
    .main-header {
        font-size: 3.5rem;
        color: #e85d75 !important;
        text-align: center;
        font-weight: 700;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
    }
    
    .sub-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #1a202c !important;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #319795;
        display: inline-block;
    }
    
    /* Card styling */
    .metric-card {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        border: 1px solid #e2e8f0;
        margin: 0.5rem 0;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.12);
    }
    
    /* Sidebar styling - Light theme for better readability */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2d3748 0%, #1a202c 100%);
    }
    
    section[data-testid="stSidebar"] .stMarkdown {
        color: #f7fafc !important;
    }
    
    section[data-testid="stSidebar"] .stMarkdown p {
        color: #e2e8f0 !important;
    }
    
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #81e6d9 !important;
        font-weight: 600;
    }
    
    /* Sidebar radio buttons - all text white */
    section[data-testid="stSidebar"] .stRadio label {
        color: #ffffff !important;
        font-weight: 500;
    }
    
    section[data-testid="stSidebar"] .stRadio label span {
        color: #ffffff !important;
    }
    
    section[data-testid="stSidebar"] .stRadio > label > div[data-testid="stMarkdownContainer"] > p {
        color: #ffffff !important;
        font-size: 1rem !important;
    }
    
    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
        color: #ffffff !important;
    }
    
    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label p {
        color: #ffffff !important;
    }
    
    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label div {
        color: #ffffff !important;
    }
    
    section[data-testid="stSidebar"] [data-baseweb="radio"] {
        color: #ffffff !important;
    }
    
    section[data-testid="stSidebar"] [data-baseweb="radio"] > div {
        color: #ffffff !important;
    }
    
    section[data-testid="stSidebar"] [data-baseweb="radio"] label {
        color: #ffffff !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #e53e3e 0%, #c53030 100%);
        color: #ffffff !important;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(229, 62, 62, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(229, 62, 62, 0.4);
        color: #ffffff !important;
    }
    
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #319795 0%, #2c7a7b 100%);
        box-shadow: 0 4px 15px rgba(49, 151, 149, 0.3);
    }
    
    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 6px 20px rgba(49, 151, 149, 0.4);
    }
    
    /* Form styling */
    .stForm {
        background: #ffffff;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.1);
        border: 1px solid #e2e8f0;
    }
    
    /* Metric styling */
    [data-testid="stMetric"] {
        background: #ffffff;
        padding: 1rem 1.25rem;
        border-radius: 14px;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.06);
        border-left: 4px solid #319795;
    }
    
    [data-testid="stMetric"] label {
        color: #4a5568 !important;
        font-weight: 500;
    }
    
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #1a202c !important;
        font-weight: 700;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #edf2f7;
        padding: 0.5rem;
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        color: #2d3748 !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #319795 0%, #2c7a7b 100%);
        color: #ffffff !important;
    }
    
    /* DataFrame styling */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.08);
    }
    
    /* Info/Success/Warning/Error boxes */
    .stAlert {
        border-radius: 12px;
        border: none;
    }
    
    /* Slider styling */
    .stSlider label {
        color: #2d3748 !important;
    }
    
    .stSlider [data-baseweb="slider"] {
        margin-top: 1rem;
    }
    
    /* Select box */
    .stSelectbox label {
        color: #2d3748 !important;
        font-weight: 500;
    }
    
    .stSelectbox [data-baseweb="select"] {
        border-radius: 10px;
        background-color: #2d3748 !important;
    }
    
    .stSelectbox [data-baseweb="select"] > div {
        color: #ffffff !important;
        font-weight: 500;
    }
    
    .stSelectbox [role="button"] {
        color: #ffffff !important;
    }
    
    /* Dropdown menu */
    ul[role="listbox"] {
        background-color: #ffffff !important;
    }
    
    ul[role="listbox"] li {
        color: #2d3748 !important;
    }
    
    ul[role="listbox"] li:hover {
        background-color: #e2e8f0 !important;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background: #ffffff;
        padding: 2rem;
        border-radius: 16px;
        border: 2px dashed #319795;
    }
    
    [data-testid="stFileUploader"] label {
        color: #2d3748 !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: #ffffff;
        border-radius: 12px;
        font-weight: 600;
        color: #2d3748 !important;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(135deg, #319795 0%, #e53e3e 100%);
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #319795 !important;
    }
    
    /* Hero section */
    .hero-subtitle {
        text-align: center;
        color: #1a202c !important;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        font-weight: 500;
    }
    
    /* Feature cards */
    .feature-card {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        text-align: center;
        transition: transform 0.2s ease;
        border: 1px solid #e2e8f0;
    }
    
    .feature-card h4 {
        color: #1a202c !important;
        margin-bottom: 0.5rem;
    }
    
    .feature-card p {
        color: #4a5568 !important;
        font-size: 0.9rem;
        margin: 0;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.12);
    }
    
    .feature-icon {
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    
    /* Divider */
    .custom-divider {
        height: 3px;
        background: linear-gradient(90deg, transparent, #319795, transparent);
        margin: 2rem 0;
        border: none;
    }
    
    /* Status badge */
    .status-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    .status-online {
        background: linear-gradient(135deg, #38a169 0%, #48bb78 100%);
        color: #ffffff !important;
    }
    
    .status-offline {
        background: linear-gradient(135deg, #e53e3e 0%, #fc8181 100%);
        color: #ffffff !important;
    }
    
    /* Form labels */
    .stTextInput label, .stNumberInput label, .stTextArea label {
        color: #2d3748 !important;
    }
    
    /* Radio buttons in main area */
    .stRadio label {
        color: #2d3748 !important;
    }
    
    /* Checkbox */
    .stCheckbox label {
        color: #2d3748 !important;
    }
    
    /* Multiselect */
    .stMultiSelect label {
        color: #2d3748 !important;
    }
    
    /* Date input */
    .stDateInput label {
        color: #2d3748 !important;
    }
    
    /* Text colors for various elements */
    h1, h2, h3, h4, h5, h6 {
        color: #1a202c !important;
    }
    
    /* Markdown text */
    .stMarkdown {
        color: #2d3748;
    }
    
    /* Table text */
    .stDataFrame td, .stDataFrame th {
        color: #2d3748 !important;
    }
    
    /* Expander content */
    .streamlit-expanderContent {
        color: #2d3748 !important;
    }
    
    .streamlit-expanderContent p {
        color: #2d3748 !important;
    }
    
    /* Chart text visibility */
    .js-plotly-plot .plotly text {
        fill: #2d3748 !important;
    }
    
    .js-plotly-plot .plotly .xtick text,
    .js-plotly-plot .plotly .ytick text {
        fill: #2d3748 !important;
        font-weight: 500 !important;
    }
    
    /* Responsive Design */
    @media (max-width: 1200px) {
        [data-testid="column"] {
            min-width: 200px;
        }
    }
    
    @media (max-width: 768px) {
        .main-header {
            font-size: 2.5rem;
        }
        
        .sub-header {
            font-size: 1.4rem;
        }
        
        .hero-subtitle {
            font-size: 1rem;
        }
        
        .feature-card {
            padding: 1rem;
            margin-bottom: 1rem;
        }
        
        .feature-icon {
            font-size: 2rem;
        }
        
        [data-testid="stMetric"] {
            padding: 0.75rem 1rem;
            margin-bottom: 0.5rem;
        }
        
        .stButton > button {
            padding: 0.6rem 1rem;
            font-size: 0.9rem;
        }
        
        [data-testid="column"] {
            flex: 1 1 100% !important;
            min-width: 100% !important;
        }
        
        .stTabs [data-baseweb="tab"] {
            padding: 0.5rem 1rem;
            font-size: 0.9rem;
        }
    }
    
    @media (max-width: 480px) {
        .main-header {
            font-size: 2rem;
        }
        
        .sub-header {
            font-size: 1.2rem;
        }
        
        .hero-subtitle {
            font-size: 0.9rem;
        }
        
        .feature-card {
            padding: 0.75rem;
        }
        
        [data-testid="stMetric"] {
            padding: 0.5rem 0.75rem;
        }
        
        [data-testid="stMetric"] label {
            font-size: 0.8rem !important;
        }
        
        [data-testid="stMetric"] [data-testid="stMetricValue"] {
            font-size: 1.2rem !important;
        }
    }
    
    /* Improve text contrast everywhere */
    .stMarkdown strong {
        color: #1a202c !important;
    }
    
    /* File uploader text */
    [data-testid="stFileUploader"] section {
        color: #2d3748 !important;
    }
    
    [data-testid="stFileUploader"] section button {
        color: #2d3748 !important;
        font-weight: 500;
    }
    
    /* Hide Streamlit branding */
    footer {
        visibility: hidden;
    }
    
    footer:after {
        content: '';
        visibility: visible;
        display: block;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if 'api_client' not in st.session_state:
    st.session_state.api_client = MapleAPIClient()

if 'dataset_loaded' not in st.session_state:
    st.session_state.dataset_loaded = False

if 'clustering_done' not in st.session_state:
    st.session_state.clustering_done = False

if 'dataset_info' not in st.session_state:
    st.session_state.dataset_info = None

if 'clustering_results' not in st.session_state:
    st.session_state.clustering_results = None

if 'evaluation_results' not in st.session_state:
    st.session_state.evaluation_results = None


def main():
    """Main application function."""
    
    # Header
    st.markdown('<h1 class="main-header">🍁 MaPle Clustering</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-subtitle">Maximal Pattern-based Clustering Algorithm</p>',
        unsafe_allow_html=True
    )
    
    # Check API health
    with st.sidebar:
        st.markdown("## 🎛️ Control Panel")
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        
        st.markdown("### 📡 System Status")
        health = st.session_state.api_client.health_check()
        
        if health.get('status') == 'healthy':
            st.markdown('<span class="status-badge status-online">✓ API Online</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-badge status-offline">✗ API Offline</span>', unsafe_allow_html=True)
            st.info("Please ensure the backend is running:\n```bash\npython -m backend.main\n```")
            return
        
        st.markdown("---")
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown("### 🧭 Navigation")
        page = st.radio(
            "Select Page",
            ["📤 Upload Dataset", "⚙️ Run MaPle", "📊 Results & Evaluation", "ℹ️ About"],
            label_visibility="collapsed"
        )
    
    # Render selected page
    if page == "📤 Upload Dataset":
        render_upload_page()
    elif page == "⚙️ Run MaPle":
        render_maple_page()
    elif page == "📊 Results & Evaluation":
        render_results_page()
    elif page == "ℹ️ About":
        render_about_page()


def render_upload_page():
    """Render dataset upload page."""
    st.markdown('<p class="sub-header">📤 Upload Dataset</p>', unsafe_allow_html=True)
    
    # Info section with icons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🔍</div>
            <h4>Auto Detection</h4>
            <p style="color: #4a5568; font-size: 0.9rem;">Automatically identifies numerical and categorical columns</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🧹</div>
            <h4>Data Cleaning</h4>
            <p style="color: #4a5568; font-size: 0.9rem;">Handles missing values and prepares data for analysis</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">⚡</div>
            <h4>Fast Processing</h4>
            <p style="color: #4a5568; font-size: 0.9rem;">Optimized for quick data transformation</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "📁 Drag and drop your CSV file here",
        type=['csv'],
        help="Upload a CSV file with your data"
    )
    
    # Sample dataset option
    st.markdown("#### 📋 Or try with sample data")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📋 Use Sample Dataset (Customer)", use_container_width=True):
            # Create sample customer dataset
            sample_df = create_sample_customer_dataset()
            upload_dataset(sample_df, "sample_customer.csv")
    
    with col2:
        if st.button("📋 Use Sample Dataset (Transactions)", use_container_width=True):
            # Create sample transaction dataset
            sample_df = create_sample_transaction_dataset()
            upload_dataset(sample_df, "sample_transactions.csv")
    
    if uploaded_file is not None:
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ File loaded: {uploaded_file.name}")
            
            # Display preview
            st.markdown("#### Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Upload button
            if st.button("🚀 Upload to Backend", type="primary", use_container_width=True):
                upload_dataset(df, uploaded_file.name)
        
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
    
    # Display dataset info if loaded
    if st.session_state.dataset_loaded and st.session_state.dataset_info:
        st.markdown("---")
        display_dataset_info(st.session_state.dataset_info)


def upload_dataset(df: pd.DataFrame, file_name: str):
    """Upload dataset to backend."""
    with st.spinner("Uploading dataset..."):
        try:
            result = st.session_state.api_client.upload_dataset(df, file_name)
            
            if result.get('success'):
                st.session_state.dataset_loaded = True
                st.session_state.dataset_info = result
                st.session_state.clustering_done = False
                st.success(f"✅ {result.get('message')}")
                st.rerun()
            else:
                st.error(f"❌ Upload failed: {result.get('message')}")
        
        except Exception as e:
            st.error(f"❌ Upload failed: {str(e)}")


def display_dataset_info(info: Dict):
    """Display dataset information."""
    st.markdown("#### 📋 Dataset Information")
    
    # Main metrics in a highlighted box
    st.markdown("""
    <div style="background: linear-gradient(135deg, #e8f5e9 0%, #f1f8e9 100%); padding: 1rem; border-radius: 12px; border-left: 4px solid #4ECDC4; margin-bottom: 1rem;">
        <p style="margin: 0; color: #2e7d32; font-weight: 600;">✅ Dataset loaded successfully!</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Rows", info['n_rows'])
    with col2:
        st.metric("📋 Columns", info['n_columns'])
    with col3:
        st.metric("🔢 Numerical", len(info['numerical_columns']))
    with col4:
        st.metric("🏷️ Categorical", len(info['categorical_columns']))
    
    # Column details
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🔢 Numerical Columns:**")
        if info['numerical_columns']:
            st.info(", ".join(info['numerical_columns']))
        else:
            st.write("None")
    
    with col2:
        st.markdown("**🏷️ Categorical Columns:**")
        if info['categorical_columns']:
            st.info(", ".join(info['categorical_columns']))
        else:
            st.write("None")
    
    # Missing values
    missing = info.get('missing_values', {})
    if any(v > 0 for v in missing.values()):
        st.markdown("**Missing Values:**")
        missing_df = pd.DataFrame([
            {"Column": k, "Missing": v}
            for k, v in missing.items() if v > 0
        ])
        st.dataframe(missing_df, use_container_width=True)


def render_maple_page():
    """Render MaPle execution page."""
    st.markdown('<p class="sub-header">⚙️ Run MaPle Clustering</p>', unsafe_allow_html=True)
    
    if not st.session_state.dataset_loaded:
        st.warning("⚠️ Please upload a dataset first!")
        st.markdown("""
        <div style="text-align: center; padding: 3rem; background: linear-gradient(145deg, #fff5f5, #ffffff); border-radius: 16px; margin-top: 1rem;">
            <div style="font-size: 4rem; margin-bottom: 1rem;">📤</div>
            <h3 style="color: #c53030;">No Dataset Loaded</h3>
            <p style="color: #4a5568;">Go to the Upload Dataset page to get started</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Parameter explanation cards
    st.markdown("### 🎯 Configure Algorithm Parameters")
    
    with st.expander("ℹ️ **What do these parameters mean?**", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Min Support** 📊
            - Controls how frequent a pattern must be
            - Lower values = more patterns discovered
            - Recommended: 0.03 - 0.10
            
            **Max Pattern Length** 📏
            - Maximum number of items in a pattern
            - Higher values = more complex patterns
            - Recommended: 5 - 10
            """)
        with col2:
            st.markdown("""
            **Number of Bins** 🎯
            - How to discretize numerical features
            - More bins = finer granularity
            - Recommended: 3 - 7
            
            **Assignment Strategy** 🎪
            - How transactions are assigned to clusters
            - `best_match`: One cluster per transaction
            - `threshold`: Multiple if similarity is high
            """)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Parameter configuration
    with st.form("maple_params"):
        col1, col2 = st.columns(2)
        
        with col1:
            min_support = st.slider(
                "Minimum Support",
                min_value=0.01,
                max_value=0.5,
                value=0.05,
                step=0.01,
                help="Minimum support threshold for frequent patterns"
            )
            
            max_pattern_length = st.slider(
                "Max Pattern Length",
                min_value=2,
                max_value=15,
                value=10,
                help="Maximum length of patterns to mine"
            )
            
            n_bins = st.slider(
                "Number of Bins",
                min_value=2,
                max_value=10,
                value=5,
                help="Number of bins for discretization"
            )
        
        with col2:
            discretization_strategy = st.selectbox(
                "Discretization Strategy",
                options=['quantile', 'uniform', 'kmeans'],
                help="Strategy for binning numerical features"
            )
            
            assignment_strategy = st.selectbox(
                "Assignment Strategy",
                options=['best_match', 'all_matching', 'threshold'],
                help="How to assign transactions to clusters"
            )
            
            overlap_threshold = st.slider(
                "Overlap Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.1,
                help="Threshold for cluster assignment"
            )
        
        # Submit button
        submitted = st.form_submit_button(
            "🚀 Run MaPle Algorithm",
            type="primary",
            use_container_width=True
        )
        
        if submitted:
            run_maple_clustering(
                min_support=min_support,
                max_pattern_length=max_pattern_length,
                n_bins=n_bins,
                discretization_strategy=discretization_strategy,
                assignment_strategy=assignment_strategy,
                overlap_threshold=overlap_threshold
            )


def run_maple_clustering(**params):
    """Execute MaPle clustering."""
    with st.spinner("Running MaPle clustering... This may take a few moments."):
        try:
            start_time = time.time()
            
            # Call API
            result = st.session_state.api_client.run_maple(**params)
            
            execution_time = time.time() - start_time
            
            if result.get('success'):
                st.session_state.clustering_done = True
                st.session_state.clustering_results = result
                
                st.success(f"✅ Clustering complete in {execution_time:.2f} seconds!")
                
                # Display quick summary
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Clusters", result['n_clusters'])
                with col2:
                    st.metric("Outliers", result['n_outliers'])
                with col3:
                    st.metric("Patterns", result['n_maximal_patterns'])
                with col4:
                    st.metric("Avg Size", f"{result['avg_cluster_size']:.1f}")
                
                st.info("📊 Go to 'Results & Evaluation' page to see detailed results!")
            else:
                st.error(f"❌ Clustering failed: {result.get('message')}")
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")


def render_results_page():
    """Render results and evaluation page."""
    st.markdown('<p class="sub-header">📊 Results & Evaluation</p>', unsafe_allow_html=True)
    
    if not st.session_state.clustering_done:
        st.warning("⚠️ Please run MaPle clustering first!")
        return
    
    results = st.session_state.clustering_results
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Overview",
        "🔍 Patterns",
        "🎯 Clusters",
        "📊 Evaluation"
    ])
    
    with tab1:
        render_overview_tab(results)
    
    with tab2:
        render_patterns_tab(results)
    
    with tab3:
        render_clusters_tab(results)
    
    with tab4:
        render_evaluation_tab()


def render_overview_tab(results: Dict):
    """Render overview tab."""
    st.markdown("### 📈 Clustering Summary")
    
    # Hero metrics in a highlighted section
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); padding: 2rem; border-radius: 20px; margin-bottom: 2rem;">
        <h3 style="color: #81e6d9; text-align: center; margin-bottom: 1.5rem;">🎯 Key Results</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Key metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("🎪 Clusters", results['n_clusters'])
    with col2:
        st.metric("🚫 Outliers", results['n_outliers'])
    with col3:
        st.metric("📊 Outlier %", f"{results['outlier_percentage']:.1f}%")
    with col4:
        st.metric("🔗 Frequent Patterns", results['n_frequent_patterns'])
    with col5:
        st.metric("⭐ Maximal Patterns", results['n_maximal_patterns'])
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Avg Cluster Size", f"{results['avg_cluster_size']:.1f}")
    with col2:
        st.metric("Min Cluster Size", results['min_cluster_size'])
    with col3:
        st.metric("Max Cluster Size", results['max_cluster_size'])
    
    # Cluster size distribution
    st.markdown("### 📊 Cluster Size Distribution")
    
    cluster_sizes = [c['size'] for c in results['clusters']]
    
    if cluster_sizes:
        # Create a colorful gradient for bars
        colors = px.colors.sequential.Tealgrn[::-1]
        
        fig = go.Figure(data=[go.Bar(
            x=[f"Cluster {c['cluster_id']}" for c in results['clusters']],
            y=cluster_sizes,
            marker=dict(
                color=cluster_sizes,
                colorscale='Tealgrn',
                showscale=True,
                colorbar=dict(title="Size")
            ),
            text=cluster_sizes,
            textposition='outside',
            textfont=dict(size=14, color='#1a202c')
        )])
        
        fig.update_layout(
            title=dict(
                text="<b>Cluster Sizes</b>",
                font=dict(size=18, color='#1a202c')
            ),
            xaxis_title="Cluster",
            yaxis_title="Number of Members",
            height=450,
            plot_bgcolor='rgba(255,255,255,1)',
            paper_bgcolor='rgba(255,255,255,1)',
            font=dict(family="Inter, sans-serif", color='#2d3748', size=12),
            xaxis=dict(
                showgrid=False,
                tickfont=dict(color='#2d3748', size=11),
                titlefont=dict(color='#1a202c', size=13)
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='rgba(0,0,0,0.1)',
                tickfont=dict(color='#2d3748', size=11),
                titlefont=dict(color='#1a202c', size=13)
            ),
            margin=dict(l=60, r=20, t=60, b=80),
            hoverlabel=dict(
                bgcolor="white",
                font_size=13,
                font_family="Inter, sans-serif",
                font_color="#1a202c"
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)


def render_patterns_tab(results: Dict):
    """Render patterns tab."""
    st.markdown("### ⭐ Top Maximal Patterns")
    
    patterns = results.get('top_patterns', [])
    
    if not patterns:
        st.info("No patterns found.")
        return
    
    # Display patterns in a styled table
    pattern_data = []
    for i, pattern in enumerate(patterns[:20], 1):
        pattern_data.append({
            "🏆 Rank": i,
            "🔗 Pattern": ", ".join(pattern['pattern']),
            "📊 Support": pattern['support'],
            "📈 Support %": f"{pattern['support_percentage']:.2f}%",
            "📏 Length": len(pattern['pattern'])
        })
    
    pattern_df = pd.DataFrame(pattern_data)
    st.dataframe(
        pattern_df, 
        use_container_width=True, 
        hide_index=True,
        column_config={
            "🏆 Rank": st.column_config.NumberColumn(width="small"),
            "📊 Support": st.column_config.NumberColumn(width="small"),
            "📏 Length": st.column_config.NumberColumn(width="small"),
        }
    )
    
    # Pattern length distribution
    st.markdown("### 📏 Pattern Length Distribution")
    
    patterns_by_length = results.get('patterns_by_length', {})
    
    if patterns_by_length:
        lengths = list(patterns_by_length.keys())
        counts = list(patterns_by_length.values())
        
        fig = go.Figure(data=[go.Bar(
            x=lengths,
            y=counts,
            marker=dict(
                color=counts,
                colorscale='Reds',
                showscale=False
            ),
            text=counts,
            textposition='outside',
            textfont=dict(size=14, color='#1a202c')
        )])
        
        fig.update_layout(
            title=dict(
                text="<b>Frequent Patterns by Length</b>",
                font=dict(size=18, color='#1a202c')
            ),
            xaxis_title="Pattern Length",
            yaxis_title="Number of Patterns",
            height=400,
            plot_bgcolor='rgba(255,255,255,1)',
            paper_bgcolor='rgba(255,255,255,1)',
            font=dict(family="Inter, sans-serif", color='#2d3748', size=12),
            xaxis=dict(
                showgrid=False,
                tickfont=dict(color='#2d3748', size=11),
                titlefont=dict(color='#1a202c', size=13)
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='rgba(0,0,0,0.1)',
                tickfont=dict(color='#2d3748', size=11),
                titlefont=dict(color='#1a202c', size=13)
            ),
            margin=dict(l=60, r=20, t=60, b=60),
            hoverlabel=dict(
                bgcolor="white",
                font_size=13,
                font_family="Inter, sans-serif",
                font_color="#1a202c"
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)


def render_clusters_tab(results: Dict):
    """Render clusters tab."""
    st.markdown("### 🎯 Cluster Details")
    
    clusters = results.get('clusters', [])
    
    if not clusters:
        st.info("No clusters found.")
        return
    
    # Cluster selection with better styling
    cluster_options = {
        f"🎪 Cluster {c['cluster_id']} — {c['size']} members": c
        for c in clusters
    }
    
    st.markdown("**🔍 Select a cluster to explore:**")
    selected_cluster_name = st.selectbox(
        "Cluster Selection",
        options=list(cluster_options.keys()),
        label_visibility="collapsed"
    )
    
    selected_cluster = cluster_options[selected_cluster_name]
    
    # Display cluster details in a nice card
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%); padding: 1.5rem; border-radius: 16px; border-left: 5px solid #4ECDC4; margin: 1rem 0;">
        <h4 style="color: #1a202c; margin-bottom: 1rem;">📋 Cluster {selected_cluster['cluster_id']} Summary</h4>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🆔 Cluster ID", selected_cluster['cluster_id'])
    with col2:
        st.metric("👥 Members", selected_cluster['size'])
    with col3:
        st.metric("💪 Pattern Support", selected_cluster['pattern_support'])
    
    st.markdown("**🔗 Defining Pattern:**")
    pattern_text = ", ".join(selected_cluster['pattern'])
    st.success(f"📌 {pattern_text}")
    
    with st.expander("👀 View Member IDs (first 50)"):
        members = selected_cluster['member_ids'][:50]
        st.write(members)
    
    # All clusters summary
    st.markdown("---")
    st.markdown("### 📊 All Clusters Overview")
    
    cluster_data = []
    for cluster in clusters:
        cluster_data.append({
            "🆔 ID": cluster['cluster_id'],
            "👥 Size": cluster['size'],
            "📏 Pattern Len": cluster['pattern_length'],
            "💪 Support": cluster['pattern_support'],
            "🔗 Pattern Preview": ", ".join(cluster['pattern'][:3]) + ("..." if len(cluster['pattern']) > 3 else "")
        })
    
    cluster_df = pd.DataFrame(cluster_data)
    st.dataframe(cluster_df, use_container_width=True, hide_index=True)


def render_evaluation_tab():
    """Render evaluation tab."""
    st.markdown("### 🔬 Clustering Evaluation")
    
    st.markdown("""
    <div style="background: linear-gradient(145deg, #fff5f5, #ffffff); padding: 1rem; border-radius: 12px; margin-bottom: 1rem;">
        <p style="margin: 0; color: #4a5568;">📊 Evaluate your clustering quality using industry-standard metrics</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🔬 Run Evaluation", type="primary", use_container_width=True):
        evaluate_clustering()
    
    # Display evaluation results if available
    if st.session_state.evaluation_results:
        display_evaluation_results(st.session_state.evaluation_results)


def evaluate_clustering():
    """Evaluate clustering quality."""
    with st.spinner("Evaluating clustering..."):
        try:
            result = st.session_state.api_client.evaluate_clustering(
                include_visualization=True
            )
            
            if result.get('success'):
                st.session_state.evaluation_results = result
                st.success("✅ Evaluation complete!")
                st.rerun()
            else:
                st.error(f"❌ Evaluation failed: {result.get('message')}")
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")


def display_evaluation_results(results: Dict):
    """Display evaluation results."""
    metrics = results.get('metrics', {})
    
    st.markdown("#### 📊 Clustering Quality Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        silhouette = metrics.get('silhouette_score')
        if silhouette is not None:
            st.metric("🎯 Silhouette Score", f"{silhouette:.4f}")
            if silhouette > 0.5:
                st.success("✨ Excellent clustering quality")
            elif silhouette > 0.25:
                st.info("👍 Good clustering quality")
            else:
                st.warning("⚠️ Consider adjusting parameters")
        else:
            st.metric("🎯 Silhouette Score", "N/A")
    
    with col2:
        coverage = metrics.get('pattern_coverage', 0)
        st.metric("📈 Pattern Coverage", f"{coverage:.1f}%")
    
    with col3:
        intra_sim = metrics.get('avg_intra_cluster_similarity')
        if intra_sim is not None:
            st.metric("🔗 Intra-Cluster Similarity", f"{intra_sim:.4f}")
        else:
            st.metric("Avg Intra-Cluster Similarity", "N/A")
    
    # Cluster distribution
    st.markdown("#### 🥧 Cluster Distribution")
    
    distribution = metrics.get('cluster_distribution', {})
    
    if distribution:
        labels = list(distribution.keys())
        values = list(distribution.values())
        
        # Beautiful donut chart with custom colors
        colors = ['#4ECDC4', '#FF6B6B', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9']
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.4,
            marker=dict(colors=colors[:len(labels)]),
            textinfo='label+percent',
            textfont=dict(size=13, color='#1a202c'),
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
        )])
        
        fig.update_layout(
            title=dict(
                text="<b>Transaction Distribution Across Clusters</b>",
                font=dict(size=16, color='#1a202c')
            ),
            height=450,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5,
                font=dict(color='#2d3748', size=11)
            ),
            paper_bgcolor='rgba(255,255,255,1)',
            font=dict(color='#2d3748'),
            annotations=[dict(
                text='<b>Clusters</b>',
                x=0.5, y=0.5,
                font=dict(size=14, color='#2d3748'),
                showarrow=False
            )],
            margin=dict(l=20, r=20, t=60, b=100),
            hoverlabel=dict(
                bgcolor="white",
                font_size=13,
                font_family="Inter, sans-serif",
                font_color="#1a202c"
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Visualization
    viz_data = results.get('visualization_data')
    
    if viz_data:
        st.markdown("#### 🎨 2D Cluster Visualization (PCA)")
        
        # Create custom color sequence
        custom_colors = ['#4ECDC4', '#FF6B6B', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9']
        
        fig = px.scatter(
            x=viz_data['x'],
            y=viz_data['y'],
            color=[str(label) for label in viz_data['cluster_labels']],
            labels={'x': 'Principal Component 1', 'y': 'Principal Component 2', 'color': 'Cluster'},
            title=f"<b>Clusters in 2D Space</b><br><sub>Explained Variance: {sum(viz_data['explained_variance']):.1%}</sub>",
            color_discrete_sequence=custom_colors
        )
        
        fig.update_traces(
            marker=dict(size=10, opacity=0.7, line=dict(width=1, color='white'))
        )
        
        fig.update_layout(
            height=550,
            plot_bgcolor='rgba(255,255,255,1)',
            paper_bgcolor='rgba(255,255,255,1)',
            font=dict(family="Inter, sans-serif", color='#2d3748', size=12),
            xaxis=dict(
                showgrid=True,
                gridcolor='rgba(0,0,0,0.1)',
                zeroline=False,
                tickfont=dict(color='#2d3748', size=11),
                titlefont=dict(color='#1a202c', size=13)
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='rgba(0,0,0,0.1)',
                zeroline=False,
                tickfont=dict(color='#2d3748', size=11),
                titlefont=dict(color='#1a202c', size=13)
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5,
                font=dict(color='#2d3748', size=11)
            ),
            title=dict(
                font=dict(color='#1a202c')
            ),
            margin=dict(l=60, r=20, t=80, b=100),
            hoverlabel=dict(
                bgcolor="white",
                font_size=13,
                font_family="Inter, sans-serif",
                font_color="#1a202c"
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)


def render_about_page():
    """Render about page."""
    st.markdown('<p class="sub-header">ℹ️ About MaPle</p>', unsafe_allow_html=True)
    
    # Hero section
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); padding: 3rem; border-radius: 20px; text-align: center; margin-bottom: 2rem;">
        <h1 style="color: #81e6d9; margin-bottom: 0.5rem;">🍁 MaPle</h1>
        <h3 style="color: #ffffff; font-weight: 400;">Maximal Pattern-based Clustering</h3>
        <p style="color: #e2e8f0; margin-top: 1rem;">Discover hidden patterns in your data with interpretable clustering</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Features in cards
    st.markdown("### ✨ Key Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🔍</div>
            <h4>Pattern-based</h4>
            <p style="color: #4a5568; font-size: 0.9rem;">Groups items with similar behavioral patterns instead of distances</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🎯</div>
            <h4>Auto Discovery</h4>
            <p style="color: #4a5568; font-size: 0.9rem;">No need to specify the number of clusters in advance</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">💡</div>
            <h4>Interpretable</h4>
            <p style="color: #4a5568; font-size: 0.9rem;">Clusters defined by clear, human-readable patterns</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🚫</div>
            <h4>Outlier Detection</h4>
            <p style="color: #4a5568; font-size: 0.9rem;">Explicit identification of anomalous transactions</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🔄</div>
            <h4>Mixed Data</h4>
            <p style="color: #4a5568; font-size: 0.9rem;">Handles both numerical and categorical features</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">⚡</div>
            <h4>Fast & Efficient</h4>
            <p style="color: #4a5568; font-size: 0.9rem;">Optimized algorithms for quick processing</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Algorithm explanation
    with st.expander("🔬 **Algorithm Overview**", expanded=True):
        st.markdown("""
        | Step | Process | Description |
        |------|---------|-------------|
        | 1️⃣ | **Preprocessing** | Handle missing values and normalize features |
        | 2️⃣ | **Discretization** | Convert continuous features to categorical bins |
        | 3️⃣ | **Pattern Mining** | Discover frequent itemsets using Apriori algorithm |
        | 4️⃣ | **Maximal Extraction** | Filter to keep only maximal patterns |
        | 5️⃣ | **Cluster Formation** | Create clusters from maximal patterns |
        | 6️⃣ | **Assignment** | Assign transactions to best-matching clusters |
        | 7️⃣ | **Evaluation** | Assess clustering quality with multiple metrics |
        """)
    
    # Use cases
    with st.expander("🎯 **When to Use MaPle**"):
        st.markdown("""
        MaPle is particularly effective for:
        - 🛒 **Market Basket Analysis** - Understanding purchasing patterns
        - 👥 **Customer Segmentation** - Grouping customers by behavior
        - 💳 **Transaction Clustering** - Finding similar transactions
        - 📊 **Mixed-type Datasets** - Handling numerical and categorical data
        - 🔍 **Interpretable Results** - When you need to explain clusters
        """)
    
    # Tech stack
    with st.expander("🛠️ **Technology Stack**"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Backend:**
            - 🐍 Python 3.10+
            - ⚡ FastAPI
            - 📊 NumPy & Pandas
            - 🔬 Scikit-learn (evaluation)
            """)
        
        with col2:
            st.markdown("""
            **Frontend:**
            - 🎨 Streamlit
            - 📈 Plotly Visualizations
            - 🎭 Custom CSS Styling
            """)
    
    # Footer
    st.markdown("""
    <div style="text-align: center; padding: 2rem; margin-top: 2rem; border-top: 1px solid #eee;">
        <p style="color: #4a5568; margin-bottom: 0.5rem;">📚 Course: Data Warehouse and Data Mining</p>
        <p style="color: #718096; font-size: 0.9rem;">Built with ❤️ using Python • Version 1.0.0</p>
    </div>
    """, unsafe_allow_html=True)


def create_sample_customer_dataset() -> pd.DataFrame:
    """Create sample customer dataset."""
    np.random.seed(42)
    
    n_samples = 200
    
    # Define categorical options
    genders = ['Male', 'Female', 'Other']
    membership_tiers = ['Bronze', 'Silver', 'Gold', 'Platinum']
    cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia']
    preferred_channels = ['Online', 'In-Store', 'Mobile App', 'Phone']
    
    data = {
        'CustomerID': range(1, n_samples + 1),
        'Age': np.random.randint(18, 70, n_samples),
        'Gender': np.random.choice(genders, n_samples),
        'Income': np.random.randint(20000, 150000, n_samples),
        'SpendingScore': np.random.randint(1, 100, n_samples),
        'Purchases': np.random.randint(1, 50, n_samples),
        'MembershipYears': np.random.randint(0, 15, n_samples),
        'MembershipTier': np.random.choice(membership_tiers, n_samples),
        'City': np.random.choice(cities, n_samples),
        'PreferredChannel': np.random.choice(preferred_channels, n_samples)
    }
    
    return pd.DataFrame(data)


def create_sample_transaction_dataset() -> pd.DataFrame:
    """Create sample transaction dataset."""
    np.random.seed(42)
    
    n_samples = 150
    
    # Define categorical options
    payment_methods = ['Credit Card', 'Debit Card', 'Cash', 'Mobile Payment', 'Digital Wallet']
    product_categories = ['Electronics', 'Clothing', 'Food', 'Home & Garden', 'Sports', 'Books']
    transaction_types = ['Purchase', 'Return', 'Exchange']
    store_locations = ['Store A', 'Store B', 'Store C', 'Store D', 'Online']
    
    data = {
        'TransactionID': range(1, n_samples + 1),
        'Amount': np.random.uniform(10, 500, n_samples),
        'Quantity': np.random.randint(1, 20, n_samples),
        'DayOfWeek': np.random.randint(0, 7, n_samples),
        'Hour': np.random.randint(8, 22, n_samples),
        'CustomerAge': np.random.randint(18, 70, n_samples),
        'PaymentMethod': np.random.choice(payment_methods, n_samples),
        'ProductCategory': np.random.choice(product_categories, n_samples),
        'TransactionType': np.random.choice(transaction_types, n_samples, p=[0.85, 0.10, 0.05]),
        'StoreLocation': np.random.choice(store_locations, n_samples)
    }
    
    return pd.DataFrame(data)


if __name__ == "__main__":
    main()
