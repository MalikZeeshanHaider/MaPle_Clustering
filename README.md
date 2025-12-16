# 🍁 MaPle: Maximal Pattern-based Clustering

**A Full-Stack Data Mining Project**  
Course: Data Warehouse and Data Mining  
Algorithm: Custom Pattern-based Clustering Implementation

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Algorithm Explanation](#algorithm-explanation)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [API Documentation](#api-documentation)
- [Screenshots](#screenshots)
- [Evaluation Metrics](#evaluation-metrics)
- [Academic References](#academic-references)
- [Future Enhancements](#future-enhancements)
- [License](#license)

---

## 🎯 Overview

**MaPle (Maximal Pattern-based Clustering)** is a novel clustering algorithm that groups data based on **maximal frequent patterns** rather than traditional distance-based measures. This project implements MaPle from scratch with a complete full-stack solution featuring:

- ✅ **Backend API**: FastAPI-based REST API with enterprise-grade architecture
- ✅ **Frontend UI**: Interactive Streamlit web application
- ✅ **Custom Algorithm**: No external clustering libraries (KMeans, DBSCAN, etc.)
- ✅ **Complete Pipeline**: Data preprocessing, pattern mining, clustering, and evaluation
- ✅ **Comprehensive Documentation**: Academic-level explanations and code documentation

### Why MaPle?

Traditional clustering algorithms like K-Means work well for numerical data with spherical clusters, but struggle with:
- Categorical data
- Mixed-type datasets
- Non-geometric relationships
- Interpretability requirements

MaPle addresses these limitations by identifying **co-occurrence patterns** in data and forming clusters based on shared patterns.

---

## ✨ Features

### Core Algorithm
- 🔍 **Frequent Pattern Mining**: Modified Apriori algorithm implementation
- 🎯 **Maximal Pattern Extraction**: Filters redundant patterns
- 📊 **Pattern-based Clustering**: Forms clusters from maximal patterns
- 🚨 **Outlier Detection**: Explicitly identifies transactions not matching patterns
- 🔄 **Automatic Cluster Count**: No need to specify k

### Data Processing
- 📥 **CSV Import**: Upload datasets via web interface
- 🧹 **Preprocessing**: Automatic missing value handling
- 🔢 **Discretization**: Multiple binning strategies (uniform, quantile, k-means)
- 🔀 **Mixed Data Types**: Handles numerical and categorical features

### Evaluation
- 📈 **Silhouette Score**: Cluster cohesion and separation
- 🎯 **Cluster Purity**: If ground truth labels available
- 📊 **Pattern Coverage**: Percentage of data covered
- 📉 **Davies-Bouldin Index**: Alternative quality metric
- 📐 **Calinski-Harabasz Score**: Variance-based evaluation

### User Interface
- 🖥️ **Interactive Dashboard**: Streamlit-based web interface
- 📊 **Visualizations**: 2D cluster plots using PCA
- 🎛️ **Parameter Tuning**: Real-time parameter adjustment
- 📋 **Sample Datasets**: Built-in example datasets
- 💾 **Result Export**: Download clustering results

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                            │
│                     (Streamlit Frontend)                         │
│   📤 Upload  ⚙️ Configure  🚀 Execute  📊 Visualize            │
└───────────────────────────┬──────────────────────────────────────┘
                            │ HTTP/REST
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│                      REST API LAYER                              │
│                     (FastAPI Backend)                            │
│   /upload-dataset  /run-maple  /evaluate  /get-clusters         │
└───────────────────────────┬──────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│                     SERVICE LAYER                                │
│   DataService │ MapleService │ EvaluationService                │
└───────────────────────────┬──────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│                    ALGORITHM LAYER                               │
│   DataDiscretizer │ FrequentPatternMiner │ MaPleClusterer       │
└───────────────────────────┬──────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│                       DATA LAYER                                 │
│              Pandas DataFrames │ NumPy Arrays                    │
└──────────────────────────────────────────────────────────────────┘
```

---

## 🧮 Algorithm Explanation

### Step-by-Step Process

#### 1. Data Preprocessing
```python
# Handle missing values
# Normalize numerical features
# Identify feature types
```

#### 2. Discretization
Convert continuous numerical features into categorical bins:

```
Example:
Age: 25 → "Young" (0-30)
Income: 75000 → "High" (60k+)
```

Supported strategies:
- **Uniform**: Equal-width bins
- **Quantile**: Equal-frequency bins
- **K-Means**: Cluster-based boundaries

#### 3. Transaction Transformation
Convert each row to a set of items:

```
Row: {Age: 25, Income: 75000}
  ↓
Transaction: {"Age=Young", "Income=High"}
```

#### 4. Frequent Pattern Mining (Apriori)

Find all itemsets with support ≥ minimum threshold:

```python
# Pseudo-code
L1 = find_frequent_1_itemsets()  # Single items
for k = 2 to max_length:
    Ck = generate_candidates(Lk-1)  # Candidate generation
    Lk = filter_by_support(Ck)      # Support counting
    if Lk is empty: break
```

**Key Property**: All subsets of a frequent itemset are also frequent (Apriori property)

#### 5. Maximal Pattern Extraction

A pattern is **maximal** if no superset is frequent:

```
Frequent: {A,B}, {A,C}, {B,C}, {A,B,C}
Maximal: {A,B,C} only
```

This reduces redundancy and focuses on most informative patterns.

#### 6. Cluster Formation

Each maximal pattern becomes a cluster seed:

```
Pattern: {"Age=Young", "Income=High"}
  → Cluster 1: All transactions containing these items
```

#### 7. Transaction Assignment

Assign each transaction to best-matching cluster:

```python
for each transaction T:
    best_cluster = None
    best_score = 0
    
    for each cluster C with pattern P:
        if P ⊆ T:
            score = |P ∩ T| / |T|
            if score > best_score:
                best_score = score
                best_cluster = C
    
    assign T to best_cluster (or mark as outlier)
```

### Mathematical Formulation

**Support**: Frequency of pattern occurrence
```
Support(X) = |{T ∈ D | X ⊆ T}| / |D|
```

**Jaccard Similarity**: For pattern matching
```
Jaccard(T, P) = |T ∩ P| / |T ∪ P|
```

**Cluster Quality**: Maximize intra-cluster similarity
```
Minimize: Σᵢ Σ_{T∈Cᵢ} distance(T, Pᵢ)
```

### Complexity Analysis

- **Time**: O(2^|I| × n) worst case, where |I| = unique items, n = transactions
- **Space**: O(|F|) for storing frequent patterns
- **Practical**: Much faster than worst case due to pruning

---

## 🚀 Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager
- 4GB RAM minimum (8GB recommended)

### Step 1: Clone Repository

```bash
git clone <repository-url>
cd Data_Mining
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment

```bash
# Copy example environment file
copy .env.example .env

# Edit .env if needed (default settings work fine)
```

---

## 💻 Usage

### Running the Application

#### 1. Start Backend API

Open a terminal and run:

```bash
python -m backend.main
```

The API will start at `http://localhost:8000`

You can access the API documentation at:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

#### 2. Start Frontend Application

Open a **new terminal** (keep backend running) and run:

```bash
streamlit run frontend/app.py
```

The frontend will open in your browser at `http://localhost:8501`

### Using the Application

#### Step 1: Upload Dataset
1. Go to "📤 Upload Dataset" page
2. Either:
   - Upload your own CSV file, or
   - Click "Use Sample Dataset" button
3. Review dataset information

#### Step 2: Configure Parameters
1. Go to "⚙️ Run MaPle" page
2. Adjust parameters:
   - **Min Support**: 0.01-0.50 (lower = more patterns)
   - **Max Pattern Length**: 2-15 (higher = longer patterns)
   - **Number of Bins**: 2-10 (higher = finer discretization)
   - **Discretization Strategy**: uniform, quantile, or kmeans
   - **Assignment Strategy**: best_match, all_matching, or threshold

3. Click "🚀 Run MaPle Algorithm"

#### Step 3: View Results
1. Go to "📊 Results & Evaluation" page
2. Explore tabs:
   - **Overview**: Cluster summary and sizes
   - **Patterns**: Discovered maximal patterns
   - **Clusters**: Detailed cluster information
   - **Evaluation**: Quality metrics and visualizations

---

## 📁 Project Structure

```
Data_Mining/
├── backend/                      # FastAPI Backend
│   ├── algorithms/               # Core algorithms
│   │   ├── discretization.py    # Data discretization
│   │   ├── pattern_mining.py    # Frequent pattern mining
│   │   └── maple_clustering.py  # MaPle clustering
│   ├── services/                 # Business logic
│   │   ├── data_service.py      # Data handling
│   │   ├── maple_service.py     # Clustering orchestration
│   │   └── evaluation_service.py # Evaluation metrics
│   ├── routes/                   # API endpoints
│   │   └── api_routes.py        # REST routes
│   ├── models/                   # Pydantic models
│   │   ├── requests.py          # Request schemas
│   │   └── responses.py         # Response schemas
│   ├── utils/                    # Utilities
│   │   ├── logger.py            # Logging setup
│   │   └── exceptions.py        # Custom exceptions
│   ├── config.py                 # Configuration
│   └── main.py                   # Application entry
│
├── frontend/                     # Streamlit Frontend
│   ├── components/               # UI components (future)
│   ├── utils/
│   │   └── api_client.py        # Backend API client
│   └── app.py                    # Main Streamlit app
│
├── data/                         # Sample datasets
│   ├── sample_customer.csv
│   └── sample_transactions.csv
│
├── docs/                         # Documentation
│   ├── ALGORITHM_EXPLANATION.md # Detailed algorithm theory
│   ├── API_DOCUMENTATION.md     # API reference (to be created)
│   └── USER_GUIDE.md            # User manual (to be created)
│
├── tests/                        # Unit tests (future)
│
├── .env.example                  # Environment variables template
├── .gitignore                    # Git ignore file
├── requirements.txt              # Python dependencies
├── README.md                     # This file
└── PROJECT_OVERVIEW.md          # Academic project overview
```

---

## 📚 API Documentation

### Base URL
```
http://localhost:8000/api/v1
```

### Endpoints

#### 1. Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "service": "MaPle Clustering API",
  "version": "1.0.0"
}
```

#### 2. Upload Dataset
```http
POST /upload-dataset
Content-Type: application/json

{
  "file_content": "csv_string_here",
  "file_name": "data.csv"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Dataset uploaded successfully",
  "n_rows": 100,
  "n_columns": 5,
  "numerical_columns": ["Age", "Income"],
  "categorical_columns": ["Gender"]
}
```

#### 3. Run MaPle
```http
POST /run-maple
Content-Type: application/json

{
  "min_support": 0.05,
  "max_pattern_length": 10,
  "n_bins": 5,
  "discretization_strategy": "quantile",
  "assignment_strategy": "best_match"
}
```

**Response:**
```json
{
  "success": true,
  "n_clusters": 5,
  "n_outliers": 10,
  "n_maximal_patterns": 8,
  "execution_time_seconds": 2.5,
  "clusters": [...],
  "top_patterns": [...]
}
```

#### 4. Get Clusters
```http
GET /get-clusters
```

#### 5. Evaluate Clustering
```http
POST /evaluate
Content-Type: application/json

{
  "include_visualization": true,
  "label_column": null
}
```

**Response:**
```json
{
  "success": true,
  "metrics": {
    "silhouette_score": 0.45,
    "pattern_coverage": 85.5,
    "cluster_distribution": {...}
  },
  "visualization_data": {...}
}
```

For complete API documentation, visit: http://localhost:8000/docs

---

## 📊 Evaluation Metrics

### Silhouette Score
- **Range**: -1 to 1
- **Interpretation**:
  - > 0.5: Excellent
  - 0.25 - 0.5: Good
  - < 0.25: Poor
- **Formula**: Measures cluster cohesion vs separation

### Pattern Coverage
- **Range**: 0% to 100%
- **Meaning**: Percentage of transactions matched by patterns
- **Goal**: Higher is better (fewer outliers)

### Cluster Purity (if labels available)
- **Range**: 0 to 1
- **Meaning**: Majority class percentage per cluster
- **Use**: Comparing with ground truth

### Davies-Bouldin Index
- **Range**: 0 to ∞
- **Interpretation**: Lower is better
- **Measures**: Average similarity between clusters

---

## 🎓 Academic References

### Core Papers

1. **Agrawal, R., & Srikant, R. (1994)**  
   "Fast Algorithms for Mining Association Rules"  
   *Proceedings of VLDB*

2. **Han, J., Pei, J., & Yin, Y. (2000)**  
   "Mining Frequent Patterns without Candidate Generation"  
   *ACM SIGMOD Record*

3. **Wang, K., Xu, C., & Liu, B. (1999)**  
   "Clustering Transactions Using Large Items"  
   *CIKM*

### Books

4. **Han, J., Kamber, M., & Pei, J. (2011)**  
   "Data Mining: Concepts and Techniques" (3rd ed.)  
   Morgan Kaufmann

5. **Tan, P., Steinbach, M., & Kumar, V. (2019)**  
   "Introduction to Data Mining" (2nd ed.)  
   Pearson

---

## 🔮 Future Enhancements

### Algorithm Improvements
- [ ] Implement FP-Growth for faster mining
- [ ] Add closed pattern mining
- [ ] Support hierarchical clustering
- [ ] Implement incremental pattern mining

### Features
- [ ] Export results to CSV/JSON
- [ ] Save/load clustering models
- [ ] Batch processing for multiple datasets
- [ ] Real-time pattern visualization
- [ ] Custom distance functions

### Performance
- [ ] Parallel pattern mining
- [ ] Database backend for large datasets
- [ ] Caching frequent patterns
- [ ] GPU acceleration

### User Experience
- [ ] More visualization types
- [ ] Interactive pattern exploration
- [ ] Cluster comparison tools
- [ ] Guided parameter selection

---

## 🧪 Testing

### Run Unit Tests (future)

```bash
pytest tests/ -v
```

### Test Coverage

```bash
pytest --cov=backend tests/
```

---

## 📝 Code Quality

This project follows enterprise-grade standards:

- ✅ **PEP 8**: Python style guide compliance
- ✅ **Type Hints**: Comprehensive typing throughout
- ✅ **Docstrings**: Google-style documentation
- ✅ **SOLID Principles**: Clean architecture
- ✅ **DRY**: Don't Repeat Yourself
- ✅ **Error Handling**: Comprehensive exception management
- ✅ **Logging**: Centralized logging system

### Code Formatting

```bash
# Format code with Black
black backend/ frontend/

# Lint with flake8
flake8 backend/ frontend/

# Type checking with mypy
mypy backend/
```

---

## 🤝 Contributing

This is an academic project, but suggestions are welcome:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

---

## 📄 License

This project is for **academic purposes only**.  
Created as a semester project for Data Warehouse and Data Mining course.

---

## 👨‍💻 Author

**Data Mining Course Project**  
Semester Project - 2025  
Algorithm: MaPle (Maximal Pattern-based Clustering)

---

## 🆘 Troubleshooting

### Backend Won't Start

**Error**: `ModuleNotFoundError: No module named 'fastapi'`

**Solution**:
```bash
pip install -r requirements.txt
```

### Frontend Can't Connect

**Error**: `Connection refused`

**Solution**: Ensure backend is running first:
```bash
python -m backend.main
```

### Out of Memory

**Issue**: Large dataset causes memory error

**Solution**:
- Reduce `max_pattern_length`
- Increase `min_support`
- Use a smaller sample of data

### No Patterns Found

**Issue**: `n_maximal_patterns: 0`

**Solution**:
- Lower `min_support` (try 0.02 or 0.03)
- Check if data has sufficient variation
- Reduce `n_bins` for discretization

---

## 📧 Support

For questions or issues:
1. Check documentation in `docs/` folder
2. Review API docs at http://localhost:8000/docs
3. Check troubleshooting section above

---

## 🙏 Acknowledgments

- Data Mining course instructors
- scikit-learn for evaluation metrics
- FastAPI and Streamlit communities
- Academic papers cited above

---

**Built with ❤️ for Data Mining Education**

*Last Updated: December 16, 2025*
