# MaPle: Algorithm for Maximal Pattern-based Clustering

## Project Overview

**Course**: Data Warehouse and Data Mining  
**Project Type**: Semester Project  
**Academic Level**: Graduate/Advanced Undergraduate  
**Implementation Language**: Python 3.10+

---

## 1. Executive Summary

This project implements **MaPle (Maximal Pattern-based Clustering)**, a novel data mining algorithm that discovers clusters based on maximal frequent patterns rather than traditional distance-based measures. Unlike conventional clustering algorithms (K-Means, DBSCAN), MaPle leverages association rule mining principles to identify meaningful groupings in transactional or categorical data.

The project delivers a full-stack enterprise-grade solution with:
- **Backend**: FastAPI-based REST API implementing core MaPle algorithm
- **Frontend**: Streamlit-based interactive web application
- **Algorithm**: Custom implementation of maximal pattern mining and pattern-based clustering
- **Evaluation**: Multiple metrics including Silhouette Score and cluster purity

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                          │
│                    (Streamlit Frontend)                         │
│  • Dataset Upload    • Parameter Config    • Visualization     │
└────────────────────────┬────────────────────────────────────────┘
                         │ HTTP Requests
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                       REST API LAYER                            │
│                      (FastAPI Backend)                          │
│  • /upload-dataset   • /run-maple   • /get-clusters            │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                      SERVICE LAYER                              │
│  • DataService  • MapleService  • EvaluationService            │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   ALGORITHM LAYER                               │
│  • Frequent Pattern Mining (Apriori-based)                     │
│  • Maximal Pattern Extraction                                  │
│  • Pattern-based Clustering (MaPle Core)                       │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                      DATA LAYER                                 │
│  • Data Preprocessing  • Feature Engineering                   │
│  • Normalization      • Discretization                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Technology Stack

### Backend
- **Framework**: FastAPI 0.104+
- **Data Processing**: Pandas, NumPy
- **Validation**: Pydantic
- **Evaluation**: Scikit-learn (metrics only)
- **Logging**: Python logging module
- **Configuration**: python-dotenv

### Frontend
- **Framework**: Streamlit 1.28+
- **Visualization**: Matplotlib, Seaborn, Plotly
- **HTTP Client**: Requests

### Development
- **Python Version**: 3.10+
- **Code Style**: PEP8, Black formatter
- **Type Hints**: Comprehensive typing
- **Documentation**: Docstrings (Google style)

---

## 4. Project Structure

```
Data_Mining/
├── backend/
│   ├── algorithms/
│   │   ├── __init__.py
│   │   ├── pattern_mining.py      # Frequent & maximal pattern mining
│   │   ├── maple_clustering.py    # Core MaPle algorithm
│   │   └── discretization.py      # Data transformation
│   ├── services/
│   │   ├── __init__.py
│   │   ├── data_service.py        # Data handling & preprocessing
│   │   ├── maple_service.py       # MaPle orchestration
│   │   └── evaluation_service.py  # Clustering evaluation
│   ├── routes/
│   │   ├── __init__.py
│   │   └── api_routes.py          # REST API endpoints
│   ├── models/
│   │   ├── __init__.py
│   │   ├── requests.py            # Pydantic request models
│   │   └── responses.py           # Pydantic response models
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── logger.py              # Logging configuration
│   │   └── exceptions.py          # Custom exceptions
│   ├── config.py                   # Configuration management
│   └── main.py                     # FastAPI application entry
├── frontend/
│   ├── app.py                      # Streamlit application
│   ├── components/
│   │   ├── __init__.py
│   │   ├── upload.py              # Dataset upload component
│   │   ├── parameters.py          # Parameter selection
│   │   └── visualization.py       # Chart components
│   └── utils/
│       ├── __init__.py
│       └── api_client.py          # Backend API client
├── data/
│   ├── sample_transactions.csv    # Sample transactional data
│   └── sample_customer.csv        # Sample customer data
├── docs/
│   ├── ALGORITHM_EXPLANATION.md   # Detailed algorithm theory
│   ├── API_DOCUMENTATION.md       # API reference
│   └── USER_GUIDE.md              # End-user instructions
├── tests/
│   ├── test_pattern_mining.py
│   ├── test_maple_clustering.py
│   └── test_api.py
├── .env.example                    # Environment variables template
├── .gitignore
├── requirements.txt                # Python dependencies
├── README.md                       # Main documentation
└── PROJECT_OVERVIEW.md            # This file
```

---

## 5. Core Concepts

### 5.1 Pattern-based Clustering
Traditional clustering algorithms (K-Means, Hierarchical) use geometric distances. MaPle uses **pattern similarity** - items that share frequent patterns belong to the same cluster.

### 5.2 Maximal Frequent Patterns
- **Frequent Pattern**: Itemset appearing in data with frequency ≥ minimum support
- **Maximal Pattern**: Frequent pattern with no frequent superset
- **Example**: If {A,B,C} is frequent but {A,B,C,D} is not, then {A,B,C} is maximal

### 5.3 MaPle Clustering Process
1. **Discretization**: Convert continuous features to categorical items
2. **Transaction Creation**: Transform data into transactional format
3. **Frequent Pattern Mining**: Discover all frequent itemsets
4. **Maximal Pattern Extraction**: Filter to keep only maximal patterns
5. **Cluster Formation**: Group transactions by shared maximal patterns
6. **Cluster Assignment**: Assign each transaction to best-matching cluster

---

## 6. Algorithm Workflow

```
Input Dataset (CSV)
    ↓
[Preprocessing]
    ├── Handle missing values
    ├── Normalize numerical features
    └── Discretize into bins
    ↓
[Transaction Transformation]
    ├── Convert each row to itemset
    └── Format: {Feature1=Value1, Feature2=Value2, ...}
    ↓
[Frequent Pattern Mining]
    ├── Apply modified Apriori algorithm
    ├── Generate candidate itemsets
    └── Filter by minimum support threshold
    ↓
[Maximal Pattern Extraction]
    ├── Remove non-maximal patterns
    └── Keep only patterns with no frequent superset
    ↓
[Cluster Formation]
    ├── Each maximal pattern → potential cluster
    ├── Assign transactions to clusters
    └── Handle overlaps and outliers
    ↓
[Evaluation]
    ├── Silhouette Score
    ├── Cluster Purity (if labels available)
    └── Pattern Coverage
    ↓
Output: Cluster Assignments + Visualizations
```

---

## 7. Key Features

### Algorithmic
✓ Custom frequent pattern mining implementation  
✓ Maximal pattern extraction without external libraries  
✓ Intelligent cluster assignment strategy  
✓ Automatic handling of overlapping patterns  
✓ Outlier detection for unmatched transactions  

### Engineering
✓ Clean Architecture (layered design)  
✓ SOLID principles compliance  
✓ Comprehensive type hints  
✓ Detailed docstrings  
✓ Centralized logging  
✓ Configuration management  
✓ Error handling and validation  

### User Experience
✓ Interactive web interface  
✓ Real-time progress updates  
✓ Multiple visualization types  
✓ Downloadable results  
✓ Parameter tuning interface  

---

## 8. Academic Contributions

This project demonstrates understanding of:

1. **Data Mining Theory**
   - Association rule mining
   - Apriori algorithm principles
   - Pattern growth techniques

2. **Clustering Theory**
   - Non-distance-based clustering
   - Pattern-based similarity measures
   - Cluster validation techniques

3. **Software Engineering**
   - Enterprise architecture patterns
   - RESTful API design
   - Full-stack development

4. **Data Science Workflow**
   - Data preprocessing pipelines
   - Algorithm evaluation
   - Result visualization

---

## 9. Evaluation Metrics

### Silhouette Score
Measures cluster cohesion and separation (-1 to 1, higher is better)

### Cluster Purity
Percentage of correctly clustered items (requires ground truth labels)

### Pattern Coverage
Percentage of transactions covered by discovered patterns

### Cluster Size Distribution
Analyzes balance of cluster sizes

---

## 10. Future Enhancements

- Implement FP-Growth for faster pattern mining
- Add closed pattern mining alongside maximal patterns
- Support hierarchical pattern-based clustering
- Integrate with big data frameworks (Spark)
- Add incremental clustering for streaming data
- Implement pattern evolution tracking
- Support multi-label clustering

---

## 11. Academic References

While this is a custom implementation, the theoretical foundations draw from:

- Agrawal, R., & Srikant, R. (1994). Fast Algorithms for Mining Association Rules
- Han, J., Pei, J., & Yin, Y. (2000). Mining Frequent Patterns without Candidate Generation
- Wang, K., Xu, C., & Liu, B. (1999). Clustering Transactions Using Large Items

---

## 12. Compliance & Standards

- **Code Quality**: PEP8, Black formatted
- **Documentation**: Google-style docstrings
- **Testing**: Unit tests for core algorithms
- **Security**: Input validation, error handling
- **Performance**: O(2^n) worst case for pattern mining (typical for Apriori variants)

---

**Project Status**: Complete Implementation  
**Last Updated**: December 16, 2025  
**Author**: Data Mining Course Project  
**License**: Academic Use Only
