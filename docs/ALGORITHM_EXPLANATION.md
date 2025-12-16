# MaPle Algorithm: Theoretical Foundation and Implementation

## Table of Contents
1. [Introduction](#introduction)
2. [Theoretical Background](#theoretical-background)
3. [Algorithm Overview](#algorithm-overview)
4. [Detailed Algorithm Steps](#detailed-algorithm-steps)
5. [Mathematical Formulation](#mathematical-formulation)
6. [Complexity Analysis](#complexity-analysis)
7. [Comparison with Other Algorithms](#comparison-with-other-algorithms)
8. [Implementation Details](#implementation-details)

---

## 1. Introduction

**MaPle (Maximal Pattern-based Clustering)** is a clustering algorithm that groups data based on shared maximal frequent patterns rather than geometric distance measures. This approach is particularly effective for:

- Transactional data (market basket analysis)
- Categorical datasets
- Mixed-type data (numerical + categorical)
- High-dimensional sparse data

The key insight is that items sharing common behavioral patterns (frequent co-occurrences) are more meaningfully similar than items that are merely close in Euclidean space.

---

## 2. Theoretical Background

### 2.1 Frequent Pattern Mining

**Definition**: A pattern (itemset) is *frequent* if its support exceeds a user-defined minimum threshold.

**Support**: The proportion of transactions containing the itemset.

```
Support(X) = |{T ∈ D | X ⊆ T}| / |D|
```

Where:
- X = itemset
- D = database of transactions
- T = individual transaction

**Example**:
```
Transactions:
T1: {Milk, Bread, Butter}
T2: {Milk, Bread}
T3: {Milk, Eggs}
T4: {Bread, Butter}

Support({Milk, Bread}) = 2/4 = 50%
```

### 2.2 Maximal Frequent Patterns

**Definition**: A frequent pattern X is *maximal* if it is frequent but no proper superset of X is frequent.

**Formal Definition**:
```
X is maximal ⟺ Support(X) ≥ min_support ∧ ∀Y ⊃ X: Support(Y) < min_support
```

**Example**:
Given min_support = 40%:
- {Milk, Bread} is frequent (50%)
- {Milk, Bread, Butter} is not frequent (25%)
- Therefore, {Milk, Bread} is **maximal**

### 2.3 Why Maximal Patterns?

**Advantages**:
1. **Compactness**: Fewer patterns to store (eliminates redundancy)
2. **Informativeness**: Represent the longest interesting associations
3. **Efficiency**: Reduce computational overhead in clustering
4. **Interpretability**: Easier to understand and explain clusters

**Comparison**:
- **All Frequent Patterns**: {A}, {B}, {C}, {A,B}, {A,C}, {B,C}, {A,B,C}
- **Maximal Patterns**: {A,B,C} only (if all are frequent)
- **Closed Patterns**: Subset of frequent, superset of maximal

---

## 3. Algorithm Overview

### 3.1 MaPle Pipeline

```
┌────────────────────────────────────────────┐
│         PHASE 1: DATA PREPARATION          │
├────────────────────────────────────────────┤
│ 1. Load raw dataset                        │
│ 2. Handle missing values                   │
│ 3. Normalize numerical features            │
│ 4. Discretize continuous variables         │
│ 5. Transform to transactional format       │
└────────────────────────────────────────────┘
                    ↓
┌────────────────────────────────────────────┐
│    PHASE 2: FREQUENT PATTERN MINING        │
├────────────────────────────────────────────┤
│ 1. Initialize: Find frequent 1-itemsets    │
│ 2. Generate candidates (k+1 from k)        │
│ 3. Count support in database               │
│ 4. Filter by minimum support               │
│ 5. Repeat until no more candidates         │
└────────────────────────────────────────────┘
                    ↓
┌────────────────────────────────────────────┐
│   PHASE 3: MAXIMAL PATTERN EXTRACTION      │
├────────────────────────────────────────────┤
│ 1. Sort patterns by length (descending)    │
│ 2. For each pattern P:                     │
│    - Check if any superset exists          │
│    - If no superset, mark as maximal       │
│ 3. Return maximal pattern set              │
└────────────────────────────────────────────┘
                    ↓
┌────────────────────────────────────────────┐
│     PHASE 4: CLUSTER FORMATION             │
├────────────────────────────────────────────┤
│ 1. Each maximal pattern → cluster seed     │
│ 2. For each transaction T:                 │
│    - Find maximal patterns contained in T  │
│    - Assign to best-matching cluster       │
│ 3. Handle overlaps (multi-assignment)      │
│ 4. Mark uncovered items as outliers        │
└────────────────────────────────────────────┘
                    ↓
┌────────────────────────────────────────────┐
│        PHASE 5: EVALUATION                 │
├────────────────────────────────────────────┤
│ 1. Calculate silhouette scores             │
│ 2. Measure cluster purity (if labels)      │
│ 3. Compute pattern coverage               │
│ 4. Generate visualizations                 │
└────────────────────────────────────────────┘
```

---

## 4. Detailed Algorithm Steps

### 4.1 Data Discretization

**Purpose**: Convert continuous numerical features into categorical bins.

**Methods**:
1. **Equal-Width Binning**: Divide range into equal intervals
2. **Equal-Frequency Binning**: Each bin contains same number of samples
3. **K-Means Binning**: Use K-Means cluster centers as boundaries

**Algorithm**:
```
For each numerical feature F:
  1. Calculate min, max, mean, std
  2. Determine number of bins (default: 5)
  3. Create bin edges:
     - Equal-Width: edges = linspace(min, max, n_bins+1)
     - Equal-Frequency: edges = quantiles(F, n_bins)
  4. Assign each value to bin:
     - value → bin_label (e.g., "Low", "Medium", "High")
  5. Create item: "Feature=BinLabel"
```

**Example**:
```
Age: [25, 32, 45, 53, 67]
Bins: [0-30): "Young", [30-50): "Middle", [50+): "Senior"
Result: {Age=Young, Age=Middle, Age=Middle, Age=Senior, Age=Senior}
```

### 4.2 Transaction Transformation

**Purpose**: Convert tabular data to transactional format.

**Transformation**:
```
Tabular Format:
| CustomerID | Age | Income  | Education |
|------------|-----|---------|-----------|
| C1         | 25  | 30000   | Bachelor  |
| C2         | 45  | 75000   | Master    |

Transactional Format:
T1: {Age=Young, Income=Low, Education=Bachelor}
T2: {Age=Middle, Income=High, Education=Master}
```

### 4.3 Frequent Pattern Mining (Modified Apriori)

**Apriori Property**: All subsets of a frequent itemset must also be frequent.

**Algorithm**:
```python
def apriori(transactions, min_support):
    # Step 1: Find frequent 1-itemsets
    L1 = find_frequent_1_itemsets(transactions, min_support)
    L = [L1]
    k = 1
    
    # Step 2: Generate candidates and find frequent k-itemsets
    while L[k-1]:
        Ck = generate_candidates(L[k-1])  # Candidate generation
        for transaction in transactions:
            candidates_in_t = subset(Ck, transaction)
            for candidate in candidates_in_t:
                candidate.count += 1
        
        Lk = [c for c in Ck if c.count >= min_support]
        if not Lk:
            break
        L.append(Lk)
        k += 1
    
    return flatten(L)  # All frequent patterns
```

**Candidate Generation**:
```python
def generate_candidates(Lk_minus_1):
    """
    Join step: Join Lk-1 with Lk-1
    """
    Ck = []
    for i, itemset1 in enumerate(Lk_minus_1):
        for itemset2 in Lk_minus_1[i+1:]:
            # Join if first k-2 items are same
            union = itemset1 | itemset2
            if len(union) == len(itemset1) + 1:
                # Prune step: Check if all subsets are frequent
                if all_subsets_frequent(union, Lk_minus_1):
                    Ck.append(union)
    return Ck
```

### 4.4 Maximal Pattern Extraction

**Algorithm**:
```python
def extract_maximal_patterns(frequent_patterns):
    """
    A pattern is maximal if no proper superset is frequent
    """
    maximal = []
    
    # Sort by length (descending) for efficiency
    sorted_patterns = sorted(frequent_patterns, 
                            key=len, 
                            reverse=True)
    
    for pattern in sorted_patterns:
        is_maximal = True
        
        # Check if pattern is subset of any maximal pattern
        for max_pattern in maximal:
            if pattern.issubset(max_pattern):
                is_maximal = False
                break
        
        if is_maximal:
            maximal.append(pattern)
    
    return maximal
```

**Optimization**: Use hash tree or trie for faster subset checking.

### 4.5 Cluster Assignment

**Strategy**: Assign each transaction to cluster based on best pattern match.

**Matching Score**:
```
Score(Transaction T, Pattern P) = |T ∩ P| / |P|
```

**Algorithm**:
```python
def assign_clusters(transactions, maximal_patterns):
    clusters = {i: [] for i in range(len(maximal_patterns))}
    outliers = []
    
    for trans_id, transaction in enumerate(transactions):
        best_cluster = -1
        best_score = 0
        
        # Find best matching pattern
        for cluster_id, pattern in enumerate(maximal_patterns):
            if pattern.issubset(transaction):
                score = len(pattern) / len(transaction)
                if score > best_score:
                    best_score = score
                    best_cluster = cluster_id
        
        # Assign to cluster or mark as outlier
        if best_cluster != -1:
            clusters[best_cluster].append(trans_id)
        else:
            outliers.append(trans_id)
    
    return clusters, outliers
```

**Alternative Strategies**:
1. **Hard Assignment**: Each transaction → exactly one cluster
2. **Soft Assignment**: Transaction → multiple clusters with weights
3. **Overlap Threshold**: Assign if overlap > threshold

---

## 5. Mathematical Formulation

### 5.1 Problem Statement

**Given**:
- Dataset D = {T₁, T₂, ..., Tₙ} (transactions)
- Minimum support threshold σ ∈ [0,1]

**Find**:
- Set of maximal patterns M = {P₁, P₂, ..., Pₖ}
- Cluster assignment function: f: D → {1, 2, ..., k, outlier}

**Constraints**:
1. ∀P ∈ M: Support(P) ≥ σ
2. ∀P ∈ M, ∀Q ∈ M: P ⊄ Q (no pattern is subset of another)
3. Minimize number of outliers

### 5.2 Objective Function

**Maximize Pattern Coverage**:
```
Maximize: Coverage = |{T ∈ D | ∃P ∈ M: P ⊆ T}| / |D|
```

**Minimize Intra-cluster Dissimilarity**:
```
Minimize: Σᵢ Σ_{T∈Cᵢ} distance(T, Pᵢ)

where distance(T, P) = 1 - |T ∩ P| / |T ∪ P|  (Jaccard distance)
```

### 5.3 Support Calculation

**Absolute Support**:
```
Support_abs(X) = |{T ∈ D | X ⊆ T}|
```

**Relative Support**:
```
Support_rel(X) = Support_abs(X) / |D|
```

### 5.4 Similarity Measures

**Jaccard Similarity**:
```
Jaccard(T₁, T₂) = |T₁ ∩ T₂| / |T₁ ∪ T₂|
```

**Cosine Similarity** (for binary vectors):
```
Cosine(T₁, T₂) = |T₁ ∩ T₂| / √(|T₁| × |T₂|)
```

---

## 6. Complexity Analysis

### 6.1 Time Complexity

**Data Discretization**: O(n × m)
- n = number of records
- m = number of features

**Frequent Pattern Mining**: O(2^|I| × n)
- |I| = number of unique items
- n = number of transactions
- Worst case: exponential in number of items
- Practical case: polynomial for sparse data

**Maximal Pattern Extraction**: O(|F|²)
- |F| = number of frequent patterns

**Cluster Assignment**: O(n × k)
- n = number of transactions
- k = number of maximal patterns

**Total**: O(2^|I| × n) dominated by pattern mining

### 6.2 Space Complexity

**Transaction Storage**: O(n × avg_transaction_size)

**Frequent Patterns**: O(2^|I|) worst case
- Typically much smaller due to support threshold

**Maximal Patterns**: O(|F_max|)
- |F_max| << |F| (much smaller than all frequent)

### 6.3 Optimization Techniques

1. **Early Pruning**: Remove infrequent items before mining
2. **Hash Trees**: Fast candidate lookup
3. **Transaction Projection**: Focus on relevant transactions
4. **Vertical Format**: Store tidsets instead of transactions
5. **Sampling**: Mine on sample, verify on full dataset

---

## 7. Comparison with Other Algorithms

### 7.1 K-Means vs MaPle

| Aspect | K-Means | MaPle |
|--------|---------|-------|
| Distance Metric | Euclidean | Pattern Matching |
| Data Type | Numerical | Categorical/Mixed |
| Cluster Shape | Spherical | Arbitrary |
| Number of Clusters | Fixed (k) | Automatic |
| Interpretability | Low (centroids) | High (patterns) |
| Outlier Handling | Poor | Explicit |

### 7.2 DBSCAN vs MaPle

| Aspect | DBSCAN | MaPle |
|--------|--------|-------|
| Approach | Density-based | Pattern-based |
| Parameters | ε, minPts | min_support |
| Noise Handling | Yes (outliers) | Yes (uncovered) |
| Cluster Shape | Arbitrary | Pattern-defined |
| Categorical Data | Difficult | Native |

### 7.3 Hierarchical vs MaPle

| Aspect | Hierarchical | MaPle |
|--------|--------------|-------|
| Structure | Dendrogram | Flat |
| Linkage | Distance-based | Pattern-based |
| Complexity | O(n²) or O(n³) | O(2^|I| × n) |
| Interpretability | Moderate | High |

### 7.4 Association Rules vs MaPle

**Similarity**: Both use frequent pattern mining

**Difference**:
- Association Rules: X → Y (predictive)
- MaPle: Group items by shared patterns (descriptive)

---

## 8. Implementation Details

### 8.1 Data Structures

**Transaction Representation**:
```python
# Option 1: Set of strings
transaction = {"Age=Young", "Income=High", "Education=Bachelor"}

# Option 2: Frozenset (hashable)
transaction = frozenset(["Age=Young", "Income=High"])

# Option 3: Binary vector
transaction = [1, 0, 1, 0, 1, ...]  # One-hot encoded
```

**Pattern Storage**:
```python
class FrequentPattern:
    items: frozenset[str]
    support: int
    transactions: list[int]  # Transaction IDs
```

### 8.2 Parameter Selection

**Minimum Support**:
- **Too High**: Miss rare but important patterns
- **Too Low**: Too many patterns, noise
- **Recommended**: Start with 0.05-0.10, adjust based on results
- **Rule of Thumb**: Support × |D| ≥ 5 (at least 5 occurrences)

**Number of Bins** (discretization):
- **Too Few**: Loss of information
- **Too Many**: Sparse patterns
- **Recommended**: 3-5 for small datasets, 5-10 for large

### 8.3 Edge Cases

**Empty Clusters**: Remove or merge with nearest cluster

**All Outliers**: Lower minimum support threshold

**Single Large Cluster**: Increase minimum support or use hierarchical

**No Frequent Patterns**: Check data quality, reduce support threshold

---

## 9. Example Walkthrough

### Sample Dataset
```
| CustomerID | Age | Income | Purchases |
|------------|-----|--------|-----------|
| C1         | 25  | 30000  | 5         |
| C2         | 27  | 35000  | 7         |
| C3         | 45  | 70000  | 12        |
| C4         | 48  | 75000  | 15        |
| C5         | 65  | 40000  | 3         |
```

### Step 1: Discretization
```
Age: [0-30)=Young, [30-50)=Middle, [50+)=Senior
Income: [0-40k)=Low, [40k-60k)=Mid, [60k+)=High
Purchases: [0-5]=Few, [6-10]=Some, [11+]=Many
```

### Step 2: Transactions
```
T1: {Age=Young, Income=Low, Purchases=Few}
T2: {Age=Young, Income=Low, Purchases=Some}
T3: {Age=Middle, Income=High, Purchases=Many}
T4: {Age=Middle, Income=High, Purchases=Many}
T5: {Age=Senior, Income=Low, Purchases=Few}
```

### Step 3: Frequent Patterns (min_support=40%)
```
1-itemsets: {Age=Young}:2, {Age=Middle}:2, {Age=Senior}:1, {Income=Low}:3, {Income=High}:2, ...
2-itemsets: {Age=Young, Income=Low}:2, {Age=Middle, Income=High}:2, ...
3-itemsets: {Age=Middle, Income=High, Purchases=Many}:2
```

### Step 4: Maximal Patterns
```
M1: {Age=Young, Income=Low}
M2: {Age=Middle, Income=High, Purchases=Many}
M3: {Age=Senior, Income=Low, Purchases=Few}
```

### Step 5: Cluster Assignment
```
Cluster 1 (Young Low-Income): C1, C2
Cluster 2 (Middle High-Income Frequent): C3, C4
Cluster 3 (Senior Low-Income): C5
Outliers: None
```

---

## 10. Conclusion

MaPle provides an alternative clustering paradigm particularly suited for categorical and transactional data. By leveraging maximal frequent patterns, it produces interpretable clusters based on co-occurrence patterns rather than geometric proximity.

**Key Advantages**:
- Automatically determines number of clusters
- Handles categorical data natively
- Produces interpretable results
- Explicit outlier detection

**Limitations**:
- Exponential worst-case complexity
- Sensitive to support threshold
- May produce many small clusters with low support

**Best Use Cases**:
- Market basket analysis
- Customer segmentation
- Web usage mining
- Bioinformatics (gene expression patterns)

---

**References**:
- Frequent Pattern Mining: Han, Kamber & Pei. "Data Mining: Concepts and Techniques" (3rd ed.)
- Apriori Algorithm: Agrawal & Srikant (1994)
- Pattern-based Clustering: Wang, Xu & Liu (1999)
