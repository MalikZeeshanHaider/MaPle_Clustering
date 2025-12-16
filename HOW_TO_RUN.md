# How to Run the MaPle Project

## Prerequisites Check

Before starting, ensure you have:
- ✅ Python 3.10 or higher installed
- ✅ pip package manager available
- ✅ At least 4GB RAM
- ✅ Modern web browser (Chrome, Firefox, Edge)

Verify Python version:
```bash
python --version
# Should show: Python 3.10.x or higher
```

---

## Step-by-Step Instructions

### Step 1: Navigate to Project Directory

```bash
cd d:\Data_Mining
```

### Step 2: Create Virtual Environment (Recommended)

**Windows:**
```powershell
python -m venv venv
.\venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt.

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- FastAPI (backend framework)
- Streamlit (frontend framework)
- Pandas, NumPy (data processing)
- Scikit-learn (evaluation metrics)
- Plotly (visualizations)

Installation takes 2-3 minutes.

### Step 4: Start the Backend Server

**Open Terminal 1:**

```bash
# Make sure you're in Data_Mining directory
python -m backend.main
```

**Expected Output:**
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

**🎯 Success!** Backend is now running at http://localhost:8000

You can test it by visiting: http://localhost:8000/docs (API documentation)

### Step 5: Start the Frontend Application

**Open Terminal 2** (keep Terminal 1 running):

```bash
# Make sure you're in Data_Mining directory
streamlit run frontend/app.py
```

**Expected Output:**
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

**🎯 Success!** Frontend will automatically open in your browser.

If it doesn't open automatically, manually navigate to: http://localhost:8501

---

## Using the Application

### Option A: Use Sample Dataset (Fastest)

1. **In the browser**, you'll see the MaPle Clustering homepage
2. Click on **"📤 Upload Dataset"** in the sidebar
3. Click the button **"📋 Use Sample Dataset (Customer)"**
4. Wait 2-3 seconds for upload confirmation
5. Click **"⚙️ Run MaPle"** in the sidebar
6. Click **"🚀 Run MaPle Algorithm"** button
7. Wait 5-10 seconds for clustering to complete
8. Click **"📊 Results & Evaluation"** in the sidebar
9. Explore the results!

### Option B: Upload Your Own Dataset

1. Click **"📤 Upload Dataset"** in sidebar
2. Click **"Browse files"** button
3. Select your CSV file (must have column headers)
4. Click **"🚀 Upload to Backend"**
5. Continue from step 5 above

**CSV Format Requirements:**
- First row must contain column names
- Numerical columns will be auto-detected
- Categorical columns supported
- Missing values are handled automatically

---

## Understanding the Interface

### Left Sidebar (Navigation)
- **System Status**: Shows backend connection
- **Page Selection**: 4 main pages

### Main Pages

#### 1. 📤 Upload Dataset
- Upload CSV files
- Use sample datasets
- View dataset statistics

#### 2. ⚙️ Run MaPle
- Configure algorithm parameters
- Execute clustering
- See quick results

#### 3. 📊 Results & Evaluation
- **Overview**: Cluster summary
- **Patterns**: Discovered maximal patterns
- **Clusters**: Detailed cluster info
- **Evaluation**: Quality metrics and visualizations

#### 4. ℹ️ About
- Algorithm explanation
- Project information
- References

---

## Parameter Guide

### Recommended Starting Values

```
Min Support: 0.05
Max Pattern Length: 10
Number of Bins: 5
Discretization Strategy: quantile
Assignment Strategy: best_match
Overlap Threshold: 0.5
```

### When to Adjust

**Finding too few clusters?**
- Decrease `Min Support` to 0.03 or 0.02

**Too many outliers?**
- Decrease `Min Support`
- Try `discretization_strategy: uniform`

**Want simpler patterns?**
- Decrease `Max Pattern Length` to 5-7
- Increase `Number of Bins` to 7-8

**Want more detailed patterns?**
- Increase `Max Pattern Length` to 12-15
- Decrease `Number of Bins` to 3-4

---

## Stopping the Application

### Stop Frontend
In Terminal 2 (Streamlit):
- Press `Ctrl+C`

### Stop Backend
In Terminal 1 (FastAPI):
- Press `Ctrl+C`

### Deactivate Virtual Environment
```bash
deactivate
```

---

## Troubleshooting

### Issue: "Module not found" error

**Solution:**
```bash
pip install -r requirements.txt
```

### Issue: Backend shows "Address already in use"

**Solution:**
```bash
# Change port in .env or run with different port
python -m backend.main --port 8001
```

Then update frontend's `api_client.py` base_url to `http://localhost:8001`

### Issue: Frontend can't connect to backend

**Check:**
1. Is backend terminal still running?
2. Is there an error in backend terminal?
3. Try accessing http://localhost:8000/docs directly

**Solution:**
- Restart backend
- Check firewall settings

### Issue: "No patterns found"

**Reasons:**
- Min support too high
- Data has insufficient variation
- Max pattern length too short

**Solution:**
- Lower `Min Support` to 0.02
- Increase `Max Pattern Length` to 15
- Try different discretization strategy

### Issue: Clustering is slow

**Normal timing:**
- Small datasets (< 500 rows): 5-15 seconds
- Medium datasets (500-2000 rows): 15-60 seconds
- Large datasets (> 2000 rows): 1-3 minutes

**If slower:**
- Reduce `Max Pattern Length`
- Increase `Min Support`
- Use fewer bins

---

## Verifying Installation

### Test Backend Directly

```bash
# In browser, visit:
http://localhost:8000/docs

# You should see interactive API documentation
```

### Test API Health

```bash
# In terminal:
curl http://localhost:8000/api/v1/health

# Expected output:
{"status":"healthy","service":"MaPle Clustering API","version":"1.0.0"}
```

### Test Frontend

```bash
# Browser should show:
# - Large "🍁 MaPle Clustering" header
# - Sidebar with navigation
# - "System Status" showing green checkmark
```

---

## Sample Workflow

Here's a complete example workflow:

```bash
# 1. Setup (one-time)
cd d:\Data_Mining
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt

# 2. Start backend (Terminal 1)
python -m backend.main
# Wait for "Application startup complete"

# 3. Start frontend (Terminal 2)
streamlit run frontend/app.py
# Browser opens automatically

# 4. In browser:
# - Click "Use Sample Dataset (Customer)"
# - Go to "Run MaPle" page
# - Click "Run MaPle Algorithm"
# - Go to "Results & Evaluation"
# - Explore clusters and patterns

# 5. When done:
# - Ctrl+C in both terminals
# - deactivate
```

---

## File Locations

After setup, your structure should look like:

```
d:\Data_Mining\
├── venv\                    # Virtual environment (created by you)
├── backend\                 # Backend code
├── frontend\                # Frontend code
├── data\                    # Sample datasets
├── requirements.txt         # Dependencies
└── README.md               # Full documentation
```

---

## Getting Help

1. **Full documentation**: See `README.md`
2. **Algorithm details**: See `docs/ALGORITHM_EXPLANATION.md`
3. **API reference**: Visit http://localhost:8000/docs
4. **Quick start**: See `QUICKSTART.md`

---

## Success Checklist

Before running clustering, verify:

- ✅ Backend running (http://localhost:8000/docs accessible)
- ✅ Frontend running (http://localhost:8501 open in browser)
- ✅ "System Status" shows green checkmark
- ✅ Dataset uploaded successfully
- ✅ Parameters configured

If all checked, you're ready to run MaPle!

---

**Next Steps:**
1. Run with sample dataset first
2. Experiment with parameters
3. Try your own dataset
4. Explore evaluation metrics

**Happy Clustering! 🍁**
