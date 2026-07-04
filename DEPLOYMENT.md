# 🚀 Complete Deployment Guide: Crop Yield Predictor

This guide provides step-by-step instructions for deploying both the **React-based Next.js Frontend Dashboard** and the **Flask Machine Learning API** to production.

---

## 📋 Table of Contents
1. [Why Separate Platforms? (Vercel + Render/Railway)](#1-why-separate-platforms-vercel--renderrailway)
2. [Prerequisites & Git Preparation](#2-prerequisites--git-preparation)
3. [Managing Memory & Model File Sizes (Crucial!)](#3-managing-memory--model-file-sizes-crucial)
4. [Deploying the Flask Backend (Render)](#4-deploying-the-flask-backend-render)
5. [Deploying the Next.js Frontend (Vercel)](#5-deploying-the-next-js-frontend-vercel)

---

## 1. Why Separate Platforms? (Vercel + Render/Railway)

This application is split into two components:
1. **Frontend**: A React/Next.js dynamic client dashboard.
2. **Backend**: A Python Flask server that loads pre-trained scikit-learn models (pickle files) and runs CPU-bound ML inferences.

### Why not deploy the backend on Vercel?
While Vercel supports Python serverless functions, it has significant limits:
* **Cold Starts**: Serverless functions spin down when inactive. On spin-up, the function has to reload all heavy `.pkl` models (KNN, SVM, Random Forest) from disk into RAM. This would cause a **5 to 10-second delay** on API requests during cold starts.
* **Size Limits**: Vercel serverless function bundles are capped at **250 MB unzipped**. The Python dependencies (pandas, scikit-learn, numpy) combined with model files easily exceed this.
* **Persistent State**: A persistent web service (like Render or Railway) loads the ML models once on startup and keeps them warm in memory, allowing for instant (< 100ms) predictions.

**Therefore, the standard, high-performance architecture is:**
- Frontend client deployed to **Vercel**.
- Python backend API deployed to a persistent runner like **Render** or **Railway**.

---

## 2. Prerequisites & Git Preparation

Before launching, prepare your code for Git.

### A. The GitHub 100MB Limit Warning
GitHub does not accept files larger than **100MB**.
- The default trained **`RandomForest_model.pkl` is ~1.85 GB** (too large for Git).
- The dataset **`crop_yield.csv` is ~93.4 MB** (extremely close to the warning limit).

A root `.gitignore` file has been added to exclude these files. **Do not commit these files to GitHub.**

### B. Push the Code to GitHub
Initialize your Git repository and push your project to a new GitHub repository:

```bash
git init
git add .
git commit -m "Initialize project structure with deployment configs"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git push -u origin main
```

---

## 3. Managing Memory & Model File Sizes (Crucial!)

Free cloud services (e.g., Render, Railway) restrict standard web service RAM to **512MB**. 
Loading a **1.85 GB Random Forest model** in Python will consume over **2.5 GB of RAM** and immediately crash the server with an **Out Of Memory (OOM)** error.

### How to Fix: Use Production-Friendly Models
We have provided a helper script in `backend/train_small_models.py` which samples the dataset and restricts model depth to train models that:
- Take less than **100MB RAM** in memory.
- Are extremely fast to load and run.
- Keep the exact same input/output interfaces.

Run this script locally to recreate light models before deploying:

```bash
cd backend
python3 train_small_models.py
```

*Note: Since they are small (< 5MB combined), you can temporarily comment out `models/` and `backend/models/` in your `.gitignore` to push these small models to GitHub so your host can build/load them instantly, or let the host build them on the fly.*

---

## 4. Deploying the Flask Backend (Render)

[Render](https://render.com/) is a simple, free-tier-friendly host for Python APIs.

### Step-by-Step Render Deployment:
1. Create a free account on [Render](https://render.com/).
2. On the dashboard, click **New** -> **Web Service**.
3. Connect your GitHub repository.
4. Set the following configuration values:
   - **Name**: `crop-yield-backend`
   - **Region**: Select the closest region to your users
   - **Branch**: `main`
   - **Root Directory**: `backend`
   - **Runtime**: `Python 3`
   - **Instance Type**: `Free`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
5. Click **Deploy Web Service**.
6. Wait for the build to finish. Once completed, copy the public URL provided by Render (e.g., `https://crop-yield-backend.onrender.com`).

---

## 5. Deploying the Next.js Frontend (Vercel)

[Vercel](https://vercel.com/) is the native platform for Next.js and hosts it for free with high performance.

### Step-by-Step Vercel Deployment:
1. Create an account on [Vercel](https://vercel.com/).
2. Click **Add New** -> **Project**.
3. Select the connected GitHub repository from the list.
4. Set the following configuration parameters:
   - **Framework Preset**: `Next.js`
   - **Root Directory**: Select `frontend` (Very Important!)
   - **Build and Output Settings**: Keep default settings.
   - **Environment Variables**:
     * Add a key: `NEXT_PUBLIC_API_URL`
     * Set the value to your hosted Render backend URL (e.g., `https://crop-yield-backend.onrender.com`).
5. Click **Deploy**.
6. Vercel will build the frontend and provide a production domain name (e.g., `https://crop-yield-frontend.vercel.app`).