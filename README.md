# 🛒 Instacart Recommendation Engine

![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![Status](https://img.shields.io/badge/Status-Production--Ready-success?style=flat-square)
![Audit](https://img.shields.io/badge/Forensic_Audit-27_Phases_Passed-brightgreen?style=flat-square)
![Model](https://img.shields.io/badge/Champion_Model-LightGBM_Hybrid-orange?style=flat-square)

An end-to-end, highly optimized, and **forensically validated** machine learning recommendation system built for the Instacart Market Basket dataset. This repository represents a production-ready ML pipeline and serving architecture, thoroughly audited and hardened across 27 rigorous engineering phases.

---

## ℹ️ About

**Instacart Recommendation Engine** is an enterprise-grade, highly scalable machine learning pipeline engineered to predict and recommend the most relevant products to users based on intricate historical purchasing patterns. Built on top of the Instacart dataset, this system goes beyond traditional collaborative filtering by employing advanced sequential modeling, deep learning components (including RNNs, SGNS, and SASRec), and a highly optimized LightGBM hybrid sequence architecture.

What sets this project apart is its rigorous, multi-phase forensic validation framework. Every stage of the pipeline—from data preprocessing and candidate generation to model blending and temporal evaluation—is subjected to strict causality audits, reproducibility checks, and data leakage prevention mechanisms.

## 🌟 Key Features

*   🏆 **Champion ML Model**: Utilizes a frozen, rigorously validated **LightGBM hybrid sequence model** (lightgbm_hybrid_seq) for ultra-accurate product ranking.
*   ⚡ **High-Performance Data Layer**: Replaced heavy CSV-based data loading with lightning-fast offline stores. Reduced memory footprint from >15GB to **~235MB**.
*   🧠 **Intelligent Candidate Generation**: Combines historical frequency, basket recency, similarity (co-purchase/affinity), and long-tail discovery strategies.
*   🛡️ **Zero Data Leakage**: Enforces strict chronological boundaries, ensuring the model never sees future interactions during training.
*   🔄 **Diversity & Business Rules**: Integrates Maximal Marginal Relevance (MMR) for result diversification and fallback policies for cold starts.

## 📊 Performance Benchmarks

Under extreme concurrent load tests:
*   **Success Rate**: 100% (0 Timeouts, 0 Failures)
*   **Throughput**: ~96.27 req/s
*   **P50 Latency**: ~188 ms
*   **P95 Latency**: ~250 ms
*   **Model-Only Inference**: ~2.45 ms / user
*   **Champion Integrity**: 100% Preserved (SHA256 Verified)

---

## 🏗️ Architecture

The system is modularly designed to separate concerns, enforce immutability where needed, and ensure maintainability:

\\	ext
Instacart-Recommendation-Engine/
├── run_inference.py     # Command-line entry point for running user predictions
├── run.sh               # Bash script to execute the entire training pipeline
├── ABOUT.md             # In-depth architectural and algorithmic vision 
├── preprocessing/       # Data processing scripts (user, product, aisle data)
├── models/              # Model definitions, phases, and auditing logic (Phases 2-5)
├── tests/               # System health checks and full isolation tests
├── artifacts/           # Saved model binaries, metrics, and generated forensic reports
└── data/                # Raw and processed datasets required for training and inference
\
## 🚀 Getting Started

### 1. Prerequisites
*   **Python 3.9+**
*   **Libraries:** LightGBM, Pandas, NumPy, psutil

### 2. Installation
Clone the repository and install dependencies:
\\ash
git clone https://github.com/saeedsad821-ux/Instacart-Recommendation-Engine-.git
cd Instacart-Recommendation-Engine-
pip install -r requirements.txt
\
### 3. Data Preparation
Ensure offline stores are generated in \data/processed/\. If you have the raw Instacart data in \data/raw/\, you can execute the full pipeline:
\\ash
bash run.sh
\*(Note: For immediate testing, mock data can be placed in \data/processed/product_data.csv\ to satisfy health checks).*

### 4. System Health Check
Verify that the champion model has not been tampered with and the environment is properly isolated:
\\ash
python tests/project_health_check.py
python tests/full_isolation_test.py
\
### 5. Running Inference
Execute the inference pipeline to generate real-time recommendations for a specific user:
\\ash
python run_inference.py --user-id 1 --top-k 10
\
---

## 🔬 Forensic Validation & Scientific Verdict

This project underwent an unprecedented **27-phase forensic audit and hardening process**. Every component—from data equivalence and candidate generation logic to ML model integrity (SHA256 lock) and concurrent thread-pool execution—has been independently verified. 

**Highlights from the Audit:**
*   **Causal Evidence**: Proven via feature ablation and sequential shuffle tests (p < 0.0001).
*   **Metric Reconciliation**: Strict evaluator divergence resolved.
*   **Status**: FROZEN CHAMPION v1.0 — VERIFIED.

For detailed audit logs, please review the markdown reports automatically generated inside \rtifacts/reports/\.

