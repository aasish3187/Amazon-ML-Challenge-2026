# Contributing to Amazon ML Challenge 2026 ER Engine

Thank you for your interest in contributing to this Entity Resolution project!

## 🚀 How to Contribute

1. **Fork the Repository**:
   Click the **Fork** button at the top right of this page.

2. **Clone your fork**:
   ```bash
   git clone https://github.com/<your-username>/Amazon-ML-Challenge-2026.git
   cd Amazon-ML-Challenge-2026
   ```

3. **Create a Feature Branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

4. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Run Tests & Module Integrity**:
   ```bash
   python -m compileall src/
   ```

6. **Submit a Pull Request**:
   Push your branch and open a PR with a clear description of your improvements or bug fixes.

## 📐 Guidelines
- Keep all feature computations vectorized or RapidFuzz C-accelerated for high throughput (>1,000 entities/sec).
- Verify that changes do not introduce candidate starvation across Source 2 or Source 3.
- Maintain high precision calibration: remember the metric is **Macro $F_{0.5}$** ($\beta = 0.5$).
