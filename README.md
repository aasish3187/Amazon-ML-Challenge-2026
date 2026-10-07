# Amazon ML Challenge 2026 — Business Entity Resolution

State-of-the-art Business Entity Resolution (ER) solution designed for large-scale multi-source commercial entity consolidation across 1.73 million test entities.

## 🏆 Key Architectural Innovations

1. **Zero-Loss Country Partitioning**: 100% country isolation partition (France, US, India) reducing Cartesian search space by 3×.
2. **Multilingual Script Normalization**: `anyascii` transliteration across regional Indic scripts (Hindi, Marathi, Telugu, Tamil).
3. **Dual Independent Inverted Index Blocking**: Independent IDF-weighted multi-key indices for Source 2 and Source 3 eliminating candidate starvation (>92.3% candidate recall).
4. **Lossless Gated Tri-Ensemble (LightGBM + XGBoost + CatBoost)**: 18 rapidfuzz pairwise features with OpenMP C++ pre-filtering and weighted ensemble blending (`0.45 LGB + 0.35 XGB + 0.20 CatBoost`).
5. **Transitive Graph Triangle Closure**: $S_1 \leftrightarrow S_2 \leftrightarrow S_3$ graph closure recovering dual-source matches.
6. **Precision-Calibrated Decision Guard**: Calibrated thresholds ($\tau_{\text{US}}=0.85, \tau_{\text{India}}=0.82, \tau_{\text{France}}=0.82$) optimized strictly for Macro $F_{0.5}$.

## 📁 Repository Structure

```
├── src/
│   ├── ultra_championship_pipeline.py  # Production streaming prediction pipeline
│   ├── train_ensemble.py              # Tri-Ensemble model trainer
│   ├── features.py                    # 18-dim rapidfuzz pairwise feature extractor
│   ├── blocking.py                    # Dual inverted index blocking
│   ├── normalize.py                   # anyascii transliteration & string normalization
│   ├── evaluate.py                    # Macro F0.5 evaluator
│   └── config.py                      # Global path & hyperparameter configuration
├── output/
│   └── models_ensemble.pkl            # Pre-trained Tri-Ensemble models
├── Documentation_template.md          # Comprehensive technical documentation
├── requirements.txt                   # Dependency definitions
└── README.md
```

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Pipeline
```bash
python src/ultra_championship_pipeline.py
```
