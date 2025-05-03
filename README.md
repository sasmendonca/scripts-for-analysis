Overview
This repository provides an automated pipeline for calculating physicochemical properties, analyzing dataset similarities, and exploring statistical correlations in molecular or materials science datasets. Designed for cheminformatics and drug discovery workflows, the tools enable rapid profiling of chemical libraries, bias detection in training/validation sets, and feature selection for QSAR modeling.

Key Features
🔬 Physicochemical Property Calculation

Batch compute 100+ descriptors (logP, molecular weight, H-bond donors/acceptors, etc.) from SMILES/SDF/CSV inputs.
Generate PDF reports with statistical summaries, distribution plots, and heatmaps.

🔄 Dataset Similarity Analysis

Calculate Tanimoto/Dice/Cosine similarities using Morgan fingerprints or MACCS keys.
Visualize dataset overlaps with PCA, t-SNE, or scaffold-based clustering.

📈 Correlation Analysis

Identify linear/non-linear relationships between properties via Pearson, Spearman, and mutual information metrics.
Export interactive plots (Plotly).
