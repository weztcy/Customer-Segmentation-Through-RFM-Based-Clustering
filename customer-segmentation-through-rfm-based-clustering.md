# Portfolio Project Data — Customer Segmentation Through RFM-Based Clustering

- **Category:** `machine-learning` (`Machine Learning`)
- **Accent Color:** `indigo`
- **Target Assets Folder:** `public/images/projects/customer_segmentation_through_rfm/`
- **Required Local Images:** `banner.webp`, `image01.webp` .. `image11.webp`

## TypeScript Data (`ProjectCaseStudy`)

```typescript
{
  "id": "customer-segmentation-through-rfm-based-clustering",
  "code": "PRO-25 // ML-10",
  "title": "Customer Segmentation Through RFM-Based Clustering",
  "category": "machine-learning",
  "categoryLabel": "Unsupervised Learning",
  "year": "2024",
  "status": "Completed",
  "accentColor": "indigo",
  "repoUrl": "https://github.com/weztcy/Customer-Segmentation-Through-RFM-Based-Clustering",
  "license": "MIT License",
  "banner": "/images/projects/customer_segmentation_through_rfm/banner.webp",
  "summary": "Unsupervised K-Means RFM clustering of 4,338 e-commerce customers with log1p normalization, PCA projection, and a Streamlit analytics dashboard.",
  "projectDescription": "An end-to-end unsupervised machine learning pipeline transforming 541,909 UK retail transactions into customer-level RFM profiles. Using log1p normalization, Z-score scaling, and K-Means (k=4), it segments 4,338 buyers into four revenue tiers served in a Streamlit dashboard.",
  "architectureSummary": "Raw Online Retail CSV -> Cleaning Funnel -> Customer RFM Aggregation -> log1p & StandardScaler -> K-Means (k=4) & 2D PCA -> CSV Artifacts & Streamlit Dashboard",
  "techStack": [
    "Python",
    "Pandas",
    "NumPy",
    "Scikit-Learn",
    "Matplotlib",
    "Seaborn",
    "Streamlit"
  ],
  "telemetryMatrix": {
    "projectType": "Unsupervised Clustering Pipeline",
    "targetPlatform": "Python CLI & Streamlit Web App",
    "primaryDomain": "E-Commerce Customer Analytics",
    "architecturePipeline": "RFM + log1p + Z-Score + K-Means",
    "testingAndAudit": "Verified",
    "projectStatus": "Completed",
    "projectYear": "2024"
  },
  "mlProjectInfo": {
    "projectType": "Unsupervised Customer Segmentation Pipeline",
    "learningParadigm": "Unsupervised Learning",
    "modelDomain": "E-Commerce Transactional Analytics",
    "projectStatus": "Completed",
    "developmentYear": "2024",
    "capabilityBadges": [
      "RFM Feature Engineering",
      "K-Means Clustering",
      "PCA Segment Visualization"
    ]
  },
  "mlProblemContext": "Uniform marketing campaigns across diverse online retail and wholesale buyers waste acquisition spend and mask churn risk. Identifying natural behavioral segments from noisy invoice logs requires resolving missing customer IDs, returns, extreme spend skewness, and optimal cluster selection.",
  "mlDatasetOverview": {
    "profileBadge": "UCI Online Retail (352)",
    "source": "UCI Machine Learning Repository (Online_Retail.csv, DOI: 10.24432/C5BW33)",
    "modality": "Tabular Transactional",
    "scale": "541,909 raw rows -> 392,692 valid purchase rows -> 4,338 customers",
    "inputShape": "4,338 Customers x 3 Standardized RFM Features",
    "dataFormat": "CSV (ISO-8859-1)",
    "description": "Contains UK-based online retail transactions from December 1, 2010 to December 9, 2011 across 38 countries, predominantly the United Kingdom (90.36% of valid customers). Aggregated into 4,338 verified purchasing accounts totaling 18,532 orders and £8,887,208.89 in revenue.",
    "provenanceNote": "Public UCI Machine Learning Repository dataset (Daqing Chen et al., 2012); 34 raw CustomerIDs with only cancellations or zero-price rows were excluded."
  },
  "mlFeaturesAndDataCharacteristics": {
    "spaceBadge": "RFM Behavioral Feature Space",
    "spaceNote": "3 continuous behavioral signals aggregated per CustomerID from purchase invoices",
    "objectiveOutput": {
      "label": "Target Output",
      "name": "Customer Segment Cluster",
      "type": "Unsupervised Multi-Cluster (k = 4)",
      "distribution": "At-Risk 37.28% (1,617), Loyal 27.04% (1,173), Recent 19.34% (839), Champions 16.34% (709)"
    },
    "inputSignals": [
      {
        "name": "Recency",
        "type": "Numerical Continuous",
        "importanceOrRole": "PC2 Loading +0.850 | Engagement Decay",
        "description": "Calendar days elapsed between the customer's latest valid purchase and the reference date (2011-12-10 12:50:00); raw median 51 days, skewness +1.25."
      },
      {
        "name": "Frequency",
        "type": "Numerical Discrete",
        "importanceOrRole": "PC1 Loading +0.618 | Repeat Loyalty",
        "description": "Count of distinct purchase InvoiceNo sessions per customer; raw median 2 orders (range 1 to 209), extreme raw skewness +12.07 reduced to +1.21 after log1p."
      },
      {
        "name": "Monetary",
        "type": "Numerical Continuous",
        "importanceOrRole": "PC1 Loading +0.597 | Revenue Contribution",
        "description": "Total gross purchase spend (Quantity x UnitPrice) in GBP; raw median £668.57 (max £280,206.02), raw skewness +19.34 normalized to +0.40 after log1p."
      }
    ],
    "dataCharacteristicsSummary": "Raw RFM distributions exhibit severe right-skewness driven by wholesale accounts (Frequency 12.07, Monetary 19.34). Applying log1p compresses skewness below 1.21 and strengthens linear correlation between Frequency and Monetary from 0.553 to 0.808 before clustering."
  },
  "mlLearningApproach": {
    "paradigm": "Unsupervised Centroid Clustering with Logarithmic Normalization",
    "methodologyDescription": "Customers are grouped without ground-truth labels using K-Means over log1p-transformed, Z-score standardized RFM vectors. Candidate cluster counts k=2..10 are swept and evaluated using Inertia, Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index, and seed stability.",
    "justification": "Logarithmic transformation paired with Z-score scaling prevents extreme wholesale monetary values from dominating Euclidean distances, while K-Means at k=4 achieves balanced segment sizes and strong seed stability (mean ARI 0.9603) compared to Ward linkage (ARI 0.4303)."
  },
  "mlDataPreparation": [
    {
      "step": "01",
      "title": "CustomerID & Timestamp Validation",
      "description": "Parses InvoiceDate timestamps and removes 135,080 rows (24.93%) missing CustomerID so every record maps to an identifiable buyer without synthetic ID imputation.",
      "badge": "406,829 Rows"
    },
    {
      "step": "02",
      "title": "Exact Record Deduplication",
      "description": "Eliminates 5,225 identical duplicate transaction lines (1.28%) across all eight columns while preserving unique invoice integrity (RFM correlation >= 0.999997).",
      "badge": "401,604 Rows"
    },
    {
      "step": "03",
      "title": "Cancellation & Zero-Price Filtering",
      "description": "Excludes 8,872 cancellation/return rows (InvoiceNo starting with 'C' or Quantity <= 0) and 40 zero-price sample rows, yielding 392,692 clean commercial purchase lines.",
      "badge": "392,692 Rows"
    },
    {
      "step": "04",
      "title": "RFM Aggregation, log1p & Z-Score Scaling",
      "description": "Aggregates Recency, Frequency, and Monetary for 4,338 valid buyers, applies np.log1p to normalize heavy tails, and standardizes features via StandardScaler.",
      "badge": "4,338 Profiles"
    }
  ],
  "mlDataPreparationNote": "Purchase-based filtering excluded 34 CustomerIDs that had only cancellation invoices or zero-price records, preventing negative or zero Monetary values from breaking log1p transformations.",
  "mlModelArchitectureAndTraining": {
    "algorithmName": "K-Means Clustering (k = 4) with PCA Projection",
    "architectureSummary": "Partitions the 3-dimensional standardized log1p(RFM) space into 4 Voronoi cells around learned centroids using k-means++ initialization, paired with a 2-component PCA projection retaining 93.87% variance for 2D inspection.",
    "configurationBadge": "k=4 | n_init=20 | Seed 42",
    "hyperparameters": [
      {
        "name": "n_clusters",
        "value": "4",
        "description": "Optimal cluster count balancing WCSS elbow inflection and actionable segment differentiation."
      },
      {
        "name": "init",
        "value": "k-means++",
        "description": "Probabilistic centroid seeding to accelerate convergence and avoid poor local minima."
      },
      {
        "name": "n_init",
        "value": "20",
        "description": "Independent centroid initializations evaluated per fit, retaining the lowest-inertia solution."
      },
      {
        "name": "random_state",
        "value": "42",
        "description": "Fixed RNG seed for reproducible K-Means centroids and 2-component PCA projections."
      }
    ],
    "objectiveCriterion": "Within-Cluster Sum of Squares (Inertia / WCSS)",
    "optimizationSolver": "Lloyd's Iterative EM with k-means++ Initialization",
    "trainingProcess": "The pipeline sweeps k from 2 to 10 on the 4,338 x 3 standardized log1p(RFM) matrix with n_init=20 restarts per k. At k=4, optimization converges to Inertia 3,938.60, and cross-seed validation across five independent seeds confirms 0.9603 mean Adjusted Rand Index stability."
  },
  "mlEvaluationStrategy": {
    "validationMethod": "Multi-Metric Internal Sweep & Seed Stability ARI",
    "description": "Evaluates cluster compactness and separation across k=2..10 using Inertia, Silhouette Score, Davies-Bouldin Index, and Calinski-Harabasz Index, then validates partition robustness via Adjusted Rand Index across 5 random seeds and Ward Hierarchical Clustering.",
    "metrics": [
      {
        "label": "SILHOUETTE SCORE (K=4)",
        "value": "0.3373",
        "benchmark": "Baseline > 0.3365"
      },
      {
        "label": "DAVIES-BOULDIN INDEX",
        "value": "1.0106",
        "benchmark": "Baseline < 1.0483"
      },
      {
        "label": "INERTIA (WCSS)",
        "value": "3,938.60",
        "benchmark": "Baseline < 4,869.49"
      },
      {
        "label": "CALINSKI-HARABASZ",
        "value": "3,328.92",
        "benchmark": "Target > 3,000"
      },
      {
        "label": "SEED STABILITY (ARI)",
        "value": "0.9603",
        "benchmark": "Target > 0.90"
      },
      {
        "label": "2D PCA VARIANCE",
        "value": "93.87%",
        "benchmark": "Target > 85%"
      }
    ]
  },
  "mlPerformanceVisualizations": [
    {
      "title": "Segment Profile Summary Dashboard (Tab 1)",
      "badge": "Streamlit Overview",
      "imageUrl": "/images/projects/customer_segmentation_through_rfm/image01.webp",
      "telemetryNote": "Displays global RFM KPIs (4,338 buyers, £8,887,209 total) alongside per-segment summary tables and share donuts."
    },
    {
      "title": "2D PCA Projection & Boxplot Explorer (Tab 2)",
      "badge": "Interactive Analytics",
      "imageUrl": "/images/projects/customer_segmentation_through_rfm/image02.webp",
      "telemetryNote": "Interactive 2D PCA cluster scatter plot paired with log1p(Monetary) segment distribution boxplots."
    },
    {
      "title": "Segmented Customer Data Explorer (Tab 3)",
      "badge": "Customer Search",
      "imageUrl": "/images/projects/customer_segmentation_through_rfm/image03.webp",
      "telemetryNote": "Filterable customer table sorted by Monetary spend with top Champions accounts and CSV export."
    },
    {
      "title": "Segment-Specific Strategic Recommendations (Tab 4)",
      "badge": "Business Playbook",
      "imageUrl": "/images/projects/customer_segmentation_through_rfm/image04.webp",
      "telemetryNote": "Maps Champions, Loyal, Recent, and At-Risk segments to tailored retention, upselling, and win-back tactics."
    },
    {
      "title": "Customer Share vs. Revenue Contribution (Pareto)",
      "badge": "Donut Comparison",
      "imageUrl": "/images/projects/customer_segmentation_through_rfm/image05.webp",
      "telemetryNote": "Champions (16.3% buyers) drive 64.7% of £8,887,208 revenue while At-Risk (37.3%) generate 6.3%."
    },
    {
      "title": "2D PCA Cluster Map & Inertia Elbow Curve",
      "badge": "PCA + Elbow",
      "imageUrl": "/images/projects/customer_segmentation_through_rfm/image06.webp",
      "telemetryNote": "2D PCA preserves 93.9% variance across 4 distinct centroids; WCSS elbow confirms inflection at k=4 (3,939)."
    },
    {
      "title": "Internal Cluster Validation & log1p RFM Boxplots",
      "badge": "Validation Grid",
      "imageUrl": "/images/projects/customer_segmentation_through_rfm/image07.webp",
      "telemetryNote": "Silhouette peaks post-binary at 0.337 (k=4) with DBI 1.011; boxplots confirm normalized medians (3.95, 1.10, 6.51)."
    },
    {
      "title": "Monthly Transaction Volume & Revenue Trend",
      "badge": "Time Series EDA",
      "imageUrl": "/images/projects/customer_segmentation_through_rfm/image08.webp",
      "telemetryNote": "Tracks Q4 seasonality peaking in November 2011 at 2,657 unique invoices and £1,156k monthly revenue."
    },
    {
      "title": "RFM Distributions Before vs. After log1p Transform",
      "badge": "Distribution Grid",
      "imageUrl": "/images/projects/customer_segmentation_through_rfm/image09.webp",
      "telemetryNote": "Compresses raw skewness (Recency 1.25, Frequency 12.07, Monetary 19.34) to -0.38, 1.21, and 0.40."
    },
    {
      "title": "Missing Values Audit & Pearson Correlation Matrices",
      "badge": "Data Quality + Corr",
      "imageUrl": "/images/projects/customer_segmentation_through_rfm/image10.webp",
      "telemetryNote": "CustomerID missingness is 24.93% (135,080 rows); log1p boosts Frequency-Monetary correlation from 0.553 to 0.808."
    },
    {
      "title": "Segment Behavioral Z-Score Heatmap & Normalized Medians",
      "badge": "Cluster Profiling",
      "imageUrl": "/images/projects/customer_segmentation_through_rfm/image11.webp",
      "telemetryNote": "Champions exhibit Z-scores of -1.25 Recency, +1.67 Frequency, +1.41 Monetary versus At-Risk (+0.89 Recency)."
    }
  ],
  "mlVisualizationTelemetryNote": "All 4 clusters remain geometrically distinct in 2D PCA space (93.87% variance) with balanced population sizes (709 to 1,617) and strong multi-seed reproducibility (ARI = 0.9603).",
  "mlModelInterpretationAndLimitations": {
    "interpretationSummary": "Cluster separation is governed by two orthogonal behavioral axes: engagement and spend intensity (PC1, 75.08% variance) and recency versus dormancy (PC2, 18.79% variance). Just 43.38% of buyers (Champions and Loyal) generate 88.54% of total revenue (£7.87M).",
    "keyDrivers": [
      {
        "factor": "Champions / High-Value (n=709, 16.34%)",
        "impact": "Generates £5,752,065.07 (64.72% of revenue); median 8d Recency, 10 orders, £3,722.82 spend (Z-scores: R -1.25, F +1.67, M +1.41)."
      },
      {
        "factor": "Loyal Customers (n=1,173, 27.04%)",
        "impact": "Generates £2,117,050.02 (23.82% of revenue); median 55d Recency, 4 orders, £1,341.65 spend (Z-scores: R +0.13, F +0.33, M +0.53)."
      },
      {
        "factor": "Recent / Promising Customers (n=839, 19.34%)",
        "impact": "Generates £455,624.56 (5.13% of revenue); median 17d Recency, 2 orders, £466.70 spend (Z-scores: R -0.84, F -0.38, M -0.41)."
      },
      {
        "factor": "At-Risk / Hibernating (n=1,617, 37.28%)",
        "impact": "Generates £562,469.24 (6.33% of revenue); median 177d Recency, 1 order, £298.40 spend (Z-scores: R +0.89, F -0.78, M -0.79)."
      }
    ],
    "limitations": [
      {
        "title": "Historical Observation Window & Censoring",
        "description": "Transactions span December 2010 to December 2011 only, introducing left/right censoring and requiring recalibration for modern retail baselines."
      },
      {
        "title": "Mixed B2B Wholesale and B2C Retail Accounts",
        "description": "High-volume wholesale buyers share the Champions segment with individual retail customers, which can blur promotional tactics without B2B/B2C tagging."
      },
      {
        "title": "Gross Revenue Without Profit Margin",
        "description": "Monetary measures gross sales rather than net margin, meaning high-turnover low-margin buyers appear equivalent to high-margin accounts."
      }
    ],
    "futureImprovements": [
      {
        "title": "B2B vs. B2C Sub-Segmentation",
        "target": "Separate wholesale distributors from consumer shoppers before clustering"
      },
      {
        "title": "Predictive CLV & Churn Modeling",
        "target": "Pair RFM cluster features with BG/NBD or survival models for forward-looking retention scoring"
      }
    ],
    "discoveryNote": "PCA loadings confirm PC1 captures Frequency (+0.618) and Monetary (+0.597) engagement while PC2 isolates Recency (+0.850) dormancy."
  },
  "mlModelArtifacts": [
    {
      "name": "customer_segments.csv",
      "format": "CSV"
    },
    {
      "name": "rfm_customer.csv",
      "format": "CSV"
    },
    {
      "name": "model_metrics.csv",
      "format": "CSV"
    },
    {
      "name": "cluster_pca.png",
      "format": "PNG"
    },
    {
      "name": "cluster_profiles.png",
      "format": "PNG"
    },
    {
      "name": "cluster_contributions.png",
      "format": "PNG"
    }
  ],
  "mlMethodologySpecs": {
    "learningParadigm": "Unsupervised Learning",
    "modelArchitecture": "K-Means & PCA Projection",
    "coreFramework": "Scikit-Learn",
    "validationMethod": "Multi-Metric & Seed ARI"
  }
}
```
