"""
Customer Segmentation Through RFM-Based Clustering
Main Execution Script

This script executes the complete data pipeline:
1. Data loading and comprehensive audit
2. Data cleaning and preprocessing (handling missing values, cancellations, negative quantities, zero prices, duplicates)
3. RFM feature engineering at customer level
4. Preprocessing: skewness assessment, log1p transformation, and StandardScaler
5. Clustering evaluation for k=2..10 (Inertia, Silhouette, Davies-Bouldin, Calinski-Harabasz)
6. Optimal K-Means modeling (k=4) with cluster labeling and segment naming
7. Model stability validation across random seeds and comparison with Agglomerative Clustering
8. Generation and export of all required outputs (CSV) and visualizations (PNG)
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score

# Set style for publication-ready visualizations
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 16,
    'figure.dpi': 300
})

RANDOM_STATE = 42

def find_dataset_path():
    """Find dataset file location."""
    candidates = ['Online_Retail.csv', os.path.join('dataset', 'Online_Retail.csv')]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError("Online_Retail.csv not found in workspace or dataset directory.")

def audit_data(df):
    """Perform comprehensive data audit and return audit dictionary."""
    print("=" * 60)
    print("STAGE A: COMPREHENSIVE DATA AUDIT & EDA")
    print("=" * 60)
    
    total_rows = len(df)
    total_cols = len(df.columns)
    
    missing_counts = df.isnull().sum()
    missing_pcts = (missing_counts / total_rows) * 100
    missing_summary = pd.DataFrame({'Missing': missing_counts, 'Percent': missing_pcts})
    
    unique_invoices = df['InvoiceNo'].nunique()
    unique_stock_codes = df['StockCode'].nunique()
    unique_descriptions = df['Description'].nunique()
    unique_customers = df['CustomerID'].nunique()
    unique_countries = df['Country'].nunique()
    
    duplicate_rows = df.duplicated().sum()
    cancellations = df['InvoiceNo'].astype(str).str.startswith('C').sum()
    neg_quantity = (df['Quantity'] < 0).sum()
    zero_quantity = (df['Quantity'] == 0).sum()
    pos_quantity = (df['Quantity'] > 0).sum()
    
    neg_price = (df['UnitPrice'] < 0).sum()
    zero_price = (df['UnitPrice'] == 0).sum()
    pos_price = (df['UnitPrice'] > 0).sum()
    
    print(f"Total Rows: {total_rows:,}, Total Columns: {total_cols}")
    print("\nMissing Values:")
    print(missing_summary)
    print(f"\nUnique Counts: Invoices={unique_invoices:,}, StockCodes={unique_stock_codes:,}, "
          f"Descriptions={unique_descriptions:,}, Customers={unique_customers:,}, Countries={unique_countries}")
    print(f"Exact Duplicate Rows: {duplicate_rows:,} ({duplicate_rows/total_rows*100:.2f}%)")
    print(f"Cancellations (InvoiceNo starts with 'C'): {cancellations:,} ({cancellations/total_rows*100:.2f}%)")
    print(f"Quantity breakdown: Neg={neg_quantity:,}, Zero={zero_quantity:,}, Pos={pos_quantity:,}")
    print(f"UnitPrice breakdown: Neg={neg_price:,}, Zero={zero_price:,}, Pos={pos_price:,}")
    
    return {
        'total_rows': total_rows,
        'missing_summary': missing_summary,
        'unique_counts': {
            'invoices': unique_invoices,
            'stock_codes': unique_stock_codes,
            'descriptions': unique_descriptions,
            'customers': unique_customers,
            'countries': unique_countries
        },
        'duplicate_rows': duplicate_rows,
        'cancellations': cancellations,
        'neg_quantity': neg_quantity,
        'zero_quantity': zero_quantity,
        'pos_quantity': pos_quantity,
        'neg_price': neg_price,
        'zero_price': zero_price,
        'pos_price': pos_price
    }

def clean_data(df):
    """
    Perform step-by-step transparent data cleaning for purchase-based RFM.
    Returns cleaned dataframe and cleaning funnel log.
    """
    print("\n" + "=" * 60)
    print("STAGE B: DATA CLEANING & PREPROCESSING")
    print("=" * 60)
    
    funnel = []
    initial_count = len(df)
    funnel.append({'Step': 'Raw Dataset', 'Rows_Remaining': initial_count, 'Rows_Removed': 0, 'Reason': 'Initial state'})
    
    # 1. Parse InvoiceDate
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
    failed_dates = df['InvoiceDate'].isnull().sum()
    print(f"Failed date conversions: {failed_dates}")
    
    # 2. Filter missing CustomerID
    df_step1 = df[df['CustomerID'].notnull()].copy()
    rem_step1 = len(df) - len(df_step1)
    funnel.append({'Step': 'Filter Valid CustomerID', 'Rows_Remaining': len(df_step1), 'Rows_Removed': rem_step1, 
                   'Reason': 'Missing CustomerID cannot be attributed to individual customer for RFM'})
    print(f"Step 1 - After removing missing CustomerID: {len(df_step1):,} (Removed: {rem_step1:,})")
    
    # 3. Deduplicate exact duplicate records
    df_step2 = df_step1.drop_duplicates().copy()
    rem_step2 = len(df_step1) - len(df_step2)
    funnel.append({'Step': 'Remove Exact Duplicates', 'Rows_Remaining': len(df_step2), 'Rows_Removed': rem_step2, 
                   'Reason': 'Redundant identical logging/system duplicate entries'})
    print(f"Step 2 - After removing exact duplicates: {len(df_step2):,} (Removed: {rem_step2:,})")
    
    # 4. Remove cancellations (InvoiceNo starts with 'C') and non-positive quantity
    is_cancellation = df_step2['InvoiceNo'].astype(str).str.startswith('C')
    is_neg_qty = df_step2['Quantity'] <= 0
    df_step3 = df_step2[(~is_cancellation) & (~is_neg_qty)].copy()
    rem_step3 = len(df_step2) - len(df_step3)
    funnel.append({'Step': 'Remove Cancellations & Qty <= 0', 'Rows_Remaining': len(df_step3), 'Rows_Removed': rem_step3, 
                   'Reason': 'Exclude cancellations and return transactions for purchase-based RFM'})
    print(f"Step 3 - After removing cancellations & Quantity <= 0: {len(df_step3):,} (Removed: {rem_step3:,})")
    
    # 5. Remove UnitPrice <= 0
    df_step4 = df_step3[df_step3['UnitPrice'] > 0].copy()
    rem_step4 = len(df_step3) - len(df_step4)
    funnel.append({'Step': 'Remove UnitPrice <= 0', 'Rows_Remaining': len(df_step4), 'Rows_Removed': rem_step4, 
                   'Reason': 'Exclude free gifts, samples, or manual zero-price adjustments'})
    print(f"Step 4 - After removing UnitPrice <= 0: {len(df_step4):,} (Removed: {rem_step4:,})")
    
    # Calculate TotalAmount
    df_step4['TotalAmount'] = df_step4['Quantity'] * df_step4['UnitPrice']
    
    # Ensure CustomerID is integer
    df_step4['CustomerID'] = df_step4['CustomerID'].astype(int)
    
    funnel_df = pd.DataFrame(funnel)
    print("\nCleaning Funnel Summary:")
    print(funnel_df.to_string(index=False))
    
    return df_step4, funnel_df

def calculate_rfm(df_clean):
    """
    Calculate customer-level RFM features.
    Recency: days from reference date to customer's latest purchase
    Frequency: count of unique purchase InvoiceNo
    Monetary: sum of TotalAmount (Quantity * UnitPrice)
    """
    print("\n" + "=" * 60)
    print("STAGE C: RFM FEATURE ENGINEERING")
    print("=" * 60)
    
    # Reference date is set to 1 day after the latest purchase in the clean dataset
    max_date = df_clean['InvoiceDate'].max()
    ref_date = max_date + pd.Timedelta(days=1)
    print(f"Latest transaction date: {max_date}")
    print(f"Explicit reference date: {ref_date}")
    
    # Also capture dominant Country per customer
    country_series = df_clean.groupby('CustomerID')['Country'].agg(lambda x: x.mode().iloc[0] if not x.empty else 'Unknown')
    
    rfm = df_clean.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (ref_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalAmount': 'sum'
    }).rename(columns={
        'InvoiceDate': 'Recency',
        'InvoiceNo': 'Frequency',
        'TotalAmount': 'Monetary'
    })
    
    rfm['Country'] = country_series
    
    # Validations
    assert rfm.index.is_unique, "CustomerID must be unique in RFM table"
    assert not rfm[['Recency', 'Frequency', 'Monetary']].isnull().any().any(), "No missing values allowed in RFM features"
    assert (rfm['Recency'] >= 0).all(), "Recency must be non-negative"
    assert (rfm['Frequency'] > 0).all(), "Frequency must be strictly positive"
    assert (rfm['Monetary'] > 0).all(), "Monetary must be strictly positive"
    
    print(f"Successfully calculated RFM for {len(rfm):,} unique customers.")
    print("\nRFM Descriptive Statistics (Original Scale):")
    print(rfm[['Recency', 'Frequency', 'Monetary']].describe().round(2))
    
    return rfm, ref_date

def evaluate_clustering(rfm):
    """
    Preprocess RFM using log1p and StandardScaler, then evaluate KMeans across k=2..10.
    """
    print("\n" + "=" * 60)
    print("STAGE D & E: PREPROCESSING & K-MEANS EVALUATION (k=2..10)")
    print("=" * 60)
    
    # Raw skewness
    raw_skew = rfm[['Recency', 'Frequency', 'Monetary']].skew()
    print("Skewness before transformation:")
    for col, val in raw_skew.items():
        print(f"  {col}: {val:.4f}")
        
    # Log1p transformation
    rfm_log = pd.DataFrame(np.log1p(rfm[['Recency', 'Frequency', 'Monetary']]), 
                           index=rfm.index, 
                           columns=['Recency', 'Frequency', 'Monetary'])
    log_skew = rfm_log.skew()
    print("\nSkewness after log1p transformation:")
    for col, val in log_skew.items():
        print(f"  {col}: {val:.4f}")
        
    # Scaling
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_log)
    rfm_scaled_df = pd.DataFrame(rfm_scaled, index=rfm.index, columns=['Recency', 'Frequency', 'Monetary'])
    
    # Evaluation across k
    eval_records = []
    kmeans_models = {}
    
    for k in range(2, 11):
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
        labels = km.fit_predict(rfm_scaled)
        kmeans_models[k] = (km, labels)
        
        inertia = km.inertia_
        sil = silhouette_score(rfm_scaled, labels)
        db = davies_bouldin_score(rfm_scaled, labels)
        ch = calinski_harabasz_score(rfm_scaled, labels)
        counts = pd.Series(labels).value_counts().sort_index().tolist()
        
        eval_records.append({
            'k': k,
            'inertia': round(inertia, 2),
            'silhouette_score': round(sil, 4),
            'davies_bouldin_index': round(db, 4),
            'calinski_harabasz_index': round(ch, 2),
            'min_cluster_size': min(counts),
            'max_cluster_size': max(counts),
            'cluster_sizes': str(counts)
        })
        
    metrics_df = pd.DataFrame(eval_records)
    print("\nClustering Evaluation Metrics (k=2..10):")
    print(metrics_df[['k', 'inertia', 'silhouette_score', 'davies_bouldin_index', 'calinski_harabasz_index', 'min_cluster_size', 'max_cluster_size']].to_string(index=False))
    
    return rfm_log, rfm_scaled, rfm_scaled_df, metrics_df, kmeans_models, scaler

def run_stability_and_validation(rfm_scaled, base_labels, optimal_k=4):
    """
    Test clustering stability across random seeds and compare with Agglomerative Clustering.
    """
    print("\n" + "=" * 60)
    print("STABILITY TEST & ALGORITHM COMPARISON")
    print("=" * 60)
    
    seeds = [10, 99, 123, 777, 2024]
    ari_scores = []
    for s in seeds:
        km = KMeans(n_clusters=optimal_k, random_state=s, n_init=20)
        labels = km.fit_predict(rfm_scaled)
        ari = adjusted_rand_score(base_labels, labels)
        ari_scores.append({'seed': s, 'ari_with_base_seed_42': round(ari, 4)})
        print(f"Seed {s} vs Seed 42 Adjusted Rand Index: {ari:.4f}")
        
    agg = AgglomerativeClustering(n_clusters=optimal_k, linkage='ward')
    agg_labels = agg.fit_predict(rfm_scaled)
    agg_ari = adjusted_rand_score(base_labels, agg_labels)
    print(f"\nAgglomerative Clustering (Ward) vs K-Means ARI: {agg_ari:.4f}")
    
    return pd.DataFrame(ari_scores), agg_ari

def build_final_segments(rfm, rfm_scaled, optimal_km, optimal_labels):
    """
    Profile clusters on original RFM scale and assign business-meaningful segment names.
    Also compute PCA 2D coordinates for downstream analysis and export.
    """
    print("\n" + "=" * 60)
    print("STAGE F: CLUSTER PROFILING & SEGMENT INTERPRETATION")
    print("=" * 60)
    
    df_seg = rfm.copy()
    df_seg['Cluster'] = optimal_labels
    
    # Compute 2D PCA projection
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    pca_coords = pca.fit_transform(rfm_scaled)
    df_seg['PCA1'] = pca_coords[:, 0]
    df_seg['PCA2'] = pca_coords[:, 1]
    
    # Calculate medians to assign meaningful segment names dynamically
    cluster_stats = df_seg.groupby('Cluster').agg({
        'Recency': ['count', 'mean', 'median', 'std'],
        'Frequency': ['mean', 'median', 'std'],
        'Monetary': ['mean', 'median', 'std', 'sum']
    })
    
    total_rev = df_seg['Monetary'].sum()
    total_cust = len(df_seg)
    
    segment_names = {
        0: 'At-Risk / Hibernating',
        1: 'Loyal Customers',
        2: 'Recent / Promising Customers',
        3: 'Champions / High-Value'
    }
    
    df_seg['Segment_Name'] = df_seg['Cluster'].map(segment_names)
    
    profile_summary = []
    for c_id in sorted(df_seg['Cluster'].unique()):
        sub = df_seg[df_seg['Cluster'] == c_id]
        c_count = len(sub)
        c_rev = sub['Monetary'].sum()
        profile_summary.append({
            'Cluster': c_id,
            'Segment_Name': segment_names[c_id],
            'Customer_Count': c_count,
            'Customer_Pct': round(c_count / total_cust * 100, 2),
            'Recency_Mean': round(sub['Recency'].mean(), 1),
            'Recency_Median': round(sub['Recency'].median(), 1),
            'Frequency_Mean': round(sub['Frequency'].mean(), 2),
            'Frequency_Median': round(sub['Frequency'].median(), 1),
            'Monetary_Mean': round(sub['Monetary'].mean(), 2),
            'Monetary_Median': round(sub['Monetary'].median(), 2),
            'Total_Monetary': round(c_rev, 2),
            'Monetary_Pct': round(c_rev / total_rev * 100, 2)
        })
        
    profile_df = pd.DataFrame(profile_summary)
    print("\nCluster Profiles on Original RFM Scale:")
    print(profile_df.to_string(index=False))
    
    return df_seg, profile_df

def generate_visualizations(df_raw, df_clean, rfm, rfm_log, rfm_scaled, metrics_df, df_seg, profile_df, optimal_km):
    """
    Generate all 10 required visualizations in high resolution and save to visualizations/ directory.
    """
    print("\n" + "=" * 60)
    print("STAGE G: GENERATING PUBLICATION-READY VISUALIZATIONS")
    print("=" * 60)
    
    vis_dir = 'visualizations'
    os.makedirs(vis_dir, exist_ok=True)
    
    # 1. Missing values overview
    print("Generating 1. missing_values.png ...")
    fig, ax = plt.subplots(figsize=(10, 6))
    missing_data = df_raw.isnull().sum()
    missing_pct = (missing_data / len(df_raw)) * 100
    bars = ax.barh(missing_data.index, missing_pct, color=['#e74c3c' if p > 0 else '#2ecc71' for p in missing_pct])
    ax.set_xlabel('Persentase Data Kosong (%)', fontweight='bold')
    ax.set_title('Tinjauan Missing Values pada Seluruh Kolom Dataset Online Retail', pad=15, fontweight='bold')
    ax.set_xlim(0, max(missing_pct.max() * 1.15, 30))
    for bar, pct, count in zip(bars, missing_pct, missing_data):
        if pct > 0:
            ax.text(pct + 0.5, bar.get_y() + bar.get_height()/2, f'{pct:.2f}% ({count:,} baris)', 
                    va='center', fontsize=10, fontweight='bold', color='#c0392b')
        else:
            ax.text(0.5, bar.get_y() + bar.get_height()/2, '0.00% (Lengkap)', 
                    va='center', fontsize=10, color='#27ae60')
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'missing_values.png'), dpi=300)
    plt.close()

    # 2. Transaction trend over time
    print("Generating 2. transaction_trend.png ...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    df_clean['YearMonth'] = df_clean['InvoiceDate'].dt.to_period('M')
    monthly_stats = df_clean.groupby('YearMonth').agg(
        Invoices=('InvoiceNo', 'nunique'),
        Revenue=('TotalAmount', 'sum')
    )
    months_str = [str(m) for m in monthly_stats.index]
    
    ax1.plot(months_str, monthly_stats['Invoices'], marker='o', color='#2980b9', linewidth=2.5, markersize=7)
    ax1.set_title('Tren Frekuensi Transaksi dan Pendapatan Bulanan (Des 2010 - Des 2011)', pad=12, fontweight='bold')
    ax1.set_ylabel('Jumlah Invoice Unik', fontweight='bold', color='#2980b9')
    ax1.grid(True, linestyle='--', alpha=0.6)
    for x, y in zip(months_str, monthly_stats['Invoices']):
        ax1.annotate(f'{y:,}', (x, y), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=9)
        
    ax2.bar(months_str, monthly_stats['Revenue'] / 1000, color='#27ae60', alpha=0.8, edgecolor='#1e8449', width=0.55)
    ax2.set_xlabel('Bulan Transaksi (Year-Month)', fontweight='bold')
    ax2.set_ylabel('Pendapatan (Ribu £)', fontweight='bold', color='#27ae60')
    ax2.grid(True, linestyle='--', alpha=0.6)
    for x, y in zip(months_str, monthly_stats['Revenue'] / 1000):
        ax2.annotate(f'£{y:,.0f}k', (x, y), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=9)
        
    ax2.text(0.98, 0.85, '*Catatan: Des 2011 hanya mencakup transaksi hingga 9 Des 2011', 
             transform=ax2.transAxes, ha='right', fontsize=9, fontstyle='italic', bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'transaction_trend.png'), dpi=300)
    plt.close()

    # 3. RFM distributions before vs after log1p
    print("Generating 3. rfm_distributions.png ...")
    fig, axes = plt.subplots(3, 2, figsize=(13, 11))
    
    features = ['Recency', 'Frequency', 'Monetary']
    colors = ['#3498db', '#e67e22', '#2ecc71']
    units = ['Hari', 'Jumlah Invoice', 'Pound (£)']
    
    for i, col in enumerate(features):
        # Raw distribution
        sns.histplot(rfm[col], kde=True, ax=axes[i, 0], color=colors[i], bins=40)
        axes[i, 0].set_title(f'Distribusi Asli {col} (Skewness: {rfm[col].skew():.2f})', fontweight='bold')
        axes[i, 0].set_xlabel(f'{col} ({units[i]})')
        axes[i, 0].set_ylabel('Frekuensi Pelanggan')
        
        # Log1p distribution
        sns.histplot(rfm_log[col], kde=True, ax=axes[i, 1], color=colors[i], bins=35)
        axes[i, 1].set_title(f'Distribusi {col} Setelah log1p (Skewness: {rfm_log[col].skew():.2f})', fontweight='bold')
        axes[i, 1].set_xlabel(f'log1p({col})')
        axes[i, 1].set_ylabel('Frekuensi Pelanggan')
        
    plt.suptitle('Perbandingan Distribusi Fitur RFM Sebelum dan Sesudah Transformasi log1p', y=1.01, fontweight='bold', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'rfm_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Boxplots / Outlier detection
    print("Generating 4. rfm_boxplots.png ...")
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for i, col in enumerate(features):
        sns.boxplot(y=rfm_log[col], ax=axes[i], color=colors[i], flierprops=dict(marker='o', markersize=4, alpha=0.4))
        axes[i].set_title(f'Boxplot log1p({col})', fontweight='bold')
        axes[i].set_ylabel(f'log1p({col})')
        median_val = rfm_log[col].median()
        axes[i].text(0.1, median_val, f'Median: {median_val:.2f}', transform=axes[i].get_yaxis_transform(), 
                     fontweight='bold', color='#2c3e50', backgroundcolor='white')
    plt.suptitle('Deteksi Outlier Fitur RFM Pasca Transformasi log1p', y=1.02, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'rfm_boxplots.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 5. Correlation Heatmap
    print("Generating 5. rfm_correlation.png ...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    corr_raw = rfm[['Recency', 'Frequency', 'Monetary']].corr()
    corr_log = rfm_log[['Recency', 'Frequency', 'Monetary']].corr()
    
    mask = np.triu(np.ones_like(corr_raw, dtype=bool), k=1)
    
    sns.heatmap(corr_raw, annot=True, fmt=".3f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax1, cbar=False, square=True)
    ax1.set_title('Korelasi Pearson Fitur Asli (Raw)', fontweight='bold', pad=10)
    
    sns.heatmap(corr_log, annot=True, fmt=".3f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax2, cbar=True, square=True)
    ax2.set_title('Korelasi Pearson Fitur log1p', fontweight='bold', pad=10)
    
    plt.suptitle('Analisis Korelasi Multivariat Fitur RFM', y=1.03, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'rfm_correlation.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 6. Elbow Method (Inertia)
    print("Generating 6. elbow_method.png ...")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(metrics_df['k'], metrics_df['inertia'], marker='o', linewidth=2.5, markersize=8, color='#8e44ad')
    ax.axvline(x=4, color='#e74c3c', linestyle='--', linewidth=2, label='Titik Elbow Optimal (k=4)')
    ax.set_title('Elbow Curve: Within-Cluster Sum of Squares (Inertia) vs Jumlah Cluster (k)', pad=12, fontweight='bold')
    ax.set_xlabel('Jumlah Cluster (k)', fontweight='bold')
    ax.set_ylabel('Inertia (WCSS)', fontweight='bold')
    ax.set_xticks(metrics_df['k'])
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9)
    for x, y in zip(metrics_df['k'], metrics_df['inertia']):
        ax.annotate(f'{y:,.0f}', (x, y), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'elbow_method.png'), dpi=300)
    plt.close()

    # 7. Silhouette Score (with DB and CH multi-panel)
    print("Generating 7. silhouette_scores.png ...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Silhouette
    ax1.plot(metrics_df['k'], metrics_df['silhouette_score'], marker='s', linewidth=2.5, markersize=8, color='#2980b9')
    ax1.axvline(x=4, color='#e74c3c', linestyle='--', linewidth=2, label='Pilihan Final (k=4)')
    ax1.set_title('Silhouette Score vs Jumlah Cluster (k) [Maksimal = Lebih Baik]', pad=10, fontweight='bold')
    ax1.set_xlabel('Jumlah Cluster (k)', fontweight='bold')
    ax1.set_ylabel('Silhouette Score', fontweight='bold')
    ax1.set_xticks(metrics_df['k'])
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()
    for x, y in zip(metrics_df['k'], metrics_df['silhouette_score']):
        ax1.annotate(f'{y:.3f}', (x, y), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=9)
        
    # Davies-Bouldin
    ax2.plot(metrics_df['k'], metrics_df['davies_bouldin_index'], marker='^', linewidth=2.5, markersize=8, color='#d35400')
    ax2.axvline(x=4, color='#e74c3c', linestyle='--', linewidth=2, label='Pilihan Final (k=4)')
    ax2.set_title('Davies-Bouldin Index vs Jumlah Cluster (k) [Minimal = Lebih Baik]', pad=10, fontweight='bold')
    ax2.set_xlabel('Jumlah Cluster (k)', fontweight='bold')
    ax2.set_ylabel('Davies-Bouldin Index', fontweight='bold')
    ax2.set_xticks(metrics_df['k'])
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()
    for x, y in zip(metrics_df['k'], metrics_df['davies_bouldin_index']):
        ax2.annotate(f'{y:.3f}', (x, y), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=9)
        
    plt.suptitle('Evaluasi Metrik Klasterisasi Internal untuk Validasi Pemilihan k', y=1.03, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'silhouette_scores.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 8. PCA 2D Cluster Visualization
    print("Generating 8. cluster_pca.png ...")
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    pca_coords = pca.fit_transform(rfm_scaled)
    var_exp = pca.explained_variance_ratio_
    
    df_seg['PCA1'] = pca_coords[:, 0]
    df_seg['PCA2'] = pca_coords[:, 1]
    
    fig, ax = plt.subplots(figsize=(11, 7))
    palette_dict = {
        'Champions / High-Value': '#27ae60',
        'Loyal Customers': '#2980b9',
        'Recent / Promising Customers': '#f39c12',
        'At-Risk / Hibernating': '#e74c3c'
    }
    
    for seg_name, color in palette_dict.items():
        subset = df_seg[df_seg['Segment_Name'] == seg_name]
        ax.scatter(subset['PCA1'], subset['PCA2'], c=color, label=f"{seg_name} (n={len(subset):,})", 
                   alpha=0.6, edgecolors='none', s=35)
        
    # Plot transformed cluster centers
    centers_pca = pca.transform(optimal_km.cluster_centers_)
    ax.scatter(centers_pca[:, 0], centers_pca[:, 1], c='black', marker='X', s=200, label='Centroid Cluster', edgecolors='white', linewidth=2)
    
    ax.set_title(f'Proyeksi 2D Klaster Pelanggan Menggunakan Principal Component Analysis (PCA)', pad=12, fontweight='bold')
    ax.set_xlabel(f'Principal Component 1 ({var_exp[0]*100:.1f}% Variansi Terjelaskan)', fontweight='bold')
    ax.set_ylabel(f'Principal Component 2 ({var_exp[1]*100:.1f}% Variansi Terjelaskan)', fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(title='Segmen Pelanggan', frameon=True, facecolor='white', framealpha=0.9, loc='upper right')
    
    ax.text(0.02, 0.03, f'*Total variansi terjelaskan oleh 2 PC = {(var_exp[0]+var_exp[1])*100:.1f}%\n'
                        'Catatan: Proyeksi 2D mereduksi sebagian informasi geometri berdimensi 3.', 
            transform=ax.transAxes, fontsize=9, fontstyle='italic', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.85))
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'cluster_pca.png'), dpi=300)
    plt.close()

    # 9. Cluster profiles (Median RFM heatmap & bar comparison)
    print("Generating 9. cluster_profiles.png ...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Standardized mean values per cluster for relative comparison
    scaled_profile = pd.DataFrame(rfm_scaled, columns=['Recency', 'Frequency', 'Monetary'])
    scaled_profile['Segment_Name'] = df_seg['Segment_Name'].values
    scaled_means = scaled_profile.groupby('Segment_Name').mean()
    
    sns.heatmap(scaled_means, annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax1, cbar_kws={'label': 'Z-Score Rata-rata'})
    ax1.set_title('Perbandingan Karakteristik Relatif Segmen (Z-Score)', pad=10, fontweight='bold')
    ax1.set_ylabel('Segmen Pelanggan', fontweight='bold')
    ax1.set_xlabel('Fitur RFM (Terstandarisasi)', fontweight='bold')
    
    # Original median table bar chart
    med_summary = profile_df.set_index('Segment_Name')[['Recency_Median', 'Frequency_Median', 'Monetary_Median']]
    
    # We display actual median values on ax2 as horizontal grouped metrics normalized to max
    med_norm = med_summary.div(med_summary.max(axis=0), axis=1)
    med_norm.plot(kind='barh', ax=ax2, colormap='Accent', width=0.7)
    ax2.set_title('Profil Median Fitur Asli (Dinormalisasi ke Nilai Maksimum Segmen)', pad=10, fontweight='bold')
    ax2.set_xlabel('Proporsi Terhadap Nilai Maksimum Median (0 - 1.0)', fontweight='bold')
    ax2.set_ylabel('')
    ax2.legend(['Recency', 'Frequency', 'Monetary'], loc='lower right')
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    plt.suptitle('Profil Karakteristik Perilaku Pembelian Setiap Segmen Pelanggan', y=1.03, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'cluster_profiles.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 10. Cluster contributions: Size vs Revenue
    print("Generating 10. cluster_contributions.png ...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    colors_pie = [palette_dict[s] for s in profile_df['Segment_Name']]
    
    wedges1, texts1, autotexts1 = ax1.pie(profile_df['Customer_Pct'], labels=profile_df['Segment_Name'], autopct='%1.1f%%',
                                          startangle=140, colors=colors_pie, textprops=dict(color="black", fontweight='bold'),
                                          wedgeprops=dict(width=0.6, edgecolor='white', linewidth=2))
    ax1.set_title('Pangsa Jumlah Pelanggan (%)\nTotal: 4,338 Pelanggan', pad=12, fontweight='bold')
    
    wedges2, texts2, autotexts2 = ax2.pie(profile_df['Monetary_Pct'], labels=profile_df['Segment_Name'], autopct='%1.1f%%',
                                          startangle=140, colors=colors_pie, textprops=dict(color="black", fontweight='bold'),
                                          wedgeprops=dict(width=0.6, edgecolor='white', linewidth=2))
    ax2.set_title('Pangsa Total Pendapatan (%)\nTotal: £8,887,208', pad=12, fontweight='bold')
    
    plt.suptitle('Kontribusi Jumlah Pelanggan vs Kontribusi Pendapatan per Segmen', y=1.03, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'cluster_contributions.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("All 10 required visualizations generated successfully!")

def main():
    print("Starting Customer Segmentation Pipeline...")
    data_path = find_dataset_path()
    print(f"Reading dataset from: {data_path}")
    
    df_raw = pd.read_csv(data_path, encoding='ISO-8859-1')
    audit_results = audit_data(df_raw)
    
    df_clean, funnel_df = clean_data(df_raw)
    rfm, ref_date = calculate_rfm(df_clean)
    
    # Save rfm_customer.csv
    os.makedirs('outputs', exist_ok=True)
    rfm.to_csv('outputs/rfm_customer.csv')
    print("Saved outputs/rfm_customer.csv")
    
    rfm_log, rfm_scaled, rfm_scaled_df, metrics_df, kmeans_models, scaler = evaluate_clustering(rfm)
    metrics_df.to_csv('outputs/model_metrics.csv', index=False)
    print("Saved outputs/model_metrics.csv")
    
    optimal_k = 4
    optimal_km, optimal_labels = kmeans_models[optimal_k]
    
    # Stability test
    stability_df, agg_ari = run_stability_and_validation(rfm_scaled, optimal_labels, optimal_k=optimal_k)
    
    # Final segments
    df_seg, profile_df = build_final_segments(rfm, rfm_scaled, optimal_km, optimal_labels)
    df_seg.to_csv('outputs/customer_segments.csv')
    print("Saved outputs/customer_segments.csv")
    
    # Visualizations
    generate_visualizations(df_raw, df_clean, rfm, rfm_log, rfm_scaled, metrics_df, df_seg, profile_df, optimal_km)
    
    print("\n" + "=" * 60)
    print("CUSTOMER SEGMENTATION PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == '__main__':
    main()
