#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Offline Evaluation Script for sCT Reconstruction

Computes MAE, PSNR, and MS-SSIM metrics on generated sCT volumes,
reports statistics, selects representative cases, and generates histograms.

Usage:
    python Evaluate_metrics.py

Requirements:
    - Trained model with test_split.json in ./Checkpoints_3D/
    - Generated sCT volumes in ./test_results_3d/
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import SimpleITK as sitk
from tqdm import tqdm
from typing import Dict, List, Tuple

from SynthRAD_metrics import ImageMetrics


# --------------------------
# Configuration
# --------------------------
dataset_root = "/mnt/asgard0/users/p25_2025/synthRAD2025_Task2_Train/synthRAD2025_Task2_Train/Task2"
save_dir = "./Checkpoints_3D"
output_dir = "./test_results_3d"

# Initialize metrics module
metrics = ImageMetrics(debug=False)


# --------------------------
# Function Definitions
# --------------------------

def load_test_patients(split_file: str, dataset_root: str, sct_dir: str) -> List[Dict]:
    """
    Load test patient information and build file paths.
    
    Parameters
    ----------
    split_file : str
        Path to test_split.json
    dataset_root : str
        Root directory of the dataset
    sct_dir : str
        Directory containing generated sCT volumes
    
    Returns
    -------
    patients : list of dict
        List of patient info with paths to gt, pred, mask
    """
    # Load test split JSON
    with open(split_file, 'r') as f:
        test_entries = json.load(f)
    
    print(f"\nLoaded {len(test_entries)} test patients from {split_file}")
    
    patients = []
    for entry in test_entries:
        cohort = entry["cohort"]
        pid = entry["pid"]
        
        # Build file paths
        patient_dir = os.path.join(dataset_root, cohort, pid)
        gt_path = os.path.join(patient_dir, "ct.mha")
        mask_path = os.path.join(patient_dir, "mask.mha")
        pred_path = os.path.join(sct_dir, f"{pid}_sct.mha")
        
        # Check if all required files exist
        if not os.path.exists(gt_path):
            print(f"  Warning: Ground truth CT not found for {pid} - skipping")
            continue
        
        if not os.path.exists(pred_path):
            print(f"  Warning: Predicted sCT not found for {pid} - skipping")
            continue
        
        if not os.path.exists(mask_path):
            print(f"  Warning: Mask not found for {pid} - will use full volume")
            mask_path = None
        
        patients.append({
            "patient_id": pid,
            "cohort": cohort,
            "gt_path": gt_path,
            "pred_path": pred_path,
            "mask_path": mask_path
        })
    
    print(f"Found {len(patients)} patients with complete data\n")
    return patients


def compute_patient_metrics(gt_path: str, pred_path: str, mask_path: str, 
                           metrics: ImageMetrics) -> Dict[str, float]:
    """
    Compute MAE, PSNR, and MS-SSIM for a single patient.
    
    Parameters
    ----------
    gt_path : str
        Path to ground truth CT volume
    pred_path : str
        Path to predicted sCT volume
    mask_path : str
        Path to body mask (can be None)
    metrics : ImageMetrics
        Initialized metrics object
    
    Returns
    -------
    results : dict
        Dictionary with 'MAE', 'PSNR', 'MS-SSIM' keys
    """
    # Load volumes (all in HU space)
    gt = sitk.GetArrayFromImage(sitk.ReadImage(gt_path))
    pred = sitk.GetArrayFromImage(sitk.ReadImage(pred_path))
    
    # Load or create mask
    if mask_path is not None and os.path.exists(mask_path):
        mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_path)).astype(np.float32)
    else:
        # Use full volume if no mask available
        mask = np.ones_like(gt, dtype=np.float32)
    
    # Compute metrics (all operate on HU-space volumes)
    mae_value = metrics.mae(gt, pred, mask)
    psnr_value = metrics.psnr(gt, pred, mask, use_population_range=True)
    
    # MS-SSIM returns (full, masked) - we use the masked value
    _, ms_ssim_value = metrics.ms_ssim(gt, pred, mask)
    
    return {
        'MAE': float(mae_value),
        'PSNR': float(psnr_value),
        'MS-SSIM': float(ms_ssim_value)
    }


def select_representative_cases(results: pd.DataFrame, metric: str = 'MAE') -> Dict[str, Dict]:
    """
    Select best, average, and worst cases based on specified metric.
    
    Parameters
    ----------
    results : pd.DataFrame
        DataFrame with patient_id and metric columns
    metric : str
        Metric name to use for selection (default: 'MAE')
    
    Returns
    -------
    cases : dict
        Dictionary with 'best', 'average', 'worst' keys
    """
    # For MAE: lower is better
    # For PSNR: higher is better
    # For MS-SSIM: higher is better
    
    metric_values = results[metric].values
    mean_value = np.mean(metric_values)
    
    # Best case: lowest MAE (minimum reconstruction error)
    best_idx = results[metric].idxmin()
    
    # Average case: closest to mean MAE
    diff_from_mean = np.abs(metric_values - mean_value)
    avg_idx = results.iloc[np.argmin(diff_from_mean)].name
    
    # Worst case: highest MAE (maximum reconstruction error)
    worst_idx = results[metric].idxmax()
    
    cases = {
        'best': results.loc[best_idx].to_dict(),
        'average': results.loc[avg_idx].to_dict(),
        'worst': results.loc[worst_idx].to_dict()
    }
    
    return cases


def plot_metric_histogram(values: np.ndarray, metric_name: str, 
                         output_path: str, xlabel: str = None):
    """
    Generate histogram with mean and std annotations.
    
    Parameters
    ----------
    values : np.ndarray
        Array of metric values
    metric_name : str
        Name of the metric for title
    output_path : str
        Path to save the histogram
    xlabel : str, optional
        Custom x-axis label
    """
    mean_val = np.mean(values)
    std_val = np.std(values)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot histogram
    n, bins, patches = ax.hist(values, bins=20, alpha=0.7, color='skyblue', 
                               edgecolor='black', linewidth=1.2)
    
    # Add vertical line at mean
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
               label=f'Mean = {mean_val:.2f}')
    
    # Add shaded region for mean ± std
    ax.axvspan(mean_val - std_val, mean_val + std_val, 
              alpha=0.2, color='gray', label=f'Std = {std_val:.2f}')
    
    # Labels and title
    ax.set_xlabel(xlabel if xlabel else metric_name, fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'{metric_name} Distribution Across Test Set', fontsize=14, fontweight='bold')
    
    # Add text annotation with statistics
    textstr = f'Mean: {mean_val:.2f}\nStd: {std_val:.2f}\nN: {len(values)}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.98, 0.97, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', horizontalalignment='right', bbox=props)
    
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved histogram: {output_path}")


def main():
    """
    Main evaluation pipeline.
    """
    print("="*60)
    print("Offline Evaluation of sCT Reconstruction Metrics")
    print("="*60)
    
    # Validation checks
    split_file = os.path.join(save_dir, "test_split.json")
    if not os.path.exists(split_file):
        print(f"ERROR: Test split file not found: {split_file}")
        print("Please run Train_condition.py first to generate test split.")
        return
    
    if not os.path.exists(output_dir):
        print(f"ERROR: Output directory not found: {output_dir}")
        print("Please run Test_condition.py first to generate sCT volumes.")
        return
    
    # --------------------------
    # Load test patients
    # --------------------------
    print("\n[1/5] Loading test patient information...")
    patients = load_test_patients(split_file, dataset_root, output_dir)
    
    if len(patients) == 0:
        print("ERROR: No patients with complete data found.")
        print("Please verify that Test_condition.py completed successfully.")
        return
    
    # --------------------------
    # Compute metrics for all patients
    # --------------------------
    print("[2/5] Computing metrics for all patients...")
    results_list = []
    
    for patient in tqdm(patients, desc="Processing patients"):
        try:
            patient_metrics = compute_patient_metrics(
                patient["gt_path"],
                patient["pred_path"],
                patient["mask_path"],
                metrics
            )
            
            # Add patient ID to results
            patient_metrics["patient_id"] = patient["patient_id"]
            results_list.append(patient_metrics)
            
        except Exception as e:
            print(f"\n  ERROR processing patient {patient['patient_id']}: {e}")
            continue
    
    if len(results_list) == 0:
        print("ERROR: No metrics computed successfully.")
        return
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results_list)
    # Reorder columns to have patient_id first
    cols = ['patient_id', 'MAE', 'PSNR', 'MS-SSIM']
    results_df = results_df[cols]
    
    # --------------------------
    # Statistical reporting
    # --------------------------
    print("\n[3/5] Computing statistics and saving results...")
    
    # Compute mean and std for each metric
    mae_mean = results_df['MAE'].mean()
    mae_std = results_df['MAE'].std()
    psnr_mean = results_df['PSNR'].mean()
    psnr_std = results_df['PSNR'].std()
    ssim_mean = results_df['MS-SSIM'].mean()
    ssim_std = results_df['MS-SSIM'].std()
    
    # Save per-patient metrics to CSV
    csv_path = os.path.join(output_dir, "per_patient_metrics.csv")
    results_df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"  Saved per-patient metrics: {csv_path}")
    
    # Save summary statistics to text file
    summary_path = os.path.join(output_dir, "evaluation_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("sCT Reconstruction Metrics - Summary Statistics\n")
        f.write("="*60 + "\n\n")
        f.write(f"Number of patients: {len(results_df)}\n\n")
        f.write("Mean ± Standard Deviation:\n")
        f.write("-"*60 + "\n")
        f.write(f"MAE:     {mae_mean:.2f} ± {mae_std:.2f} HU\n")
        f.write(f"PSNR:    {psnr_mean:.2f} ± {psnr_std:.2f} dB\n")
        f.write(f"MS-SSIM: {ssim_mean:.4f} ± {ssim_std:.4f}\n")
        f.write("="*60 + "\n")
    print(f"  Saved summary statistics: {summary_path}")
    
    # Print summary to console
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Number of patients: {len(results_df)}")
    print()
    print("Mean ± Standard Deviation:")
    print("-"*60)
    print(f"MAE:     {mae_mean:.2f} ± {mae_std:.2f} HU")
    print(f"PSNR:    {psnr_mean:.2f} ± {psnr_std:.2f} dB")
    print(f"MS-SSIM: {ssim_mean:.4f} ± {ssim_std:.4f}")
    print("="*60)
    
    # --------------------------
    # Select representative cases
    # --------------------------
    print("\n[4/5] Selecting representative cases (based on MAE)...")
    rep_cases = select_representative_cases(results_df, metric='MAE')
    
    # Save representative cases to text file
    rep_cases_path = os.path.join(output_dir, "representative_cases.txt")
    with open(rep_cases_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("Representative Cases (selected by MAE)\n")
        f.write("="*60 + "\n\n")
        
        for case_type in ['best', 'average', 'worst']:
            case = rep_cases[case_type]
            f.write(f"{case_type.upper()} CASE:\n")
            f.write(f"  Patient ID: {case['patient_id']}\n")
            f.write(f"  MAE:        {case['MAE']:.2f} HU\n")
            f.write(f"  PSNR:       {case['PSNR']:.2f} dB\n")
            f.write(f"  MS-SSIM:    {case['MS-SSIM']:.4f}\n")
            f.write("\n")
    print(f"  Saved representative cases: {rep_cases_path}")
    
    # Print representative cases to console
    print("\n" + "="*60)
    print("REPRESENTATIVE CASES (selected by MAE)")
    print("="*60)
    for case_type in ['best', 'average', 'worst']:
        case = rep_cases[case_type]
        print(f"\n{case_type.upper()} CASE:")
        print(f"  Patient ID: {case['patient_id']}")
        print(f"  MAE:        {case['MAE']:.2f} HU")
        print(f"  PSNR:       {case['PSNR']:.2f} dB")
        print(f"  MS-SSIM:    {case['MS-SSIM']:.4f}")
    print("="*60)
    
    # --------------------------
    # Generate histograms
    # --------------------------
    print("\n[5/5] Generating histograms...")
    
    # MAE histogram
    plot_metric_histogram(
        results_df['MAE'].values,
        'MAE',
        os.path.join(output_dir, 'histogram_mae.png'),
        xlabel='MAE (HU)'
    )
    
    # PSNR histogram
    plot_metric_histogram(
        results_df['PSNR'].values,
        'PSNR',
        os.path.join(output_dir, 'histogram_psnr.png'),
        xlabel='PSNR (dB)'
    )
    
    # MS-SSIM histogram
    plot_metric_histogram(
        results_df['MS-SSIM'].values,
        'MS-SSIM',
        os.path.join(output_dir, 'histogram_ssim.png'),
        xlabel='MS-SSIM'
    )
    
    # --------------------------
    # Final summary
    # --------------------------
    print("\n" + "="*60)
    print("EVALUATION COMPLETE!")
    print("="*60)
    print(f"\nAll results saved to: {output_dir}")
    print("\nGenerated files:")
    print(f"  - per_patient_metrics.csv")
    print(f"  - evaluation_summary.txt")
    print(f"  - representative_cases.txt")
    print(f"  - histogram_mae.png")
    print(f"  - histogram_psnr.png")
    print(f"  - histogram_ssim.png")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
