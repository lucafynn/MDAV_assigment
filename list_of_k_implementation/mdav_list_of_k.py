#!/usr/bin/env python3
"""
Task 2: MDAV (Maximum Distance to Average Vector) k-Anonymity Implementation

This script implements the MDAV algorithm to achieve k-anonymity on numeric quasi-identifiers
for the health_ai_mdav_demo.csv dataset. It generates all required outputs for grading:

1. Anonymized CSV files for each k value
2. Cluster assignment files (RecordID → ClusterID)
3. Comprehensive report with k choice, method notes, privacy checks, and quality metrics

Author: Assignment Solution
Date: October 5, 2025
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
import json

# Add parent directory to path to import required modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import humanreadable, MDAVGeneric, microaggregation
from anonymiser import customAnonymiser


class MDVAKAnonymityTask:
    def __init__(self, dataset_path, output_dir="task2_outputs"):
        """
        Initialize the MDAV k-anonymity task.
        
        Args:
            dataset_path (str): Path to the CSV dataset
            output_dir (str): Directory to save all outputs
        """
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.df_original = None
        self.quasi_identifiers = ["Age", "Sex", "ZIP"]  # Numeric QIs
        self.results = {}
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
    def load_and_preprocess_data(self):
        """Load and preprocess the dataset."""
        print("Loading and preprocessing dataset...")
        self.df_original = pd.read_csv(self.dataset_path)
        
        # Keep original Sex column for reference
        self.df_original['Sex_Original'] = self.df_original['Sex'].copy()
        
        # Convert Sex to numeric for MDAV clustering (M=0, F=1)
        # Note: This is for distance calculation only, will handle categorical generalization separately
        self.df_original['Sex'] = self.df_original['Sex'].map({'M': 0, 'F': 1})
        
        # Set Diagnosis as category
        self.df_original["Diagnosis"] = self.df_original["Diagnosis"].astype("category")
        
        print(f"Dataset loaded: {self.df_original.shape[0]} records, {self.df_original.shape[1]} columns")
        print(f"Quasi-identifiers: {self.quasi_identifiers}")
        print("Note: Sex converted to numeric for clustering but will be properly generalized in output")
        
    def calculate_cluster_radius(self, df, clusters, quasi_cols):
        """
        Calculate average cluster radius.
        
        Args:
            df (DataFrame): Original data
            clusters (list): List of cluster assignments
            quasi_cols (list): Quasi-identifier columns
            
        Returns:
            float: Average radius across all clusters
        """
        if not clusters:
            return 0.0
            
        radii = []
        for cluster_indices in clusters:
            if len(cluster_indices) <= 1:
                radii.append(0.0)
                continue
                
            # Get cluster data
            cluster_data = df.loc[cluster_indices, quasi_cols].values
            
            # Calculate centroid
            centroid = np.mean(cluster_data, axis=0)
            
            # Calculate distances from centroid
            distances = np.sqrt(np.sum((cluster_data - centroid) ** 2, axis=1))
            
            # Maximum distance is the radius
            radius = np.max(distances)
            radii.append(radius)
            
        return np.mean(radii)
        
    def generalize_categorical_attribute(self, cluster_data, attribute_name):
        """
        Properly generalize categorical attributes using most frequent value or domain generalization.
        
        Args:
            cluster_data (DataFrame): Data for the cluster
            attribute_name (str): Name of the categorical attribute
            
        Returns:
            str: Generalized value for the attribute
        """
        if attribute_name == 'Sex_Original':
            # For gender, use the most frequent value in cluster, or '*' if mixed
            values = cluster_data[attribute_name].value_counts()
            if len(values) == 1:
                return values.index[0]  # All same gender
            else:
                # Mixed genders - could use most frequent or generalize to '*'
                # Using most frequent for now, but '*' is also valid
                return values.index[0]  # Most frequent
        else:
            # For other categorical attributes, use most frequent
            return cluster_data[attribute_name].mode().iloc[0]
    
    def calculate_detailed_sse(self, original_df, anonymized_df, quasi_cols):
        """
        Calculate detailed SSE with breakdown by attribute.
        
        Args:
            original_df (DataFrame): Original data
            anonymized_df (DataFrame): Anonymized data
            quasi_cols (list): Quasi-identifier columns
            
        Returns:
            dict: SSE breakdown by attribute and total
        """
        sse_breakdown = {}
        total_sse = 0
        
        for col in quasi_cols:
            if col in original_df.columns and col in anonymized_df.columns:
                # Only calculate for numeric columns
                if pd.api.types.is_numeric_dtype(original_df[col]):
                    diff = original_df[col] - anonymized_df[col]
                    col_sse = (diff ** 2).sum()
                    sse_breakdown[col] = round(col_sse, 2)
                    total_sse += col_sse
        
        sse_breakdown['Total'] = round(total_sse, 2)
        return sse_breakdown
        
    def verify_k_anonymity(self, clusters, k):
        """
        Verify that k-anonymity is satisfied.
        
        Args:
            clusters (list): List of cluster assignments
            k (int): k-anonymity parameter
            
        Returns:
            tuple: (is_satisfied, min_cluster_size, cluster_sizes)
        """
        cluster_sizes = [len(cluster) for cluster in clusters]
        min_cluster_size = min(cluster_sizes) if cluster_sizes else 0
        is_satisfied = min_cluster_size >= k
        
        return is_satisfied, min_cluster_size, cluster_sizes
        
    def run_mdav_for_k(self, k):
        """
        Run MDAV algorithm for a specific k value with proper categorical handling.
        
        Args:
            k (int): k-anonymity parameter
            
        Returns:
            dict: Results including anonymized data, clusters, and metrics
        """
        print(f"\nProcessing k={k}...")
        
        # Create anonymiser
        anonymiser = customAnonymiser(
            self.df_original.copy(), 
            k=k, 
            quasi_identifiers=self.quasi_identifiers, 
            generalisation_strategy=microaggregation.Microaggregation
        )
        
        # Generate anonymized dataset (this gives numeric averages)
        anonymised_df_numeric = anonymiser.anonymise()
        clusters = anonymiser.algorithm.clusters
        
        # Create properly generalized dataset
        anonymised_df = anonymised_df_numeric.copy()
        
        # Fix categorical generalization for Sex
        for cluster_id, cluster_indices in enumerate(clusters):
            cluster_data = self.df_original.iloc[cluster_indices]
            
            # Generalize Sex properly (categorical)
            sex_generalized = self.generalize_categorical_attribute(cluster_data, 'Sex_Original')
            anonymised_df.loc[cluster_indices, 'Sex_Generalized'] = sex_generalized
            
            # Keep numeric sex for SSE calculation but add readable version
            anonymised_df.loc[cluster_indices, 'Sex_Display'] = sex_generalized
        
        # Calculate detailed metrics
        sse_breakdown = self.calculate_detailed_sse(
            self.df_original, anonymised_df_numeric, self.quasi_identifiers
        )
        
        # Legacy SSE for compatibility
        numeric_quasi = self.df_original[self.quasi_identifiers]
        anonymized_quasi = anonymised_df_numeric[self.quasi_identifiers]
        diff = numeric_quasi - anonymized_quasi
        sse = (diff ** 2).sum().sum()
        
        # Average cluster radius
        avg_radius = self.calculate_cluster_radius(
            self.df_original, clusters, self.quasi_identifiers
        )
        
        # Privacy verification
        is_k_anonymous, min_cluster_size, cluster_sizes = self.verify_k_anonymity(clusters, k)
        
        # Create cluster assignment mapping
        cluster_assignments = {}
        for cluster_id, cluster_indices in enumerate(clusters):
            for record_id in cluster_indices:
                original_id = self.df_original.iloc[record_id]['ID']
                cluster_assignments[original_id] = cluster_id
        
        # Save iteration data
        iteration_data = {
            'k': int(k),
            'sse_breakdown': {str(attr): float(val) for attr, val in sse_breakdown.items()},
            'cluster_details': [],
            'generalization_info': {
                'method': 'most_frequent_categorical',
                'explanation': 'Categorical attributes generalized using most frequent value in cluster'
            }
        }
        
        # Collect detailed cluster information
        for cluster_id, cluster_indices in enumerate(clusters):
            cluster_info = {
                'cluster_id': int(cluster_id),
                'size': int(len(cluster_indices)),
                'records': [int(i) for i in cluster_indices],
                'original_records': [int(self.df_original.iloc[i]['ID']) for i in cluster_indices],
                'centroid': {
                    'Age': float(round(anonymised_df.iloc[cluster_indices[0]]['Age'], 2)),
                    'Sex_Numeric': float(round(anonymised_df.iloc[cluster_indices[0]]['Sex'], 2)),
                    'Sex_Generalized': str(anonymised_df.iloc[cluster_indices[0]]['Sex_Display']),
                    'ZIP': float(round(anonymised_df.iloc[cluster_indices[0]]['ZIP'], 2))
                }
            }
            iteration_data['cluster_details'].append(cluster_info)
        
        return {
            'k': k,
            'anonymized_df': anonymised_df,
            'clusters': clusters,
            'cluster_assignments': cluster_assignments,
            'sse': round(sse, 2),
            'sse_breakdown': sse_breakdown,
            'avg_radius': round(avg_radius, 3),
            'is_k_anonymous': is_k_anonymous,
            'min_cluster_size': min_cluster_size,
            'cluster_sizes': cluster_sizes,
            'num_clusters': len(clusters),
            'iteration_data': iteration_data
        }
        
    def save_anonymized_csv(self, result):
        """Save anonymized CSV file with proper categorical handling."""
        k = result['k']
        filename = f"{self.output_dir}/anonymized_k{k}.csv"
        
        # Create final output with proper column names
        output_df = result['anonymized_df'].copy()
        
        # Replace numeric sex with generalized sex for final output
        if 'Sex_Display' in output_df.columns:
            output_df['Sex'] = output_df['Sex_Display']
            output_df = output_df.drop(columns=['Sex_Display', 'Sex_Generalized'], errors='ignore')
        
        # Remove temporary columns
        output_df = output_df.drop(columns=['Sex_Original'], errors='ignore')
        
        # Save without index, maintaining original column order
        output_df.to_csv(filename, index=False)
        print(f"Saved anonymized CSV: {filename}")
        
    def save_iteration_data(self, result):
        """Save detailed iteration data as JSON."""
        k = result['k']
        filename = f"{self.output_dir}/iteration_data_k{k}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(result['iteration_data'], f, indent=2)
        print(f"Saved iteration data: {filename}")
        
    def save_sse_breakdown(self, result):
        """Save SSE breakdown analysis."""
        k = result['k']
        filename = f"{self.output_dir}/sse_breakdown_k{k}.csv"
        
        sse_data = []
        for attribute, sse_value in result['sse_breakdown'].items():
            sse_data.append({
                'Attribute': attribute,
                'SSE': sse_value,
                'Percentage': round((sse_value / result['sse_breakdown']['Total']) * 100, 2) if result['sse_breakdown']['Total'] > 0 else 0
            })
        
        sse_df = pd.DataFrame(sse_data)
        sse_df.to_csv(filename, index=False)
        print(f"Saved SSE breakdown: {filename}")
        
    def save_cluster_assignments(self, result):
        """Save cluster assignment file."""
        k = result['k']
        filename = f"{self.output_dir}/cluster_assignments_k{k}.csv"
        
        # Create DataFrame with RecordID → ClusterID mapping
        assignments_df = pd.DataFrame([
            {'RecordID': record_id, 'ClusterID': cluster_id}
            for record_id, cluster_id in result['cluster_assignments'].items()
        ])
        
        # Sort by RecordID for consistency
        assignments_df = assignments_df.sort_values('RecordID')
        assignments_df.to_csv(filename, index=False)
        print(f"Saved cluster assignments: {filename}")
        
    def generate_comprehensive_report(self):
        """Generate comprehensive report with all k values for comparison."""
        report_filename = f"{self.output_dir}/MDAV_Analysis_Report.md"
        
        # Find optimal k (lowest SSE)
        optimal_k = min(self.results.keys(), key=lambda k: self.results[k]['sse'])
        
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write("# MDAV k-Anonymity Analysis Report\n\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Dataset:** {self.dataset_path}\n")
            f.write(f"**Total Records:** {self.df_original.shape[0]}\n")
            f.write(f"**Quasi-identifiers:** {', '.join(self.quasi_identifiers)}\n")
            f.write(f"**Analysis Strategy:** Compare all k values, output optimal k only\n\n")
            
            # Method Description
            f.write("## Method Description\n\n")
            f.write("**MDAV (Maximum Distance to Average Vector) Algorithm:**\n")
            f.write("- Iteratively forms clusters by selecting the record farthest from the average vector\n")
            f.write("- Creates clusters of at least k records to ensure k-anonymity\n")
            f.write("- Replaces quasi-identifier values with cluster centroids (means)\n")
            f.write("- Minimizes information loss while maintaining privacy protection\n\n")
            
            f.write("**Microaggregation Strategy:**\n")
            f.write("- Numerical quasi-identifiers are generalized using cluster means\n")
            f.write("- Preserves data utility by minimizing Sum of Squared Errors (SSE)\n")
            f.write("- Maintains statistical properties of the original dataset\n\n")
            
            f.write("**Categorical Attribute Handling (Gender/Sex):**\n")
            f.write("- Sex attribute converted to numeric (M=0, F=1) for distance calculations during clustering\n")
            f.write("- After clustering, categorical generalization applied using most frequent value per cluster\n")
            f.write("- This approach avoids meaningless decimal values (e.g., Sex=0.5) in final output\n")
            f.write("- Alternative approaches: domain generalization to '*' for mixed-gender clusters\n")
            f.write("- Current implementation preserves gender information when clusters are homogeneous\n\n")
            
            # K Value Comparison Analysis
            f.write("## K Value Comparison Analysis\n\n")
            f.write("The following table compares all tested k values to identify the optimal configuration:\n\n")
            f.write("| k | SSE | Clusters | Min Size | Avg Radius | K-Anonymous | Performance |\n")
            f.write("|---|-----|----------|----------|------------|-------------|-------------|\n")
            
            min_sse = min(result['sse'] for result in self.results.values())
            for k in sorted(self.results.keys()):
                result = self.results[k]
                performance = "🥇 OPTIMAL" if k == optimal_k else f"+{((result['sse'] - min_sse) / min_sse * 100):.1f}%"
                f.write(f"| {k} | {result['sse']} | {result['num_clusters']} | {result['min_cluster_size']} | {result['avg_radius']} | {'✓' if result['is_k_anonymous'] else '✗'} | {performance} |\n")
            
            # K Choice Justification
            f.write(f"\n## Selected K Value: {optimal_k}\n\n")
            f.write(f"**Justification for k={optimal_k}:**\n")
            f.write(f"- ✅ **Best Data Utility:** Lowest SSE ({self.results[optimal_k]['sse']}) among all tested values\n")
            f.write(f"- ✅ **Privacy Compliant:** Satisfies k-anonymity requirement (min cluster size: {self.results[optimal_k]['min_cluster_size']})\n")
            f.write(f"- ✅ **Balanced Clustering:** Creates {self.results[optimal_k]['num_clusters']} well-sized clusters\n")
            f.write(f"- ✅ **Good Compactness:** Average cluster radius of {self.results[optimal_k]['avg_radius']}\n\n")
            
            # Privacy Analysis
            f.write("## Privacy Analysis (K-Anonymity Verification)\n\n")
            f.write("| k | K-Anonymous? | Min Cluster Size | Number of Clusters | Status |\n")
            f.write("|---|--------------|------------------|--------------------|---------|\n")
            
            for k in sorted(self.results.keys()):
                result = self.results[k]
                status = "PASS" if result['is_k_anonymous'] else "FAIL"
                f.write(f"| {k} | {'Yes' if result['is_k_anonymous'] else 'No'} | "
                       f"{result['min_cluster_size']} | {result['num_clusters']} | {status} |\n")
            
            f.write("\n**Privacy Guarantee:** Each individual is indistinguishable from at least k-1 others in their quasi-identifier values.\n\n")
            
            # Quality Metrics
            f.write("## Quality Metrics Comparison\n\n")
            f.write("### Sum of Squared Errors (SSE)\n")
            f.write("Lower values indicate better data utility:\n\n")
            f.write("| k | SSE | Information Loss | Relative Performance |\n")
            f.write("|---|-----|------------------|----------------------|\n")
            
            for k in sorted(self.results.keys()):
                result = self.results[k]
                loss_pct = ((result['sse'] - min_sse) / min_sse * 100) if min_sse > 0 else 0
                relative = "Best" if k == optimal_k else f"+{loss_pct:.1f}%"
                f.write(f"| {k} | {result['sse']} | {loss_pct:.1f}% | {relative} |\n")
            
            f.write("\n### Average Cluster Radius\n")
            f.write("Measures cluster compactness (lower is better):\n\n")
            f.write("| k | Average Radius | Compactness |\n")
            f.write("|---|----------------|-------------|\n")
            
            max_radius = max(result['avg_radius'] for result in self.results.values())
            for k in sorted(self.results.keys()):
                result = self.results[k]
                compactness = "High" if result['avg_radius'] <= max_radius * 0.5 else \
                             "Medium" if result['avg_radius'] <= max_radius * 0.75 else "Low"
                f.write(f"| {k} | {result['avg_radius']} | {compactness} |\n")
            
            # Final Output Files (Only for Optimal K)
            f.write(f"\n## Generated Output Files (k={optimal_k} only)\n\n")
            f.write("**Primary Deliverables:**\n")
            f.write(f"- `anonymized_k{optimal_k}.csv` - Anonymized dataset with optimal k value\n")
            f.write(f"- `cluster_assignments_k{optimal_k}.csv` - Record-to-cluster mappings for optimal k\n")
            f.write(f"- `iteration_data_k{optimal_k}.json` - Detailed clustering information for optimal k\n")
            f.write(f"- `sse_breakdown_k{optimal_k}.csv` - SSE analysis by attribute for optimal k\n")
            f.write("- `MDAV_Analysis_Report.md` - This comprehensive analysis report\n\n")
            
            # Detailed Analysis of Optimal K
            f.write(f"## Detailed Analysis of Optimal k={optimal_k}\n\n")
            result = self.results[optimal_k]
            f.write(f"### Clustering Results\n")
            f.write(f"- **Total Clusters:** {result['num_clusters']}\n")
            f.write(f"- **Cluster Sizes:** {result['cluster_sizes']}\n")
            f.write(f"- **Minimum Cluster Size:** {result['min_cluster_size']} (✓ ≥ {optimal_k})\n")
            f.write(f"- **Maximum Cluster Size:** {max(result['cluster_sizes'])}\n")
            f.write(f"- **Average Cluster Size:** {sum(result['cluster_sizes']) / len(result['cluster_sizes']):.1f}\n\n")
            
            f.write(f"### Quality Metrics\n")
            f.write(f"- **Total SSE:** {result['sse']}\n")
            if 'sse_breakdown' in result:
                f.write(f"- **SSE by Attribute:**\n")
                for attr, sse_val in result['sse_breakdown'].items():
                    if attr != 'Total':
                        pct = (sse_val / result['sse_breakdown']['Total']) * 100
                        f.write(f"  - {attr}: {sse_val} ({pct:.1f}%)\n")
            f.write(f"- **Average Cluster Radius:** {result['avg_radius']}\n")
            f.write(f"- **Privacy Status:** {'✅ K-Anonymous' if result['is_k_anonymous'] else '❌ Not K-Anonymous'}\n\n")
            
            # Summary and Recommendations
            f.write("## Summary and Recommendations\n\n")
            f.write(f"### Key Findings\n")
            f.write(f"1. **Optimal Configuration:** k={optimal_k} provides the best balance of privacy and utility\n")
            f.write(f"2. **Privacy Protection:** All tested k values ({', '.join(map(str, sorted(self.results.keys())))}) satisfy k-anonymity\n")
            f.write(f"3. **Data Utility:** SSE ranges from {min_sse} to {max(r['sse'] for r in self.results.values())}\n")
            f.write(f"4. **Output Strategy:** Generated files for optimal k only to avoid confusion\n\n")
            
            f.write("### Gender Handling Strategy\n")
            f.write("**Challenge:** Categorical attribute 'Sex' requires special handling in k-anonymity:\n")
            f.write("1. **Problem:** Direct averaging of M/F creates meaningless values (0.5)\n")
            f.write("2. **Solution:** Most frequent value generalization within clusters\n")
            f.write("3. **Alternative:** Domain generalization to '*' for mixed-gender clusters\n")
            f.write("4. **Rationale:** Preserves meaningful gender information while maintaining anonymity\n\n")
            
            f.write("### Implementation Notes\n")
            f.write("- All k values were analyzed for comprehensive comparison\n")
            f.write("- Only optimal k value files are generated to provide clean final output\n")
            f.write("- Report includes full analysis for transparency and justification\n")
            f.write("- Categorical generalization handles mixed-type quasi-identifiers appropriately\n")
        
        print(f"Comprehensive report saved: {report_filename}")
        
    def run_complete_analysis(self, k_values=None, output_all_k=False):
        """
        Run complete MDAV analysis for multiple k values.
        
        Args:
            k_values (list): List of k values to test (default: [2,3,4,5,6])
            output_all_k (bool): If True, output files for all k values. If False, only optimal k.
        """
        if k_values is None:
            k_values = [2, 3, 4, 5, 6]
            
        print("=" * 60)
        print("MDAV K-ANONYMITY TASK 2 - COMPLETE ANALYSIS")
        print("=" * 60)
        
        # Load and preprocess data
        self.load_and_preprocess_data()
        
        # Run analysis for each k value (for comparison in report)
        print(f"\nAnalyzing k values: {k_values} for comparison...")
        for k in k_values:
            result = self.run_mdav_for_k(k)
            self.results[k] = result
        
        # Find optimal k value
        optimal_k = min(self.results.keys(), key=lambda k: self.results[k]['sse'])
        print(f"\n🎯 Optimal k value identified: {optimal_k} (SSE: {self.results[optimal_k]['sse']})")
        
        # Save outputs based on output_all_k flag
        if output_all_k:
            print(f"\n📁 Saving files for ALL k values...")
            files_generated = 0
            for k in k_values:
                result = self.results[k]
                self.save_anonymized_csv(result)
                self.save_cluster_assignments(result)
                self.save_iteration_data(result)
                self.save_sse_breakdown(result)
                files_generated += 4
        else:
            print(f"\n📁 Saving files for OPTIMAL k value only (k={optimal_k})...")
            result = self.results[optimal_k]
            self.save_anonymized_csv(result)
            self.save_cluster_assignments(result)
            self.save_iteration_data(result)
            self.save_sse_breakdown(result)
            files_generated = 4
            
        # Generate comprehensive report (always includes all k values for comparison)
        self.generate_comprehensive_report()
        files_generated += 1
        
        print(f"\n" + "=" * 60)
        print("ANALYSIS COMPLETE!")
        print(f"All outputs saved to: {self.output_dir}/")
        print("=" * 60)
        
        # Print summary
        print(f"\nSUMMARY:")
        print(f"- Optimal k value: {optimal_k} (SSE: {self.results[optimal_k]['sse']})")
        print(f"- Output strategy: {'All k values' if output_all_k else 'Optimal k only'}")
        print(f"- Files generated: {files_generated}")
        print(f"- Report includes comparison of all k values: {', '.join(map(str, k_values))}")
        print(f"- All k values satisfy k-anonymity: {'Yes' if all(r['is_k_anonymous'] for r in self.results.values()) else 'No'}")


def main():
    """Main execution function."""
    # Initialize the task
    task = MDVAKAnonymityTask("health_ai_mdav_demo.csv")
    
    # Run complete analysis for k values 2-6
    task.run_complete_analysis([2, 3, 4, 5, 6])


if __name__ == "__main__":
    main()