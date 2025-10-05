# MDAV k-Anonymity Analysis Report

**Date:** 2025-10-05 14:03:58
**Dataset:** health_ai_mdav_demo.csv
**Total Records:** 40
**Quasi-identifiers:** Age, Sex, ZIP
**Analysis Strategy:** Compare all k values, output optimal k only

## Method Description

**MDAV (Maximum Distance to Average Vector) Algorithm:**
- Iteratively forms clusters by selecting the record farthest from the average vector
- Creates clusters of at least k records to ensure k-anonymity
- Replaces quasi-identifier values with cluster centroids (means)
- Minimizes information loss while maintaining privacy protection

**Microaggregation Strategy:**
- Numerical quasi-identifiers are generalized using cluster means
- Preserves data utility by minimizing Sum of Squared Errors (SSE)
- Maintains statistical properties of the original dataset

**Categorical Attribute Handling (Gender/Sex):**
- Sex attribute converted to numeric (M=0, F=1) for distance calculations during clustering
- After clustering, categorical generalization applied using most frequent value per cluster
- This approach avoids meaningless decimal values (e.g., Sex=0.5) in final output
- Alternative approaches: domain generalization to '*' for mixed-gender clusters
- Current implementation preserves gender information when clusters are homogeneous

## K Value Comparison Analysis

The following table compares all tested k values to identify the optimal configuration:

| k | SSE | Clusters | Min Size | Avg Radius | K-Anonymous | Performance |
|---|-----|----------|----------|------------|-------------|-------------|
| 2 | 21912.0 | 20 | 2 | 2.87 | ✓ | +57.8% |
| 3 | 14549.67 | 13 | 3 | 6.23 | ✓ | +4.8% |
| 4 | 13884.5 | 10 | 4 | 9.459 | ✓ | 🥇 OPTIMAL |
| 5 | 18721.2 | 8 | 5 | 11.791 | ✓ | +34.8% |
| 6 | 14208.13 | 6 | 6 | 15.921 | ✓ | +2.3% |

## Selected K Value: 4

**Justification for k=4:**
- ✅ **Best Data Utility:** Lowest SSE (13884.5) among all tested values
- ✅ **Privacy Compliant:** Satisfies k-anonymity requirement (min cluster size: 4)
- ✅ **Balanced Clustering:** Creates 10 well-sized clusters
- ✅ **Good Compactness:** Average cluster radius of 9.459

## Privacy Analysis (K-Anonymity Verification)

| k | K-Anonymous? | Min Cluster Size | Number of Clusters | Status |
|---|--------------|------------------|--------------------|---------|
| 2 | Yes | 2 | 20 | PASS |
| 3 | Yes | 3 | 13 | PASS |
| 4 | Yes | 4 | 10 | PASS |
| 5 | Yes | 5 | 8 | PASS |
| 6 | Yes | 6 | 6 | PASS |

**Privacy Guarantee:** Each individual is indistinguishable from at least k-1 others in their quasi-identifier values.

## Quality Metrics Comparison

### Sum of Squared Errors (SSE)
Lower values indicate better data utility:

| k | SSE | Information Loss | Relative Performance |
|---|-----|------------------|----------------------|
| 2 | 21912.0 | 57.8% | +57.8% |
| 3 | 14549.67 | 4.8% | +4.8% |
| 4 | 13884.5 | 0.0% | Best |
| 5 | 18721.2 | 34.8% | +34.8% |
| 6 | 14208.13 | 2.3% | +2.3% |

### Average Cluster Radius
Measures cluster compactness (lower is better):

| k | Average Radius | Compactness |
|---|----------------|-------------|
| 2 | 2.87 | High |
| 3 | 6.23 | High |
| 4 | 9.459 | Medium |
| 5 | 11.791 | Medium |
| 6 | 15.921 | Low |

## Generated Output Files (k=4 only)

**Primary Deliverables:**
- `anonymized_k4.csv` - Anonymized dataset with optimal k value
- `cluster_assignments_k4.csv` - Record-to-cluster mappings for optimal k
- `iteration_data_k4.json` - Detailed clustering information for optimal k
- `sse_breakdown_k4.csv` - SSE analysis by attribute for optimal k
- `MDAV_Analysis_Report.md` - This comprehensive analysis report

## Detailed Analysis of Optimal k=4

### Clustering Results
- **Total Clusters:** 10
- **Cluster Sizes:** [4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
- **Minimum Cluster Size:** 4 (✓ ≥ 4)
- **Maximum Cluster Size:** 4
- **Average Cluster Size:** 4.0

### Quality Metrics
- **Total SSE:** 13884.5
- **SSE by Attribute:**
  - Age: 13806.5 (99.4%)
  - Sex: 16.0 (0.1%)
  - ZIP: 62.0 (0.4%)
- **Average Cluster Radius:** 9.459
- **Privacy Status:** ✅ K-Anonymous

## Summary and Recommendations

### Key Findings
1. **Optimal Configuration:** k=4 provides the best balance of privacy and utility
2. **Privacy Protection:** All tested k values (2, 3, 4, 5, 6) satisfy k-anonymity
3. **Data Utility:** SSE ranges from 13884.5 to 21912.0
4. **Output Strategy:** Generated files for optimal k only to avoid confusion

### Gender Handling Strategy
**Challenge:** Categorical attribute 'Sex' requires special handling in k-anonymity:
1. **Problem:** Direct averaging of M/F creates meaningless values (0.5)
2. **Solution:** Most frequent value generalization within clusters
3. **Alternative:** Domain generalization to '*' for mixed-gender clusters
4. **Rationale:** Preserves meaningful gender information while maintaining anonymity

### Implementation Notes
- All k values were analyzed for comprehensive comparison
- Only optimal k value files are generated to provide clean final output
- Report includes full analysis for transparency and justification
- Categorical generalization handles mixed-type quasi-identifiers appropriately
