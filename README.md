# Group wise flip proportionality metrics

The file `group_flip_proportionality_metrics.py` contains the code for computing the groupwise proportionality metrics that will be described in the tables below.

### Table: Metrics for Characterizing the Flips in a Solution

**Summary of the proposed metrics describing flips in the solution (can be used to describe different groups separately). Each line delineates the boundaries of the metric, includes a reference to their respective equations, and provides a short description along with any edge cases associated with them.**

| **Metric** | **Boundaries** | **Short Description & Edge Cases** |
|------------|----------------|-------------------------------------|
| `FR`       | [0, 1]         | Measures the proportion of instances where predictions change after debiasing. Is 0 when there is no flip in the post-processing stage. Values close to 1 suggest more flips. |
| `DFR`      | [0, ∞)         | Ratio of beneficial to harmful flips. Indicates the balance between favorable and unfavorable flips. Values close to 1 are desirable. Returns ∞ when there are no harmful flips. Returns 0 when there are no beneficial flips and 1 in the absence of flips. |
| `HFP`      | [0, 1]         | Proportion of flips leading to unfavorable outcomes. Values close to 1 indicate a higher incidence of harmful flips. Is 0 when there are no harmful flips. Takes 1 if all the flips are harmful. |

### Table: Metrics for Quantifying the Proportionality of the Flips Between Groups

**Summary of the proposed metrics to quantify the proportionality of the flips between groups. Each line delineates the boundaries of the metric, includes a reference to their respective equations, and provides a short description along with any edge cases associated with them.**

| **Metric**         | **Boundaries** | **Short Description & Edge Cases** |
|--------------------|----------------|-------------------------------------|
| `FRD` & `HFPD`     | [0, 1]         | Absolute differences in flip rates or harmful flip proportions between groups. Takes value 0 when FR or HFP are equal in both groups. Values close to 0 indicate greater proportionality. |
| `DI` & `HDI`       | [1, ∞)         | Ratio of flip rates or harmful flip proportions between groups. Returns ∞ when the minimum flip rate or harmful flip proportion is 0. Returns 1 when both values are equal or zero. |
| `FD` & `HFD`       | [0, ∞)         | Quantify disparities in flip rates or harmful flip proportions between groups, normalized by the overall flip rate. Can become very large when overall flip rate or harmful flip proportion approaches 0. Returns 1 when both are 0 and ∞ when one is 0. |
| `RFD` & `RHFD`     | [0, 1]         | Relative disparity in flip rates or harmful flips between groups. This normalized metric simplifies comparison across scenarios. Returns 1 when one rate is 0 and the other is not, and 0 when no flips occur in either group. |

### Computing Group Proportionality Metrics

By executing the script [`compute_group_flip_proportionality_metrics.py`](./compute_group_flip_proportionality_metrics.py), users can generate a report of group-level proportionality metrics related to prediction flips.

#### Required Inputs

The user must provide the following arguments:

- `file_path`: Path to the CSV file containing the dataset.
- `pa_name`: Name of the column representing the protected attribute.
- `y_pred_name`: Name of the column containing the predicted outcome.
- `y_corrected_name`: Name of the column containing the corrected (post-processed or debiased) outcome.


#### Output

The results are saved in [`results/Proportionality Report.pdf`](./results/Proportionality%20Report.pdf), which includes detailed metrics and visualizations assessing the proportionality of flips between groups.

#### Thresholds

The script uses predefined thresholds to categorize the results into **low**, **medium**, and **high** proportionality disparities. These are defined in the code as follows:

```python
# Define thresholds for low, medium, high
thresholds = {
    "FRD": [[0.05, 0.15]],
    "DI": [[1.1, 1.3], [0.9, 0.7]],
    "FD": [[0.1, 0.3]],
    "RFD": [[0.1, 0.3]]
}

thresholds_harmful = {
    "HFPD": [[0.05, 0.15]],
    "HDI": [[1.1, 1.3], [0.9, 0.7]],
    "HFD": [[0.1, 0.3]],
    "RHFD": [[0.1, 0.3]]
}