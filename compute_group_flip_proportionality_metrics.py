import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from fpdf import FPDF

from group_flip_proportionality_metrics import (
    get_total_flips,
    get_group_flips,
    directional_flip_ratio,
    harmful_flip_proportion,
    group_wise_flip_proportionality_metrics)


def compute_all_metrics(file_path, pa_name, y_pred_name, y_corrected_name):
    """
    Compute and display all flip-related metrics for a dataset

    Args:
        file_path: Path to the CSV file containing the dataset
        pa_name: Name of the column containing the protected attribute
        y_pred_name: Name of the column containing the predicted outcome
        y_corrected_name: Name of the column containing the correct outcome

    Returns:
        dict: Dictionary containing all computed metrics
    """
    # Read the dataset
    # try:
    if type(file_path) is not pd.DataFrame:
        df = pd.read_csv(file_path)
    else:
        df = file_path.copy()

    df.rename(columns={pa_name: 'pa', y_pred_name: 'y_pred', y_corrected_name: 'y_corrected'}, inplace=True)

    # Compute flip_yp_corrected if not already in the dataset
    if 'flip_yp_corrected' not in df.columns:
        df['flip_yp_corrected'] = df['y_pred'] != df['y_corrected']

    if 'flip_yp_harmful' not in df.columns:
        df['flip_yp_harmful'] = (df['y_pred'] == 1) & (df['y_corrected'] == 0)


    # Define the groups
    group_0 = df[df['pa'] == 0]
    group_1 = df[df['pa'] == 1]

    # Initialize results dictionary
    results = {}

    # Add dataset info
    results['dataset_info'] = {
        'total_samples': len(df),
        'group_0_samples': len(df[df['pa'] == 0]),
        'group_1_samples': len(df[df['pa'] == 1])
    }

    # Compute total flips
    total_flips, total_flip_rate = get_total_flips(df)
    results['overall_flips'] = {
        'total_flips': total_flips,
        'total_flip_rate': total_flip_rate
    }

    # Compute flips for each group
    flips_0, flip_rate_0 = get_group_flips(df, 0)
    flips_1, flip_rate_1 = get_group_flips(df, 1)
    results['flips_by_group'] = {
        'flips_group_0': flips_0,
        'flip_rate_group_0': flip_rate_0,
        'flips_group_1': flips_1,
        'flip_rate_group_1': flip_rate_1
    }

    # Directional Flip Ratio
    dfr, desc = directional_flip_ratio(df)
    dfr_group_0, desc_0 = directional_flip_ratio(group_0)
    dfr_group_1, desc_1 = directional_flip_ratio(group_1)
    results['directional flip ratio'] = {
        'dfr': dfr,
        'dfr_group_0': dfr_group_0,
        'dfr_group_1': dfr_group_1,
        'desc': desc,
        'desc_0': desc_0,
        'desc_1': desc_1
    }

    hf, hfp, desc = harmful_flip_proportion(df)
    hf_0, hfp_0, desc_o = harmful_flip_proportion(group_0)
    hf_1, hfp_1, desc_1 = harmful_flip_proportion(group_1)
    results['harmful flip proportion'] = {
        'hfp': hfp,
        'hfp_0': hfp_0,
        'hfp_1': hfp_1,
        'hf': hf,
        'hf_0': hf_0,
        'hf_1': hf_1,
        'desc': desc,
        'desc_0': desc_0,
        'desc_1': desc_1
    }

    # Flip proportionality metrics
    flip_proportionality_metrics = group_wise_flip_proportionality_metrics(flip_rate_0, flip_rate_1, total_flip_rate)
    results['flip proportionality metrics'] = flip_proportionality_metrics

    # Harmful Flip proportionality metrics
    flip_proportionality_metrics = group_wise_flip_proportionality_metrics(hfp_0, hfp_1, hfp)
    results['harmful flip proportionality metrics'] = flip_proportionality_metrics

    # Compute group-specific metrics
    # results['group_metrics'] = get_all_group_flips(df)

    # Get summary table
    # results['summary_table'] = get_flip_summary(df)

    return results

    # except Exception as e:
    #    print(f"Error processing file: {str(e)}")
    #    return None


def display_results(results):
    """
    Display the computed metrics in a formatted way

    Args:
        results: Dictionary containing all computed metrics
    """
    if results is None:
        return

    print("\n=== Dataset Information ===")
    print(f"Total samples: {results['dataset_info']['total_samples']}")
    print(f"Group 0 samples: {results['dataset_info']['group_0_samples']}")
    print(f"Group 1 samples: {results['dataset_info']['group_1_samples']}")

    print("\n=== Overall Metrics ===")
    print(f"Total flips: {results['overall_flips']['total_flips']}")
    print(f"Total flip rate: {results['overall_flips']['total_flip_rate']:.2f}")
    print(f"Harmful Flips: {results['harmful flip proportion']['hf']}")
    print(f"Harmful Flip Proportion: {results['harmful flip proportion']['hfp']}, {results['harmful flip proportion']['desc']}")

    print("\n=== Flips by Groups ===")
    print(f"Group 0 flips: {results['flips_by_group']['flips_group_0']}")
    print(f"Group 0 flip rate: {results['flips_by_group']['flip_rate_group_0']:.3f}")
    print(f"Group 0 Harmful Flips: {results['harmful flip proportion']['hf_0']}")
    print(f"Group 0 Harmful Flip Proportion: {results['harmful flip proportion']['hfp_0']}, {results['harmful flip proportion']['desc_0']}")
    print(f"Group 1 flips: {results['flips_by_group']['flips_group_1']}")
    print(f"Group 1 flip rate: {results['flips_by_group']['flip_rate_group_1']:.3f}")
    print(f"Group 1 Harmful Flips: {results['harmful flip proportion']['hf_1']}")
    print(f"Group 1 Harmful Flip Proportion: {results['harmful flip proportion']['hfp_1']}, {results['harmful flip proportion']['desc_1']}")

    print("\n=== Directional Flip Ratio ===")
    print(f"Directional Flip Ratio: {results['directional flip ratio']['dfr']}, {results['directional flip ratio']['desc']}")
    print(f"Group 0 Directional Flip Ratio: {results['directional flip ratio']['dfr_group_0']}, {results['directional flip ratio']['desc_0']}")
    print(f"Group 1 Directional Flip Ratio: {results['directional flip ratio']['dfr_group_1']}, {results['directional flip ratio']['desc_1']}")

    print("\n=== Flip Proportionality Metrics ===")
    print(f"Flip Rate Difference: {results['flip proportionality metrics']['frd']}")
    print(f"Disparity Index: {results['flip proportionality metrics']['di']}, {results['flip proportionality metrics']['di_desc']}")
    print(f"Flip Rate Disparity: {results['flip proportionality metrics']['fd']}, {results['flip proportionality metrics']['fd_desc']}")
    print(f"Relative Flip Disparity: {results['flip proportionality metrics']['nfd']}, {results['flip proportionality metrics']['nfd_desc']}")


    print("\n=== Harmful Flip Proportionality Metrics ===")
    print(f"Harmful Flip Rate Difference: {results['harmful flip proportionality metrics']['frd']}")
    print(f"Harmful Disparity Index: {results['harmful flip proportionality metrics']['di']}, {results['harmful flip proportionality metrics']['di_desc']}")
    print(f"Harmful Flip Proportionality Disparity: {results['harmful flip proportionality metrics']['fd']}, {results['harmful flip proportionality metrics']['fd_desc']}")
    print(f"Relative Harmful Flip Disparity: {results['harmful flip proportionality metrics']['nfd']}, {results['harmful flip proportionality metrics']['nfd_desc']}")

def plot_group_flip_proportionality_metrics(regular_metrics, harmful_metrics):
    metrics_values_flip = {
        "FRD": regular_metrics['frd'],
        "DI": regular_metrics['di'],
        "FD": regular_metrics['fd'],
        "RFD": regular_metrics['nfd']
    }

    metrics_values_harmful = {
        "HFPD": harmful_metrics['frd'],
        "HDI": harmful_metrics['di'],
        "HFD": harmful_metrics['fd'],
        "RHFD": harmful_metrics['nfd']
    }

    # Define thresholds for low, medium, high
    thresholds = {
        "FRD": [[0.05, 0.15]],
        "DI": [[1.1, 1.3], [0.9, 0.7]],
        "FD": [[0.1, 0.3]],
        "RFD": [[0.1, 0.3]]
    }

    # Define thresholds for low, medium, high
    thresholds_harmful = {
        "HFPD": [[0.05, 0.15]],
        "HDI": [[1.1, 1.3], [0.9, 0.7]],
        "HFD": [[0.1, 0.3]],
        "RHFD": [[0.1, 0.3]]
    }

    # Categorize metrics into low, medium, high
    def categorize(value, bounds):
        if len(bounds) == 1:
            if np.isinf(value):
                return "High"
            if value <= bounds[0][0]:
                return "Low"
            elif value <= bounds[0][1]:
                return "Medium"
            else:
                return "High"
        else:
            if np.isinf(value):
                return "High"
            if value in [bounds[1][0], bounds[0][0]]:
                return "Low"
            elif value <= bounds[0][1] or value >= bounds[1][1]:
                return "Medium"
            else:
                return "High"


    # Categorize metrics
    categories_flip = {k: categorize(v, thresholds[k]) for k, v in metrics_values_flip.items()}
    categories_harmful = {k: categorize(v, thresholds_harmful[k]) for k, v in metrics_values_harmful.items()}

    # Map categories to colors
    color_map = {"Low": "#77DD77", "Medium": "#FFB347", "High": "#FF6961"}  # Green, Yellow, Red

    # Prepare data for plotting
    def prepare_plot_data(metrics_values, categories):
        metric_names = list(metrics_values.keys())
        metric_values = [metrics_values[m] if not np.isinf(metrics_values[m]) else 10 for m in
                         metric_names]  # Cap infinity for plot
        colors = [color_map[categories[m]] for m in metric_names]
        return metric_names, metric_values, colors

    # Data for Flip Proportionality Metrics
    flip_names, flip_values, flip_colors = prepare_plot_data(metrics_values_flip, categories_flip)

    # Data for Harmful Flip Proportionality Metrics
    harmful_names, harmful_values, harmful_colors = prepare_plot_data(metrics_values_harmful, categories_harmful)

    # Plot settings
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    #fig.suptitle("Flip and Harmful Flip Proportionality Metrics", fontsize=16, fontweight="bold")

    # Bar plot for Flip Proportionality Metrics
    axs[0].barh(flip_names, flip_values, color=flip_colors, edgecolor='black')
    axs[0].set_title("Flip Proportionality Metrics", fontsize=14, fontweight="bold")
    axs[0].set_xlabel("Metric Value", fontsize=12)
    axs[0].invert_yaxis()  # Align bars from top to bottom

    # Annotate bars with values
    for i, value in enumerate(flip_values):
        axs[0].text(value + 0.1 if value < 10 else 9.5, i, f"{value:.2f}", va='center', ha='left', fontsize=10)

    # Bar plot for Harmful Flip Proportionality Metrics
    axs[1].barh(harmful_names, harmful_values, color=harmful_colors, edgecolor='black')
    axs[1].set_title("Harmful Flip Proportionality Metrics", fontsize=14, fontweight="bold")
    axs[1].set_xlabel("Metric Value", fontsize=12)
    axs[1].invert_yaxis()  # Align bars from top to bottom

    # Annotate bars with values
    for i, value in enumerate(harmful_values):
        axs[1].text(value + 0.1 if value < 10 else 9.5, i, f"{value:.2f}" if not np.isinf(value) else "inf",
                    va='center', ha='left', fontsize=10)

    # Add legends for categories
    handles = [plt.Rectangle((0, 0), 1, 1, color=color_map[cat]) for cat in color_map]
    labels = list(color_map.keys())
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=12)

    # Adjust layout and show plot
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return plt


def viasualize_flips(flips, harmful_flips):
    # Data to visualize
    values = [flips, harmful_flips]

    # Create the figure and axes
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Plot the Total Flip Rate as a circle
    axes[0].pie([values[0], 1 - values[0]], labels=['Flips', ''], colors=['blue', 'lightgray'],
                autopct=lambda p: f'{p:.0f}%' if p > 0 else '', startangle=90)
    axes[0].set_title('Total Flips')

    # Plot the Harmful Flip Proportion as a circle
    axes[1].pie([values[1], 1 - values[1]], labels=['Harmful Flips', ''], colors=['orange', 'lightgray'],
                autopct=lambda p: f'{p:.0f}%' if p > 0 else '', startangle=90)
    axes[1].set_title('Total Harmful Flips')

    # Adjust layout for better spacing
    plt.tight_layout()

    return plt


def visualize_flips_by_group(fr_0, hfp_0, fr_1, hfp_1):
    # Data to visualize
    groups = ['Group 0', 'Group 1']
    flip_rates = [fr_0, fr_1]
    harmful_proportions = [hfp_0, hfp_1]

    # Create the figure and axes
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    x = np.arange(len(groups))  # Group positions

    # Plot Flip Rates on the left
    for i, group in enumerate(groups):
        axes[0].barh(i, 1, color='lightgray', edgecolor='black')  # Full bar
        axes[0].barh(i, flip_rates[i], color='blue', edgecolor='black')  # Filled portion
    axes[0].set_yticks(np.arange(len(groups)))
    axes[0].set_yticklabels(groups)
    axes[0].set_xlim(0, 1)
    axes[0].set_title('Flips')
    # axes[0].set_xlabel('Proportion')
    for i, v in enumerate(flip_rates):
        axes[0].text(v / 2, i, f'{v * 100:.1f}%', va='center', ha='center', fontsize=10)

    # Plot Harmful Flip Proportions on the right
    for i, group in enumerate(groups):
        axes[1].barh(i, 1, color='lightgray', edgecolor='black')  # Full bar
        axes[1].barh(i, harmful_proportions[i], color='orange', edgecolor='black')  # Filled portion
    axes[1].set_yticks(np.arange(len(groups)))
    axes[1].set_yticklabels(groups)
    axes[1].set_xlim(0, 1)
    axes[1].set_title('Harmful Flips')
    # axes[1].set_xlabel('Proportion')
    for i, v in enumerate(harmful_proportions):
        axes[1].text(v / 2, i, f'{v * 100:.1f}%', va='center', ha='center', fontsize=10)

    # Adjust layout for better spacing
    plt.tight_layout()

    return plt


def save_results_to_pdf(results, flips_plot, flips_group_plot, measures_plot, output_pdf_path):
    """
    Save the computed metrics and the measures plot to a PDF.

    Args:
        results: Dictionary containing all computed metrics.
        measures_plot: Matplotlib figure object for the measures plot.
        output_pdf_path: Path to save the PDF file.
    """

    class PDF(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 12)
            self.cell(0, 10, 'Flip Proportionality Analysis', align='C', ln=True)

        def chapter_title(self, title):
            self.set_font('Arial', 'B', 12)
            self.cell(0, 10, title, ln=True)
            self.ln(5)

        def chapter_body(self, body):
            self.set_font('Arial', '', 10)
            self.multi_cell(0, 6, body)
            self.ln()

    # Generate results as text
    def generate_results_text(results):
        lines = []
        if results is None:
            return "\nNo results to display."

        lines.append("\n=== Dataset Information ===")
        lines.append(f"Total Samples: {results['dataset_info']['total_samples']}")
        lines.append(f"Group 0 samples: {results['dataset_info']['group_0_samples']}")
        lines.append(f"Group 1 samples: {results['dataset_info']['group_1_samples']}")

        lines.append("\n=== Overall Metrics ===")
        lines.append(f"Total Flips: {results['overall_flips']['total_flips']}")
        lines.append(f"Total Flip rate: {results['overall_flips']['total_flip_rate']:.2f}")
        lines.append(f"Harmful Flips: {results['harmful flip proportion']['hf']}")
        lines.append(
            f"Harmful Flip Proportion: {results['harmful flip proportion']['hfp']}, {results['harmful flip proportion']['desc']}")

        lines.append("\n=== Flips by Groups ===")
        lines.append(f"Group 0 Flips: {results['flips_by_group']['flips_group_0']}")
        lines.append(f"Group 0 Flip rate: {results['flips_by_group']['flip_rate_group_0']:.3f}")
        lines.append(f"Group 0 Harmful Flips: {results['harmful flip proportion']['hf_0']}")
        lines.append(
            f"Group 0 Harmful Flip Proportion: {results['harmful flip proportion']['hfp_0']}, {results['harmful flip proportion']['desc_0']}")
        lines.append(f"Group 1 Flips: {results['flips_by_group']['flips_group_1']}")
        lines.append(f"Group 1 Flip rate: {results['flips_by_group']['flip_rate_group_1']:.3f}")
        lines.append(f"Group 1 Harmful Flips: {results['harmful flip proportion']['hf_1']}")
        lines.append(
            f"Group 1 Harmful Flip Proportion: {results['harmful flip proportion']['hfp_1']}, {results['harmful flip proportion']['desc_1']}")

        lines.append("\n=== Directional Flip Ratio ===")
        lines.append(
            f"Directional Flip Ratio: {results['directional flip ratio']['dfr']}, {results['directional flip ratio']['desc']}")
        lines.append(
            f"Group 0 Directional Flip Ratio: {results['directional flip ratio']['dfr_group_0']}, {results['directional flip ratio']['desc_0']}")
        lines.append(
            f"Group 1 Directional Flip Ratio: {results['directional flip ratio']['dfr_group_1']}, {results['directional flip ratio']['desc_1']}")

        lines.append("\n=== Flip Proportionality Metrics ===")
        lines.append(f"Flip Rate Difference: {results['flip proportionality metrics']['frd']}")
        lines.append(
            f"Disparity Index: {results['flip proportionality metrics']['di']}, {results['flip proportionality metrics']['di_desc']}")
        lines.append(
            f"Flip Rate Disparity: {results['flip proportionality metrics']['fd']}, {results['flip proportionality metrics']['fd_desc']}")
        lines.append(
            f"Relative Flip Disparity: {results['flip proportionality metrics']['nfd']}, {results['flip proportionality metrics']['nfd_desc']}")

        lines.append("\n=== Harmful Flip Proportionality Metrics ===")
        lines.append(f"Harmful Flip Proportion Difference: {results['harmful flip proportionality metrics']['frd']}")
        lines.append(
            f"Harmful Disparity Index: {results['harmful flip proportionality metrics']['di']}, {results['harmful flip proportionality metrics']['di_desc']}")
        lines.append(
            f"Harmful Flip Proportionality Disparity: {results['harmful flip proportionality metrics']['fd']}, {results['harmful flip proportionality metrics']['fd_desc']}")
        lines.append(
            f"Relative Harmful Flip Disparity: {results['harmful flip proportionality metrics']['nfd']}, {results['harmful flip proportionality metrics']['nfd_desc']}")

        return "\n".join(lines)

    # Create PDF
    pdf = PDF()
    pdf.add_page()

    # Add results text
    pdf.chapter_title("Metrics Results")
    pdf.chapter_body(generate_results_text(results))

    # Add plot to PDF
    # Define paths for the three plots
    names_plots = {0: flips_plot, 1: flips_group_plot, 2: measures_plot}


    # Add the three plots to the same page in the PDF
    pdf.add_page()
    pdf.chapter_title("Visualization")

    # Adjust positioning for each image on the page
    positions = [(10, 40), (10, 120), (10, 200)]  # Example y-coordinates
    width = 180  # Common width for all plots

    for i, plot_path in names_plots.items():
        print(plot_path)
        x, y = positions[i]
        pdf.image(plot_path, x=x, y=y, w=width)

    # Step 4: Save PDF
    pdf.output(output_pdf_path)
    print(f"Results and plot saved to {output_pdf_path}")
    print(names_plots)


def main():
    # File path
    file_path = 'results/test_compas_binary_race_100_0.csv'

    # Name of the necessary columns
    pa = 'pa'
    y_pred = 'y_pred'
    y_corrected = 'y_corrected'

    # Compute metrics
    results = compute_all_metrics(file_path, pa, y_pred, y_corrected)

    # plot metrics
    plt_flips = viasualize_flips(results['overall_flips']['total_flip_rate'], results['harmful flip proportion']['hfp'])
    plt_flips.savefig("results/Flips.png")
    plt.close()

    plt_flips_groups = visualize_flips_by_group(results['flips_by_group']['flip_rate_group_0'],
                                                results['harmful flip proportion']['hf_0'],
                                                results['flips_by_group']['flip_rate_group_1'],
                                                results['harmful flip proportion']['hf_1'])
    plt_flips_groups.savefig("results/Flips_groups.png")
    plt.close()

    plt_group = plot_group_flip_proportionality_metrics(results['flip proportionality metrics'], results['harmful flip proportionality metrics'])
    plt_group.savefig("results/Group_Flip_proportionality_metrics_comparison.png")

    # Saving results on a pdf report
    save_results_to_pdf(results, "results/Flips.png", "results/Flips_groups.png",
                        "results/Group_Flip_proportionality_metrics_comparison.png",
                        "results/Proportionality Report.pdf")

    # Display results
    if results:
        display_results(results)

    return results


if __name__ == "__main__":
    main()