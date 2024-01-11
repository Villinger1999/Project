from Load_data import df
import matplotlib.pyplot as plt
from scipy import stats

variables = ['HR_Mean', 'HR_Median', 'HR_std', 'HR_Max', 'HR_Min', 'HR_AUC']

# Create a 2x3 grid of subplots
fig, axes = plt.subplots(2, 3, figsize=(10, 10))

# Generate QQ plots for each variable and calculate p-values
for i, variable in enumerate(variables):
    # Calculate the row and column indices for the subplot
    row = i // 3
    col = i % 3

    # Select the subplot
    ax = axes[row, col]

    # Generate the QQ-plot
    stats.probplot(df[variable], plot=ax)

    # Perform the Shapiro-Wilk test
    statistic, p_value = stats.shapiro(df[variable])

    # Set the plot title and labels
    ax.set_title(f'{variable} (p-value: {p_value:.4f})')
    ax.set_xlabel('Theoretical Quantiles')
    ax.set_ylabel('Ordered Values')

# Adjust the spacing between subplots
plt.tight_layout()

# Display the plots
plt.show()