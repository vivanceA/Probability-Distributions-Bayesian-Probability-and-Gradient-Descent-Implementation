# import necessary libraries
import numpy as np
import matplotlib.pyplot as plt


# Define the arrival rate (packets per second)
lambda_rate = 500  # 500 packets per second

# Simulate inter-arrival times using NumPy's exponential distribution
num_samples = 100000  # Large dataset to see smooth distribution
inter_arrival_times = np.random.exponential(scale=1/lambda_rate, size=num_samples) * 1000  # Convert to milliseconds

# Generate theoretical Exponential PDF
x_values = np.linspace(0, 20, 1000)  # Time range (0 to 20 ms) with smooth intervals
pdf_values = (lambda_rate / 1000) * np.exp(-lambda_rate * x_values / 1000)  # PDF formula adjusted for milliseconds

# Generate theoretical Exponential CDF
cdf_values = 1 - np.exp(-lambda_rate * x_values / 1000)  # CDF formula

# Create figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot the histogram and PDF on the first subplot
axes[0].hist(inter_arrival_times, bins=100, density=True, alpha=0.6, color='b', label='Simulated Data')
axes[0].plot(x_values, pdf_values, 'r-', label='Theoretical PDF', linewidth=2)
axes[0].set_xlabel("Inter-arrival Time (milliseconds)")
axes[0].set_ylabel("Probability Density")
axes[0].set_title("Exponential Distribution of Network Packet Arrival Times (PDF)")
axes[0].legend()
axes[0].grid(True)

# Plot only the CDF on the second subplot
axes[1].plot(x_values, cdf_values, 'g-', label='Theoretical CDF', linewidth=2)
axes[1].set_xlabel("Inter-arrival Time (milliseconds)")
axes[1].set_ylabel("Cumulative Probability")
axes[1].set_title("Cumulative Distribution Function (CDF) of Packet Arrival Times")
axes[1].legend()
axes[1].grid(True)

# Adjust x-axis ticks for both plots
for ax in axes:
    ax.set_xticks(np.arange(0, 21, 5))

# Show the plots
plt.tight_layout()
plt.show()
