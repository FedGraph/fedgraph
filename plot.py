import matplotlib.pyplot as plt
import numpy as np

# Set global font sizes
plt.rcParams['font.size'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14

# Data preparation
datasets = ['Cora', 'Citeseer', 'PubMed']
time_plaintext = [13.29, 20.39, 25.78]
time_encrypted = [27.71, 79.35, 70.28]
comm_plaintext = [59.21, 187.99, 150.43]
comm_encrypted = [3279.15, 5791.42, 3612.60]

# Set width of bars
width = 0.35
x = np.arange(len(datasets))

# First plot - Time Comparison
plt.figure(figsize=(8, 6))
ax1 = plt.gca()
ax1.bar(x - width/2, time_plaintext, width, label='Plain-text')
ax1.bar(x + width/2, time_encrypted, width, label='Encrypted (HE)')
ax1.set_ylabel('Time (seconds)', fontsize=14)
#ax1.set_title('Execution Time Comparison', fontsize=16, pad=15)
ax1.set_xticks(x)
ax1.set_xticklabels(datasets, fontsize=14)
ax1.legend(fontsize=14)

# # Add value labels for time comparison
# for i, v in enumerate(time_plaintext):
#     ax1.text(i - width/2, v, f'{v}', ha='center', va='bottom', fontsize=12)
# for i, v in enumerate(time_encrypted):
#     ax1.text(i + width/2, v, f'{v}', ha='center', va='bottom', fontsize=12)

plt.tight_layout()
plt.savefig('time_comparison.pdf', format='pdf', bbox_inches='tight', dpi=300)

# Second plot - Communication Cost
plt.figure(figsize=(8, 6))
ax2 = plt.gca()
ax2.bar(x - width/2, comm_plaintext, width, label='Plain-text')
ax2.bar(x + width/2, comm_encrypted, width, label='Encrypted (HE)')
ax2.set_ylabel('Communication Cost (MB)', fontsize=14)
#ax2.set_title('Communication Cost Comparison', fontsize=16, pad=15)
ax2.set_xticks(x)
ax2.set_xticklabels(datasets, fontsize=14)
ax2.legend(fontsize=14)
ax2.set_yscale('log')

# # Add value labels for communication cost
# for i, v in enumerate(comm_plaintext):
#     ax2.text(i - width/2, v*1.1, f'{v}', ha='center', va='bottom', fontsize=12)
# for i, v in enumerate(comm_encrypted):
#     ax2.text(i + width/2, v*1.1, f'{v}', ha='center', va='bottom', fontsize=12)

plt.tight_layout()
plt.savefig('comm_cost_comparison.pdf', format='pdf', bbox_inches='tight', dpi=300)

plt.show()