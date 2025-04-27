import matplotlib.pyplot as plt
import seaborn as sns
from Data import data

# Set up the plotting style
sns.set(style="whitegrid")

# 1. Visualize distribution of Age for survived and not survived
plt.figure(figsize=(10, 6))
sns.histplot(data[data['Survived'] == 1]['Age'], kde=True, color='green', label='Survived', bins=20)
sns.histplot(data[data['Survived'] == 0]['Age'], kde=True, color='red', label='Did not survive', bins=20)
plt.title("Age Distribution by Survival")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.legend()
plt.show()
