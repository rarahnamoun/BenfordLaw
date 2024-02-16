import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.DataFrame({'x': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                         12, 13, 14, 15, 16, 17, 18, 19, 20, 20, 20, 22, 23],
                   'y': [10953, 7832, 4514, 4161, 3702, 260, 106, 62, 60, 52, 48,
                         40, 30, 28, 26, 9, 8, 6, 4, 3, 3, 3, 2, 1]})

# Calculate Zipf's Law distribution
C = max(df.y)  # Choose C such that the maximum count matches the data
s = 2.0  # Adjust the value of s as needed
zipf_pmf = C / np.power(df.x, s)

xlog = np.log(df.x)
ylog = np.log(df.y)
zipf_pmf_log = np.log(zipf_pmf)

plt.scatter(xlog, ylog, color='black', label='Data')
plt.plot(xlog, zipf_pmf_log, color='red', label="Zipf's Law (Log-Log)")
plt.xlabel('Log Rank (log(x))')
plt.ylabel('Log Count (log(y))')

plt.title('Log-Log Plot with Zipf\'s Law')
plt.legend()
plt.show()
