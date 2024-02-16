

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

ally = open("Text-2.txt", "r")
wordcount = {}
for word in ally.read().split():
    if word not in wordcount:
        wordcount[word] = 1
    else:
        wordcount[word] += 1

#for k, v, in sorted(wordcount.items(), key=lambda words: words[1], reverse=True):
   # print(k, v)

#1the 10953
#2of 7832
#3and 4514
#4in 4161
#5to 3702
#6nature 260
#7within 106
#8size 62
#9distribution 60
#10body 52
#11existence 48
#12head40
#13eye 30
#14conclusion 28
#15second 26
#16words 9
#17hair 8
#18necessity 6
#19muscles 4
#20darwin 3
#20touch 3
#20revolution 3
#22publication 2
#23month 1






df = pd.DataFrame({'x': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                         12, 13, 14,15, 16, 17, 18, 19, 20,20,20,22,23],
                   'y': [10953, 7832, 4514, 4161,3702,  260, 106, 62, 60, 52, 48,
                         40, 30, 28, 26, 9, 8, 6, 4,3,3,3,2,1]})
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
xlog = np.log(df.x)
ylog = np.log(df.y)
plt.scatter(df.x,df.y, color='black')
plt.xlabel('Rank(x)')
plt.ylabel('Count(y)')
plt.title(' Plot')
plt.show()

plt.scatter(xlog, ylog, color='blue')
plt.xscale('log')
plt.yscale('log')
plt.title('Log-Log Plot')
plt.show()