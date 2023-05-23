from curses.ascii import alt
import scipy.stats
import matplotlib.pyplot as plt
import random

vals = [
    0.8,
    0.7,
    0.5,
    0.6,
    0.5,
    0.7,
    0.4,
    0.4,
    0.8,
    0.4,
    0.9,
    0.4,
    0.4,
    0.6,
    0.6,
    0.6,
    0.7,
    0.6,
    0.7,
    0.6
]

result = scipy.stats.ttest_1samp(vals, 0.5, alternative='greater')
# print(result)

# energy difference scatterplots

bitch = ['$a$', '$b$', '$c$', '$d$', '$e$']

TN_VE = [46,15,6,76,64,64,76,11,100,27,76,50,14,100,49,6,14,64,89,89,46,89,15,6,49,76,76,27,53,15,6,50,6,76,60,46,50,13,27,15,50,64,6]
TN_AE = [35,6,46,27,35,53,53,100,13,6,89,76,100,50,18,46,76,46,18,6,6,11,49,89,53,11,46,6,50,64,100,49,60,46,15,76,34,100,53,50,49,14,100]
plt.scatter(TN_VE, TN_AE, color='Green', marker='$✓$', label='TN')

FP_VE = [35,49,27,53,14,49,6,14,27,6,27,11,50,64,13,34,18,11,50,13,11,18,15,49,11,50,50,100,27,13,50,50,100,11,35,6,64,49,11,50,14,100,50,11,27,53,34,35,60,6,18,53,53,34,76,11,6]
FP_AE = [13,64,6,76,13,6,13,49,46,6,64,64,13,18,89,46,50,64,50,11,100,34,18,11,49,6,11,60,15,27,15,35,6,18,50,64,34,6,15,14,34,76,6,53,50,11,60,50,15,18,6,34,35,15,64,34,27]
plt.scatter(FP_VE, FP_AE, color='Red', marker=random.choice(bitch), label='FP')

plt.plot([0, 100], [0, 100])

plt.xlabel("video energy score")
plt.ylabel("music energy score")
plt.legend()
plt.axis('square')
plt.show()