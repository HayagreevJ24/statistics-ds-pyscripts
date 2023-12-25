import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def loadedDiceMethod(x: list, n: int, p: list):
    roll1 = np.array([np.random.choice(x, p=p) for _ in range(n)])
    roll2 = np.array([np.random.choice(x, p=p) for _ in range(n)])
    rollf = roll1 + roll2

    distinctRolls = np.array(list(set(rollf)))
    distinctRollsCount = np.array([0 for i in range(len(distinctRolls))])
    for i in range(len(distinctRolls)):
        for x in rollf:
            if x == distinctRolls[i]:
                distinctRollsCount[i] += 1

    distinctRollsProbabilities = distinctRollsCount * (1 / n)

    plt.bar(distinctRolls, distinctRollsProbabilities)
    plt.xlabel('Roll sum')
    plt.ylabel('Roll Probabilities')
    plt.show()
    # sns.histplot(rollf, discrete=True)

    return distinctRolls, distinctRollsProbabilities


x = np.arange(1, 11, 1)
p = np.array([1 / len(x) for i in range(len(x))])
n = 100000
sum, prob = loadedDiceMethod(x, n, p)

cumulativeProb = [0 for i in range(len(prob))]
cumulativeCount = 0
for i in range(len(prob)):
    cumulativeCount += prob[i]
    cumulativeProb[i] = cumulativeCount

plt.bar(sum, cumulativeProb)
plt.show()
print(sum, "\n", cumulativeProb)