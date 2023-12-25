# Importing the required modules
import numpy as np
import warnings
import matplotlib.pyplot as plt
# import seaborn as sns

# Class for the dice
def createLoadedProbabilities(loadingFactor: float, numberOfSides: int) -> np.ndarray:
    # First validate the loadingFactor.
    if not (0 <= loadingFactor <= 1):
        raise ValueError(f"Expected a value between 0 and 1 for loading factor. Got {loadingFactor}")
    if loadingFactor == 0:
        warnings.warn("Note: a loading factor of 0 will create an object which will behave like an unbiased dice.")

    # Create the probability array and modify according to loadingFactor.
    probabilityArray = np.array([(1/numberOfSides) for i in range(numberOfSides)])
    changeIndices = [np.random.randint(0, numberOfSides) for i in range(int(loadingFactor * numberOfSides))]
    for x in changeIndices:
        probabilityArray[x] *= np.random.random()
    # Normalise the probabilities by dividing by their sum so that the sum of the array is 1.
    probabilityArray /= np.sum(probabilityArray)

    return probabilityArray


class LoadingTypeError(Exception):
    pass

class Dice:
    def __init__(self, numberOfSides, loadingType, loadingFactor=None):
        self.numberOfSides = numberOfSides
        match loadingType:
            case 'unbiased':
                if loadingFactor:
                    warnings.warn("Warning: loadingFactor was not expected in the case of loadingType 'unbiased'. loadingFactor will be reset to None.")
                self.probabilityArray = [(1/self.numberOfSides) for i in range(self.numberOfSides)]
            case 'biased':
                if not loadingFactor and loadingFactor != 0:
                    raise AttributeError("Expected third parameter loadingFactor for loadingType 'biased', Found None.")
                self.probabilityArray = createLoadedProbabilities(loadingFactor, self.numberOfSides)
            case _:
                raise LoadingTypeError(f'Expected \'unbiased\' or \'biased\' for loadingType. Got {loadingType} instead.')

    def singleRoll(self):
        return np.random.choice(np.arange(1, self.numberOfSides + 1), p=self.probabilityArray)

    def multiRoll(self, numberOfThrows):
        return np.random.choice(np.arange(1, self.numberOfSides + 1), numberOfThrows, p=self.probabilityArray)


def main():
    MyDice = Dice(130, 'biased', 0.99)

    numberOfThrows = 100000
    Rolldata = MyDice.multiRoll(numberOfThrows)
    distinctRolls = np.unique(Rolldata)
    distinctRollsCounts = np.zeros(len(distinctRolls))

    for x in Rolldata:
        for i in range(len(distinctRolls)):
            if x == distinctRolls[i]:
                distinctRollsCounts[i] += 1

    distinctRollsProbabilities = distinctRollsCounts/numberOfThrows
    cumulativeRollsProbabilities = np.zeros(len(distinctRolls))
    for i in range(len(distinctRollsCounts)):
        if i == 0:
            cumulativeRollsProbabilities[i] = distinctRollsProbabilities[i]
        else:
            cumulativeRollsProbabilities[i] = cumulativeRollsProbabilities[i - 1] + distinctRollsProbabilities[i]


    print(distinctRolls, distinctRollsCounts, distinctRollsProbabilities)
    plt.bar(distinctRolls, distinctRollsProbabilities)
    plt.xlabel('Number')
    plt.ylabel('Probability')
    plt.show()

    plt.bar(distinctRolls, cumulativeRollsProbabilities)
    plt.xlabel('Number')
    plt.ylabel('Cumulative Probability')

    plt.show()






main()


