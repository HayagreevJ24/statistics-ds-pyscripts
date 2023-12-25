import math
import scipy.stats
import matplotlib.pyplot
import numpy


data = numpy.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=int)

sumX = 0
sumXSquared = 0

for i in data:
    sumX += i
    sumXSquared += (i ** 2)

mean = sumX/len(data)
biasedStandardDev = math.sqrt(sumXSquared/len(data) - (mean ** 2))

print(mean, biasedStandardDev)

print(scipy.stats.norm.ppf(0.99, loc=mean, scale=biasedStandardDev))