import os
import csv
import argparse
import numpy as np
from itertools import islice
from PIL import Image


ferplus_path = './data/fer2013new.csv'

similarity = np.zeros((7,7))

with open(ferplus_path, 'r') as csvfile:
    ferplus_rows = csv.reader(csvfile, delimiter=',')

    total = 35887
    for row in islice(ferplus_rows, 1, None):

        for i in range(2, 9):
            for j in range(i, 9):
                if i==j:
                    similarity[i - 2][j - 2] += 1/total

                elif int(row[i]) != 0 and int(row[j]) != 0:
                    similarity[i-2][j-2] += min(int(row[i]),int(row[j]))/(max(int(row[i]),int(row[j]))*total)

    print(similarity)



