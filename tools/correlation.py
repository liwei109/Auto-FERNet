import pandas as pd

darts = pd.DataFrame({'search rank': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'darts': [4, 6, 1, 2, 3, 10, 5, 8, 9, 7]})

print(darts.corr())

print(darts.corr('spearman'))

print(darts.corr('kendall'))


sgas = pd.DataFrame({'search rank': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'sgas': [4, 6, 1, 2, 3, 10, 5, 8, 9, 7]})

print(sgas.corr())

print(sgas.corr('spearman'))

print(sgas.corr('kendall'))