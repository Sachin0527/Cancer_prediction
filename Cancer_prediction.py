#import all neccessary labries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import sklearn.linear_model as skl_lm
from sklearn import preprocessing
from sklearn import neighbors
from sklearn.metrics import confusion_matrix, classification_report, precision_score
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import statsmodels.formula.api as smf
sns.set(style="whitegrid", color_codes=True, font_scale=1.3)


#reading data from the file
df = pd.read_csv('data.csv', index_col=0)
df.head()
df.info()
df = df.drop('Unnamed: 32', axis=1)
df.dtypes

plt.figure(figsize=(8, 4))
sns.countplot(df['diagnosis'], palette='RdBu')
benign, malignant = df['diagnosis'].value_counts()
print('Number of cells labeled Benign: ', benign)
print('Number of cells labeled Malignant : ', malignant)
print('')
print('% of cells labeled Benign', round(benign / len(df) * 100, 2), '%')
print('% of cells labeled Malignant', round(malignant / len(df) * 100, 2), '%')




