import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from IPython.display import display
from sklearn.decomposition import PCA
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv("./input/spam_ham_dataset.csv")
display(df.head())

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(raw_documents=df['text']).toarray()

pca = PCA(n_components=2)

# Name of Vector Array (Numpy)
name_of_vector_array = X

# New D2 Dataframe (PCA)
df2d = pd.DataFrame(pca.fit_transform(name_of_vector_array), columns=list('xy'))

display(list('xy'))

# # Plot Data Visualization (Matplotlib)
# df2d.plot(kind='scatter', x='x', y='y')
# plt.show()