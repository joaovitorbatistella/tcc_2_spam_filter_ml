import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Vamos supor que temos um arquivo 'textos.csv' com colunas 'texto' e 'categoria'
# Leitura do arquivo CSV
# df = pd.read_csv('./input/spam_ham_dataset.csv', encoding='utf-8')
df = pd.read_csv('./input/spam.csv', encoding='utf-8')

# Separar features (X) e target (y)
X = df['text']
y = df['label']

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Criar pipeline com TF-IDF e Regressão Logística
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        min_df=2,
        max_df=0.95,
        stop_words='english',
        lowercase=True,
        norm='l2',
        ngram_range=(1, 1)
    )),
    ('clf', LogisticRegression(
        C=1.0,  # smaller values specify stronger regularization
        class_weight='balanced',
        max_iter=1000,
        random_state=42,
        n_jobs=-1 # This parameter is used to specify how many concurrent processes or threads should be used for routines that are parallelized with joblib.
    )),
    # ('clf', MultinomialNB(
    #     alpha=1.0,  # Smoothing parameter (Laplace/Lidstone smoothing)
    #     fit_prior=True,  # Learn class prior probabilities
    #     class_prior=None  # Prior probabilities of the classes
    # ))
])

# Treinar o modelo
pipeline.fit(X_train, y_train)

# Fazer predições
y_pred = pipeline.predict(X_test)

# Avaliar o modelo
print("Relatório de Classificação: (classification_report)")
print(classification_report(y_test, y_pred))

print("Relatório de Classificação (confusion_matrix):")
print(confusion_matrix(y_test, y_pred))

# Validação cruzada
cv_scores = cross_val_score(pipeline, X, y, cv=5)
print("\nScores da Validação Cruzada:")
print(f"Acurácia média: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

# Salvar o modelo treinado
import joblib
joblib.dump(pipeline, './output/modelo_tfidf_logreg.joblib')