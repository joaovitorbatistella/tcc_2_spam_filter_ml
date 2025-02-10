import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# 1. Carregamento e preparação dos dados
print("1. Carregando os dados...")
df = pd.read_csv('./input/spam_ham_dataset.csv', encoding='utf-8')
X = df['text']
y = df['label']

# 2. Divisão treino/teste
print("\n2. Dividindo em treino e teste...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Criação e treinamento do TF-IDF
print("\n3. Criando e treinando o TF-IDF...")
tfidf = TfidfVectorizer(
    min_df=2,  # ignore termos que aparecem em menos de 2 documentos
    max_df=0.95,  # ignore termos que aparecem em mais de 95% dos documentos
    stop_words='english',
    lowercase=True,
    ngram_range=(1, 1)
)

# Fit e transform nos dados de treino
X_train_tfidf = tfidf.fit_transform(X_train)

# Transform nos dados de teste (usando o vocabulário aprendido no treino)
X_test_tfidf = tfidf.transform(X_test)

# 4. Análise do vocabulário e features
print("\n4. Analisando o vocabulário...")
vocab_df = pd.DataFrame({
    'termo': tfidf.get_feature_names_out(),
    'idf': tfidf.idf_
}).sort_values('idf', ascending=True)

print(f"Tamanho do vocabulário: {len(vocab_df)}")
print("\nPalavras mais raras (maior IDF):")
print(vocab_df.nlargest(10, 'idf')[['termo', 'idf']].to_string())
print("\nPalavras mais comuns (menor IDF):")
print(vocab_df.nsmallest(10, 'idf')[['termo', 'idf']].to_string())

# 5. Visualização da matriz TF-IDF de forma textual
print("\n5. Analisando os primeiros exemplos da matriz TF-IDF...")
# Pegando os primeiros 3 documentos e 10 features para exemplo
dense_matrix = X_train_tfidf[:3, :10].toarray()
feature_names = tfidf.get_feature_names_out()[:10]

print("\nPrimeiros 3 documentos x 10 features:")
df_exemplo = pd.DataFrame(dense_matrix, columns=feature_names)
print(df_exemplo.round(3).to_string())

# 6. Treinamento do Naive Bayes
print("\n6. Treinando o Naive Bayes...")
nb_classifier = MultinomialNB(alpha=1.0)

# print("\n6. Treinando a Regressão Logística...")
# log_reg = LogisticRegression(
#                                 C=1.0,  # smaller values specify stronger regularization
#                                 class_weight='balanced',
#                                 max_iter=1000,
#                                 random_state=42,
#                                 n_jobs=-1 # This parameter is used to specify how many concurrent processes or threads should be used for routines that are parallelized with joblib.
#                             )

nb_classifier.fit(X_train_tfidf, y_train)

# 7. Avaliação do modelo
print("\n7. Avaliando o modelo...")
y_pred = nb_classifier.predict(X_test_tfidf)

print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))

# 8. Matriz de confusão em formato texto
print("\n8. Matriz de Confusão:")
cm = confusion_matrix(y_test, y_pred)
print("\nMatriz de Confusão em formato texto:")
print(pd.DataFrame(cm, 
                  columns=['Pred Não-Spam', 'Pred Spam'],
                  index=['Real Não-Spam', 'Real Spam']))

# 9. Análise das palavras mais importantes para cada classe
print("\n9. Analisando palavras mais importantes para cada classe...")
feature_importance = pd.DataFrame({
    'feature': tfidf.get_feature_names_out(),
    # 'importance': log_reg.coef_[0]
    'importance': nb_classifier.feature_log_prob_[1] - nb_classifier.feature_log_prob_[0]
})

print("\nPalavras mais indicativas de spam:")
print(feature_importance.nlargest(30, 'importance').to_string())
print("\nPalavras mais indicativas de não-spam:")
print(feature_importance.nsmallest(30, 'importance').to_string())

# # 10. Exemplo de classificação
# print("\n10. Exemplo de classificação de novos textos:")
# def classificar_texto(texto):
#     # Transformar o texto usando o mesmo TF-IDF
#     texto_tfidf = tfidf.transform([texto])
#     # Fazer a predição
#     predicao = nb_or_logreg_classifier.predict(texto_tfidf)
#     # Obter as probabilidades
#     probs = nb_or_logreg_classifier.predict_proba(texto_tfidf)
#     return predicao[0], probs[0]

# # Testando com alguns exemplos
# exemplos = [
#     "Congratulations! You've won a free iPhone! Click here to claim your prize!",
#     "Hi John, can we schedule the meeting for tomorrow at 2 PM?",
#     "URGENT: Your account has been suspended. Click here to verify."
# ]

# print("\nClassificando exemplos:")
# for texto in exemplos:
#     pred, probs = classificar_texto(texto)
#     print(f"\nTexto: {texto}")
#     print(f"Classificação: {pred}")
#     print(f"Probabilidades: Não-spam: {probs[0]:.3f}, Spam: {probs[1]:.3f}")