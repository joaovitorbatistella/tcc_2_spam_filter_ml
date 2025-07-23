import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

dirname = os.path.dirname(__file__)

texto_column = 'text'

# Lendo o arquivo CSV
df = pd.read_csv(f"{dirname}/../input/spamassassin.csv", encoding='utf-8')
# print(f"Arquivo CSV carregado com sucesso!")
# print(f"Forma do dataset: {df.shape}")
# print(f"Colunas disponíveis: {list(df.columns)}")

# Verificar se a coluna de texto existe
if texto_column not in df.columns:
    # print(f"\nATENÇÃO: Coluna '{texto_column}' não encontrada!")
    # print(f"Colunas disponíveis: {list(df.columns)}")
    # print("Por favor, ajuste o nome da variável 'texto_column' para a coluna correta.")
    # Usar a primeira coluna como fallback
    texto_column = df.columns[0]
    # print(f"Usando a primeira coluna como texto: '{texto_column}'")

# Verificar dados faltantes
missing_count = df[texto_column].isna().sum()
if missing_count > 0:
    # print(f"\nAviso: {missing_count} valores faltantes na coluna '{texto_column}'")
    df = df.dropna(subset=[texto_column])
    # print(f"Linhas após remover valores faltantes: {len(df)}")

# Converter para string e limpar
df[texto_column] = df[texto_column].astype(str)

# print("Dataset carregado:")
# print(df.head())
# print("\n" + "="*50 + "\n")

# Aplicando TF-IDF
vectorizer = TfidfVectorizer(
    min_df=0.07,
    max_df=0.7,
    norm='l2',
    stop_words='english',
    sublinear_tf=True,
)

# Ajustando o vectorizer e transformando os textos
tfidf_matrix = vectorizer.fit_transform(df[texto_column])

# Obtendo os nomes das features (palavras)
feature_names = vectorizer.get_feature_names_out()

# print(f"Forma da matriz TF-IDF: {tfidf_matrix.shape}")
# print(f"Número de palavras únicas: {len(feature_names)}")
# print("\n" + "="*50 + "\n")

# Função para buscar o score de uma palavra específica
def buscar_score_palavra(palavra, vectorizer, tfidf_matrix, textos):
    """
    Busca o score TF-IDF de uma palavra específica em todos os documentos
    """
    palavra = palavra.lower()
    
    # Verificar se a palavra existe no vocabulário
    if palavra not in vectorizer.vocabulary_:
        print(f"A palavra '{palavra}' não foi encontrada no vocabulário.")
        return None
    
    # Obter o índice da palavra
    word_index = vectorizer.vocabulary_[palavra]
    
    print(f"Scores TF-IDF para a palavra '{palavra}':")
    print("-" * 40)
    
    # Obter os scores para cada documento
    scores = []
    for i in range(tfidf_matrix.shape[0]):
        score = tfidf_matrix[i, word_index]
        scores.append(score)
        if(score>0):
            print(f"Documento {i+1}: {score:.4f}")
            print(f"Texto: '{textos.iloc[i][:60]}...'")
            print()
    
    # Estatísticas
    scores_array = np.array(scores)
    print(f"Score máximo: {scores_array.max():.4f}")
    print(f"Score mínimo: {scores_array.min():.4f}")
    print(f"Score médio: {scores_array.mean():.4f}")
    
    return scores

# Exemplo de uso - buscar score para uma palavra específica
palavra_busca = "sortbysize"  # Altere para a palavra que você quer buscar

print(f"Buscando scores para a palavra: '{palavra_busca}'")
print("="*50)

scores = buscar_score_palavra(palavra_busca, vectorizer, tfidf_matrix, df[texto_column])