import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime

# Dados fictícios
data = {
    'SPAM Precision':[0.9845, 0.9288, 0.9891, 0.9781, 0.9739, 0.9974],
    'SPAM Recall':   [0.8523, 0.9567, 0.9552, 0.8993, 0.9933, 0.9934],
    'SPAM F1':       [0.9137, 0.9425, 0.9718, 0.9371, 0.9835, 0.9954],
    'HAM Precision': [0.9776, 0.9820, 0.9760, 0.9846, 0.9972, 0.9964],
    'HAM Recall':    [0.9979, 0.9700, 0.9942, 0.9968, 0.9985, 0.9985],
    'HAM F1':        [0.9876, 0.9759, 0.9850, 0.9906, 0.9931, 0.9974],
    'Accuracy':      [0.9785, 0.9662, 0.9805, 0.9839, 0.9903, 0.9967],
}

# Criar DataFrame
df = pd.DataFrame(
    data,
    index=[
        'NB D1 T=0.6',
        'NB D2 T=0.5',
        'NB D3 T=0.5',
        'LR D1 T=0.7',
        'LR D2 T=0.5',
        'LR D3 T=0.7'
    ]
)

# Configurar o heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df, annot=True, cmap='YlGnBu', fmt='.4f')
plt.title('Heatmap: Algoritmos por Dataset vs Métricas')
plt.xlabel('Métricas')
plt.ylabel('Algoritmo (Dataset) T=threshold')
plt.tight_layout()

dirname = os.path.dirname(__file__)
_datetime               = datetime.now().strftime("%Y%m%d-%H%M")
base_path               = f"{dirname}/../output/vizualizations/{_datetime}"
os.makedirs(base_path, exist_ok=True)

# Exibir o heatmap
plt.savefig(f"{base_path}/heatmap.png")