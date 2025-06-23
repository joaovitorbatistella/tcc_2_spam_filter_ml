import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_recall_curve, roc_curve, auc, confusion_matrix,
    classification_report
)
from sklearn.model_selection import validation_curve, learning_curve
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

dirname = os.path.dirname(__file__)

class BinaryClassificationPlot:
    """
    Class for complete analysis of binary classifiers
    """
    
    def __init__(self, model, algorithm_enum, X_train, X_test, y_train, y_test, output_base_path='output',
                 feature_names=None, class_names=None, y_pred=None, y_proba=None):
        """
        Inicializa o analisador
        
        Parameters:
        -----------
        model : sklearn estimator
            Modelo treinado
        algorithm_enum : AlgorithmEnum
            Algorithm used
        X_train, X_test : array-like
            Dados de treino e teste
        y_train, y_test : array-like
            Labels de treino e teste
        output_base_path : string
            Output base path
        feature_names : list, optional
            Nomes das features
        class_names : list, optional
            Nomes das classes
        """
        self.model = model
        self.algorithm_enum = algorithm_enum
        self.output_base_path = output_base_path
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.feature_names = feature_names
        self.class_names = class_names or ['Classe 0', 'Classe 1']
        
        # Predições
        self.y_pred = y_pred
        self.y_proba = y_proba
        
    def plot_precision_recall_curve(self, figsize=(10, 6)):
        """
        Plota a curva Precision-Recall
        """
        precision, recall, thresholds = precision_recall_curve(self.y_test, self.y_proba)
        pr_auc = auc(recall, precision)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Curva PR
        ax1.plot(recall, precision, color='blue', lw=2, 
                label=f'PR Curve (AUC = {pr_auc:.3f})')
        ax1.fill_between(recall, precision, alpha=0.2, color='blue')
        ax1.set_xlabel('Recall')
        ax1.set_ylabel('Precision')
        ax1.set_title('Precision-Recall Curve')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Threshold analysis
        ax2.plot(thresholds, precision[:-1], label='Precision', color='red')
        ax2.plot(thresholds, recall[:-1], label='Recall', color='blue')
        ax2.set_xlabel('Threshold')
        ax2.set_ylabel('Score')
        ax2.set_title('Precision vs Recall por Threshold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.suptitle(f"{self.algorithm_enum.get_name()}", fontsize=16, fontweight='bold')

        plt.savefig(f"{self.output_base_path}/plot_precision_recall_curve.png")
        # plt.show()
        
        return pr_auc
    
    def plot_roc_curve(self, figsize=(8, 6)):
        """
        Plota a curva ROC
        """
        fpr, tpr, thresholds = roc_curve(self.y_test, self.y_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=figsize)
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC Curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random Classifier')
        plt.fill_between(fpr, tpr, alpha=0.2, color='darkorange')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Taxa de Falsos Positivos')
        plt.ylabel('Taxa de Verdadeiros Positivos')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.suptitle(f"{self.algorithm_enum.get_name()}", fontsize=16, fontweight='bold')

        plt.savefig(f"{self.output_base_path}/plot_roc_curve.png")
        # plt.show()
        
        return roc_auc
    
    def plot_class_distribution(self, figsize=(12, 5)):
        """
        Plota a distribuição das classes
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Distribuição no conjunto de treino
        train_counts = pd.Series(self.y_train).value_counts().sort_index()
        train_counts.plot(kind='bar', ax=axes[0], color=['skyblue', 'lightcoral'])
        axes[0].set_title('Distribuição das Classes - Treino')
        axes[0].set_xlabel('Classe')
        axes[0].set_ylabel('Contagem')
        axes[0].set_xticklabels(self.class_names, rotation=0)
        
        # Adicionar percentuais
        total_train = len(self.y_train)
        for i, v in enumerate(train_counts.values):
            axes[0].text(i, v + total_train*0.01, f'{v}\n({v/total_train*100:.1f}%)', 
                        ha='center', va='bottom')
        
        # Distribuição no conjunto de teste
        test_counts = pd.Series(self.y_test).value_counts().sort_index()
        test_counts.plot(kind='bar', ax=axes[1], color=['skyblue', 'lightcoral'])
        axes[1].set_title('Distribuição das Classes - Teste')
        axes[1].set_xlabel('Classe')
        axes[1].set_ylabel('Contagem')
        axes[1].set_xticklabels(self.class_names, rotation=0)
        
        # Adicionar percentuais
        total_test = len(self.y_test)
        for i, v in enumerate(test_counts.values):
            axes[1].text(i, v + total_test*0.01, f'{v}\n({v/total_test*100:.1f}%)', 
                        ha='center', va='bottom')
        
        plt.tight_layout()
        plt.suptitle(f"{self.algorithm_enum.get_name()}", fontsize=16, fontweight='bold')

        plt.savefig(f"{self.output_base_path}/plot_class_distribution.png")
        # plt.show()
        
        return train_counts, test_counts
    
    def analyze_tfidf_features(self, top_n=20, figsize=(15, 10)):
        """
        Análise das features TF-IDF
        """
        
        # Obter coeficientes do modelo (assumindo que é logístico)
        try:
            if hasattr(self.model, 'coef_'):
                coefficients = self.model.coef_[0]
            elif hasattr(self.model, 'named_steps') and 'classifier' in self.model.named_steps:
                coefficients = self.model.named_steps['classifier'].coef_[0]
            else:
                print("Modelo não possui coeficientes acessíveis")
                return
        except:
            print("Modelo não possui coeficientes acessíveis")
            return
        
        # Criar DataFrame com features e coeficientes
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': coefficients,
            'abs_coefficient': np.abs(coefficients)
        }).sort_values('abs_coefficient', ascending=False)
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Top features positivas
        top_positive = feature_importance.head(top_n//2)
        axes[0, 0].barh(range(len(top_positive)), top_positive['coefficient'], 
                       color='green', alpha=0.7)
        axes[0, 0].set_yticks(range(len(top_positive)))
        axes[0, 0].set_yticklabels(top_positive['feature'])
        axes[0, 0].set_title(f'Top {top_n//2} Features Positivas')
        axes[0, 0].set_xlabel('Coeficiente')
        
        # Top features negativas
        top_negative = feature_importance.tail(top_n//2)
        axes[0, 1].barh(range(len(top_negative)), top_negative['coefficient'], 
                       color='red', alpha=0.7)
        axes[0, 1].set_yticks(range(len(top_negative)))
        axes[0, 1].set_yticklabels(top_negative['feature'])
        axes[0, 1].set_title(f'Top {top_n//2} Features Negativas')
        axes[0, 1].set_xlabel('Coeficiente')
        
        # Distribuição dos coeficientes
        axes[1, 0].hist(coefficients, bins=50, alpha=0.7, color='blue')
        axes[1, 0].set_title('Distribuição dos Coeficientes')
        axes[1, 0].set_xlabel('Coeficiente')
        axes[1, 0].set_ylabel('Frequência')
        axes[1, 0].axvline(x=0, color='red', linestyle='--', alpha=0.7)
        
        # WordCloud das features mais importantes
        top_features = feature_importance.head(50)
        wordcloud_dict = dict(zip(top_features['feature'], 
                                 top_features['abs_coefficient']))
        
        wordcloud = WordCloud(width=400, height=300, 
                             background_color='white').generate_from_frequencies(wordcloud_dict)
        axes[1, 1].imshow(wordcloud, interpolation='bilinear')
        axes[1, 1].axis('off')
        axes[1, 1].set_title('WordCloud - Features Importantes')
        
        plt.tight_layout()
        plt.suptitle(f"{self.algorithm_enum.get_name()}", fontsize=16, fontweight='bold')

        plt.savefig(f"{self.output_base_path}/plot_analyze_tfidf_features.png")
        # plt.show()
        
        return feature_importance
    
    def plot_learning_curves(self, results):
        """
        Plota as curvas de aprendizado baseado no exemplo oficial do sklearn
        """
        # if train_sizes is None:
        #     train_sizes = np.linspace(0.1, 1.0, 10)
        
        # # Usar learning_curve com parâmetros corretos
        # if return_times:
        #     train_sizes_abs, train_scores, val_scores, fit_times, score_times = learning_curve(
        #         self.model, self.X_train, self.y_train, 
        #         cv=cv, train_sizes=train_sizes, 
        #         scoring=scoring, n_jobs=-1, return_times=True
        #     )
        # else:
        #     train_sizes_abs, train_scores, val_scores = learning_curve(
        #         self.model, self.X_train, self.y_train, 
        #         cv=cv, train_sizes=train_sizes, 
        #         scoring=scoring, n_jobs=-1
        #     )

        for metric in results:
            scores = results[metric]
            train_scores    = np.array([scores[percent]['train_scores'] for percent in scores])
            val_scores      = np.array([scores[percent]['val_scores'] for percent in scores])
            train_sizes_abs = np.array([scores[percent]['train_sizes_abs'] for percent in scores])
            
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            val_mean = np.mean(val_scores, axis=1)
            val_std = np.std(val_scores, axis=1)

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            axes = axes.reshape(1, -1)
            
            # Curvas de aprendizado (scores)
            axes[0, 0].plot(train_sizes_abs, train_mean, 'o-', color='blue', 
                        label='Score Treino')
            axes[0, 0].fill_between(train_sizes_abs, train_mean - train_std, 
                                train_mean + train_std, alpha=0.2, color='blue')
            
            axes[0, 0].plot(train_sizes_abs, val_mean, 'o-', color='red', 
                        label='Score Validação')
            axes[0, 0].fill_between(train_sizes_abs, val_mean - val_std, 
                                val_mean + val_std, alpha=0.2, color='red')
            
            axes[0, 0].set_xlabel('Tamanho do Conjunto de Treino')
            axes[0, 0].set_ylabel(f'Score {metric.upper()}')
            axes[0, 0].set_title('Curvas de Aprendizado')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Gap entre treino e validação
            gap = train_mean - val_mean
            axes[0, 1].plot(train_sizes_abs, gap, 'o-', color='purple')
            axes[0, 1].fill_between(train_sizes_abs, gap, alpha=0.2, color='purple')
            axes[0, 1].set_xlabel('Tamanho do Conjunto de Treino')
            axes[0, 1].set_ylabel('Gap (Treino - Validação)')
            axes[0, 1].set_title('Gap entre Treino e Validação')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            plt.suptitle(f"{self.algorithm_enum.get_name()}", fontsize=16, fontweight='bold')

            plt.savefig(f"{self.output_base_path}/plot_learning_curve_{metric}.png")

        # return train_sizes_abs, train_scores, val_scores
        
    def generate_feature_importance_report(self, top_n=20):
        """
        Gera relatório de importância das features
        """
        
        # Obter coeficientes
        try:
            if hasattr(self.model, 'coef_'):
                coefficients = self.model.coef_[0]
            elif hasattr(self.model, 'named_steps') and 'classifier' in self.model.named_steps:
                coefficients = self.model.named_steps['classifier'].coef_[0]
            else:
                print("Modelo não possui coeficientes acessíveis")
                return None
        except:
            print("Modelo não possui coeficientes acessíveis")
            return
        
        # Criar relatório
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': coefficients,
            'abs_coefficient': np.abs(coefficients),
            'impact': ['Positivo' if c > 0 else 'Negativo' for c in coefficients]
        }).sort_values('abs_coefficient', ascending=False)
        
        print("="*60)
        print("RELATÓRIO DE IMPORTÂNCIA DAS FEATURES")
        print("="*60)
        print(f"\nTotal de features: {len(self.feature_names)}")
        print(f"Features com coeficiente positivo: {sum(coefficients > 0)}")
        print(f"Features com coeficiente negativo: {sum(coefficients < 0)}")
        print(f"Features com coeficiente zero: {sum(coefficients == 0)}")
        
        print(f"\nTOP {top_n} FEATURES MAIS IMPORTANTES:")
        print("-"*60)
        for i, row in feature_importance.head(top_n).iterrows():
            print(f"{row['feature']:25} | {row['coefficient']:8.4f} | {row['impact']:8}")
        
        print(f"\nESTATÍSTICAS DOS COEFICIENTES:")
        print("-"*40)
        print(f"Média: {np.mean(coefficients):.4f}")
        print(f"Desvio padrão: {np.std(coefficients):.4f}")
        print(f"Mínimo: {np.min(coefficients):.4f}")
        print(f"Máximo: {np.max(coefficients):.4f}")
        
        return feature_importance
    
    def generate_complete_report(self):
        """
        Gera relatório completo do modelo
        """
        print("="*80)
        print("RELATÓRIO COMPLETO - CLASSIFICADOR BINÁRIO")
        print("="*80)
        
        # Métricas básicas
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(self.y_test, self.y_pred)
        precision = precision_score(self.y_test, self.y_pred)
        recall = recall_score(self.y_test, self.y_pred)
        f1 = f1_score(self.y_test, self.y_pred)
        
        print(f"\nMÉTRICAS DE PERFORMANCE:")
        print("-"*40)
        print(f"Acurácia: {accuracy:.4f}")
        print(f"Precisão: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        # Matriz de confusão
        cm = confusion_matrix(self.y_test, self.y_pred)
        print(f"\nMATRIZ DE CONFUSÃO:")
        print("-"*40)
        print(f"Verdadeiros Negativos: {cm[0,0]}")
        print(f"Falsos Positivos: {cm[0,1]}")
        print(f"Falsos Negativos: {cm[1,0]}")
        print(f"Verdadeiros Positivos: {cm[1,1]}")
        
        # Distribuição das classes
        train_dist = pd.Series(self.y_train).value_counts().sort_index()
        test_dist = pd.Series(self.y_test).value_counts().sort_index()
        
        print(f"\nDISTRIBUIÇÃO DAS CLASSES:")
        print("-"*40)
        print("Treino:")
        for i, count in enumerate(train_dist):
            print(f"  {self.class_names[i]}: {count} ({count/len(self.y_train)*100:.1f}%)")
        print("Teste:")
        for i, count in enumerate(test_dist):
            print(f"  {self.class_names[i]}: {count} ({count/len(self.y_test)*100:.1f}%)")
        
        # Features importantes (se disponível)
        if self.feature_names is not None:
            self.generate_feature_importance_report(top_n=10)