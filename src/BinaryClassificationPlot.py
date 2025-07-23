import numpy as np
import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from scipy.special import expit
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

        # Configurar o logging
        logging.basicConfig(
            filename=f"{self.output_base_path}/stdout.log",
            filemode='w',  # 'w' para sobrescrever, 'a' para adicionar
            format='%(asctime)s - %(levelname)s - %(message)s',
            level=logging.INFO,
            encoding='utf-8'
        )
        
    def plot_precision_recall_curve(self, figsize=(10, 6)):
        """
        Plota a curva Precision-Recall
        """
        precision, recall, thresholds = precision_recall_curve(self.y_test, self.y_proba)
        pr_auc = auc(recall, precision)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Curva PR
        ax1.plot(recall, precision, color='blue', lw=2, 
                label=f'PR Curve (AUC = {pr_auc:.8f})')
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
                label=f'ROC Curve (AUC = {roc_auc:.8f})')
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
    
    def analyze_tfidf_features(self, top_n=20, figsize=(10, 8)):
        """
        Análise das features TF-IDF - Gera 4 imagens individuais
        """
        
        # Obter coeficientes do modelo (assumindo que é logístico)
        try:
            if hasattr(self.model, 'coef_'):
                coefficients = self.model.coef_[0]
            elif hasattr(self.model, 'named_steps') and 'classifier' in self.model.named_steps:
                coefficients = self.model.named_steps['classifier'].coef_[0]
            else:
                logging.info("Modelo não possui coeficientes acessíveis")
                return
        except:
            logging.info("Modelo não possui coeficientes acessíveis")
            return
        
        # Criar DataFrame com features e coeficientes
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': coefficients,
            'abs_coefficient': np.abs(coefficients)
        }).sort_values('abs_coefficient', ascending=False)
        
        algorithm_name = self.algorithm_enum.get_name()
        
        # 1. Gráfico de Features Positivas
        plt.figure(figsize=figsize)
        positive_features = feature_importance[feature_importance['coefficient'] > 0].head(top_n//2)
        plt.barh(range(len(positive_features)), positive_features['coefficient'], 
                color='green', alpha=0.7)
        plt.yticks(range(len(positive_features)), positive_features['feature'])
        plt.title(f'{algorithm_name} - Top {len(positive_features)} Features Positivas', 
                fontsize=14, fontweight='bold')
        plt.xlabel('Coeficiente')
        plt.tight_layout()
        plt.savefig(f"{self.output_base_path}/plot_tfidf_features_positivas.png", 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Gráfico de Features Negativas
        plt.figure(figsize=figsize)
        negative_features = feature_importance[feature_importance['coefficient'] < 0].tail(top_n//2)
        plt.barh(range(len(negative_features)), negative_features['coefficient'], 
                color='red', alpha=0.7)
        plt.yticks(range(len(negative_features)), negative_features['feature'])
        plt.title(f'{algorithm_name} - Top {len(negative_features)} Features Negativas', 
                fontsize=14, fontweight='bold')
        plt.xlabel('Coeficiente')
        plt.tight_layout()
        plt.savefig(f"{self.output_base_path}/plot_tfidf_features_negativas.png", 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Distribuição dos Coeficientes
        plt.figure(figsize=figsize)
        plt.hist(coefficients, bins=50, alpha=0.7, color='blue')
        plt.title(f'{algorithm_name} - Distribuição dos Coeficientes', 
                fontsize=14, fontweight='bold')
        plt.xlabel('Coeficiente')
        plt.ylabel('Frequência')
        plt.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f"{self.output_base_path}/plot_tfidf_distribuicao_coeficientes.png", 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. WordCloud das Features Importantes
        plt.figure(figsize=figsize)
        top_features = feature_importance.head(50)
        wordcloud_dict = dict(zip(top_features['feature'], 
                                top_features['abs_coefficient']))
        
        wordcloud = WordCloud(
                        width=800,
                        height=600, 
                        background_color='white',
                    ).generate_from_frequencies(wordcloud_dict)
        
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'{algorithm_name} - WordCloud Features Importantes', 
                fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{self.output_base_path}/plot_tfidf_wordcloud.png", 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"4 imagens de análise TF-IDF salvas em {self.output_base_path}")
        
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
                logging.info("Modelo não possui coeficientes acessíveis")
                return None
        except:
            logging.info("Modelo não possui coeficientes acessíveis")
            return
        
        # Criar relatório
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': coefficients,
            'abs_coefficient': np.abs(coefficients),
            'impact': ['Positivo' if c > 0 else 'Negativo' for c in coefficients]
        }).sort_values('abs_coefficient', ascending=False)
        
        logging.info("="*60)
        logging.info("RELATÓRIO DE IMPORTÂNCIA DAS FEATURES")
        logging.info("="*60)
        logging.info("\n")
        logging.info(f"Total de features: {len(self.feature_names)}")
        logging.info(f"Features com coeficiente positivo: {sum(coefficients > 0)}")
        logging.info(f"Features com coeficiente negativo: {sum(coefficients < 0)}")
        logging.info(f"Features com coeficiente zero: {sum(coefficients == 0)}")
        
        logging.info("\n")
        logging.info(f"TOP {top_n} FEATURES MAIS IMPORTANTES:")
        logging.info("-"*60)
        for i, row in feature_importance.head(top_n//2).iterrows():
            logging.info(f"{row['feature']:25} | {row['coefficient']:8.4f} | {row['impact']:8}")
        for i, row in feature_importance.tail(top_n//2).iterrows():
            logging.info(f"{row['feature']:25} | {row['coefficient']:8.4f} | {row['impact']:8}")
        
        logging.info("\n")
        logging.info(f"ESTATÍSTICAS DOS COEFICIENTES:")
        logging.info("-"*40)
        logging.info(f"Média: {np.mean(coefficients):.8f}")
        logging.info(f"Desvio padrão: {np.std(coefficients):.8f}")
        logging.info(f"Mínimo: {np.min(coefficients):.8f}")
        logging.info(f"Máximo: {np.max(coefficients):.8f}")
        
        return feature_importance
    
    def generate_complete_report(self):
        """
        Gera relatório completo do modelo
        """
        logging.info("="*80)
        logging.info("RELATÓRIO COMPLETO - CLASSIFICADOR BINÁRIO")
        logging.info("="*80)
        
        # Métricas básicas
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(self.y_test, self.y_pred)
        precision = precision_score(self.y_test, self.y_pred)
        recall = recall_score(self.y_test, self.y_pred)
        f1 = f1_score(self.y_test, self.y_pred)
        
        logging.info("\n")
        logging.info(f"MÉTRICAS DE PERFORMANCE:")
        logging.info("-"*40)
        logging.info(f"Acurácia: {accuracy:.8f}")
        logging.info(f"Precisão: {precision:.8f}")
        logging.info(f"Recall: {recall:.8f}")
        logging.info(f"F1-Score: {f1:.8f}")
        
        # Matriz de confusão
        tn, fp, fn, tp = confusion_matrix(self.y_test, self.y_pred).ravel()
        logging.info("\n")
        logging.info(f"MATRIZ DE CONFUSÃO:")
        logging.info("-"*40)
        logging.info(f"Verdadeiros Negativos: {tn}")
        logging.info(f"Falsos Positivos: {fp}")
        logging.info(f"Falsos Negativos: {fn}")
        logging.info(f"Verdadeiros Positivos: {tp}")
        
        # Distribuição das classes
        train_dist = pd.Series(self.y_train).value_counts().sort_index()
        test_dist = pd.Series(self.y_test).value_counts().sort_index()
        
        logging.info("\n")
        logging.info(f"DISTRIBUIÇÃO DAS CLASSES:")
        logging.info("-"*40)
        logging.info("Treino:")
        for i, count in enumerate(train_dist):
            logging.info(f"  {self.class_names[i]}: {count} ({count/len(self.y_train)*100:.1f}%)")
        logging.info("Teste:")
        for i, count in enumerate(test_dist):
            logging.info(f"  {self.class_names[i]}: {count} ({count/len(self.y_test)*100:.1f}%)")
        
        # Features importantes (se disponível)
        if self.feature_names is not None:
            self.generate_feature_importance_report(top_n=16)

    def _old_plot_logistic_regression(self):
        """
        Plot the logistic regression curve or decision boundary using the model's data.
        Saves the plot to the output_base_path.
        """

        if self.algorithm_enum.value != 'logistic-regression':
            logging.info(f"Este plot é específico para Regressão Logística. Algoritmo atual: {self.algorithm_enum.value}")
            return
        
        # Convert inputs to NumPy arrays if they are lists
        X_train_raw = np.array(self.X_train, dtype=object)
        X_test_raw = np.array(self.X_test, dtype=object)
        y_train = np.array(self.y_train)
        y_test = np.array(self.y_test)
        
        # Check if self.model is a Pipeline
        if not isinstance(self.model, Pipeline):
            raise ValueError("self.model must be a scikit-learn Pipeline with TfidfVectorizer and LogisticRegression.")
        
        # Extract TfidfVectorizer and LogisticRegression from the pipeline
        vectorizer = None
        classifier = None
        for name, step in self.model.named_steps.items():
            if isinstance(step, TfidfVectorizer):
                vectorizer = step
            elif isinstance(step, LogisticRegression):
                classifier = step
        if vectorizer is None:
            raise ValueError("Pipeline does not contain a TfidfVectorizer step.")
        if classifier is None:
            raise ValueError("Pipeline does not contain a LogisticRegression step.")
        
        # Transform raw text to TF-IDF features
        if isinstance(X_train_raw[0], str):
            X_train = vectorizer.transform(X_train_raw)
            X_test = vectorizer.transform(X_test_raw)
        else:
            X_train = X_train_raw
            X_test = X_test_raw
        
        # Check if data is numeric
        if not np.issubdtype(X_train.dtype, np.number):
            raise ValueError("X_train contains non-numeric data after vectorization. Ensure input is valid text.")
        
        # Compute decision function scores for training and test data
        train_scores = classifier.decision_function(X_train)
        test_scores = classifier.decision_function(X_test)
        
        # Create plot
        plt.figure(figsize=(10, 6))
        
        # Scatter plot of training and test data
        plt.scatter(train_scores, y_train, color='blue', label='Training data', alpha=0.6)
        plt.scatter(test_scores, y_test, color='black', label='example data', zorder=20)
        
        # Generate points for the sigmoid curve
        score_range = np.linspace(min(train_scores.min(), test_scores.min()),
                                max(train_scores.max(), test_scores.max()), 300)
        # Compute probabilities using the logistic function: 1 / (1 + exp(-x))
        probabilities = 1 / (1 + np.exp(-score_range))
        
        # Plot sigmoid curve
        plt.plot(score_range, probabilities, color='green', label='Logistic regression curve')
        
        # Add decision boundary (threshold = 0, corresponding to probability = 0.5)
        plt.axvline(x=0, color='gray', linestyle='--', label='Decision boundary (p=0.5)')
        plt.axhline(y=0.5, color='gray', linestyle='--')
        
        plt.xlabel('Decision Function Score')
        plt.ylabel('Probability of SPAM')
        plt.title('Logistic Regression Sigmoid Curve')
        plt.legend()
        
        # Save plot
        output_path = os.path.join(self.output_base_path, 'logistic_regression_sigmoid_curve.png')
        plt.savefig(output_path)
        plt.close()