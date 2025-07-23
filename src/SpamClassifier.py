import pandas as pd
import os
import logging
import html2text
from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve, f1_score, recall_score, precision_score
from sklearn.model_selection import validation_curve, learning_curve
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns
import eli5
import string
import sys
import joblib
from AlgorithmEnum import Algorithm as AlgorithmEnum
from BinaryClassificationPlot import BinaryClassificationPlot
import re

# temporary
from sklearn.datasets import fetch_20newsgroups

# Settings for views
plt.style.use('default')
sns.set_palette("husl")

dirname = os.path.dirname(__file__)

class SpamClassifier:
    def __init__(self, algorithm_enum=AlgorithmEnum('logistic-regression'), scoring='accuracy', input_file_name="spam.csv"):
        _datetime               = datetime.now().strftime("%Y%m%d-%H%M")
        base_path               = f"{dirname}/../output/{input_file_name.replace('.csv', '')}/{algorithm_enum.value}/scoring_{scoring}/{_datetime}"
        os.makedirs(base_path, exist_ok=True)

        self.algorithm_enum     = algorithm_enum
        self.input_file_name    = input_file_name
        self.output_base_path   = base_path
        self.scoring            = scoring
        self.pipeline           = None
        self.best_params        = None
        self.cv_scores          = None
        self.plot               = None

    def remove_spamassassin_terms(self, text):
        # Remove links containing “spamassassin”
        text = re.sub(r'https?://\S*spamassassin\S*', '', text, flags=re.IGNORECASE)

        return text

    def remove_words_regex(self, text, words_to_remove, case_sensitive=False):
        """
        Versão alternativa usando regex para remoção mais precisa.
        
        Args:
            text (str): Texto original
            words_to_remove (list): Lista de palavras a serem removidas
            case_sensitive (bool): Se True, considera maiúsculas/minúsculas. Default: False
        
        Returns:
            str: Texto com as palavras removidas
        """
        import re
        
        if not text or not words_to_remove:
            return text
        
        escaped_words = [re.escape(word) for word in words_to_remove]
        
        pattern = r'\b(?:' + '|'.join(escaped_words) + r')\b'
        
        flags = re.IGNORECASE if not case_sensitive else 0
        
        result = re.sub(pattern, '', text, flags=flags)
        
        result = re.sub(r'\s+', ' ', result).strip()
        
        return result
    
    def remove_after_phrase(self, text, phrase="email is sponsored by", case_sensitive=False, include_phrase=True):
        """
        Remove tudo que estiver depois de uma frase específica no texto.
        
        Args:
            text (str): Texto original
            phrase (str): Frase a partir da qual remover o resto do texto
            case_sensitive (bool): Se True, considera maiúsculas/minúsculas. Default: False
            include_phrase (bool): Se True, remove também a frase encontrada. Default: True
        
        Returns:
            str: Texto até a frase especificada (incluindo ou não a frase)
        """
        if not text or not phrase:
            return text
        
        # Prepara o texto para busca
        search_text = text if case_sensitive else text.lower()
        search_phrase = phrase if case_sensitive else phrase.lower()
        
        # Encontra a posição da frase
        position = search_text.find(search_phrase)
        
        # Se a frase não foi encontrada, retorna o texto original
        if position == -1:
            return text
        
        # Se deve incluir a frase, corta antes dela
        if not include_phrase:
            return text[:position + len(phrase)]
        else:
            return text[:position]
        
    def remove_urls(self, text):
        # Padrão para identificar URLs
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        # Substitui URLs por uma string vazia
        return re.sub(url_pattern, '', text)

    def remove_stops(self, text, stops):
        # Remove weird formatting characters
        h = html2text.HTML2Text()
        h.ignore_links = True

        text = h.handle(text)

        text = re.sub(r'[\n\r\t]|\\n|\\r|\\t', ' ', text)

        text = re.sub(r'\s+', ' ', text).strip()

        text = self.remove_after_phrase(text, 'email is sponsored by')

        text = self.remove_words_regex(text, ['sightings', 'sight', 'spamassassin', 'DeathToSpamDeathToSpamDeathToSpam', 'remove', 'removed'])

        text = self.remove_words_regex(text, ['inject', 'scribble', 'preparation', 'bloggy', 'reflections', 'wozz'])

        text = self.remove_urls(text)

        text = self.remove_spamassassin_terms(text)

        words = text.split()

        # term = 'textual'
        # if term in words:
        #     logging.info(f"A string contém o termo '{term}': {text}")

        final = []
        for word in words:
            if word not in stops:
                final.append(word)
        final = " ".join(final)
        final = final.translate(str.maketrans("", "", string.punctuation))
        final = "".join([i for i in final if not i.isdigit()])
        # while "  " in final:
        #     final = final.replace("  ", " ")
        final = re.sub(r'\s+', ' ', final).strip()
        return (final)

    def clean_docs(self, docs):
        stops = stopwords.words("english")
        final = []
        for doc in docs:
            if(not isinstance(doc, str)):
                doc = ''
            clean_doc = self.remove_stops(doc, stops)
            final.append(clean_doc)
        return (final)
    
    def booleanizerTaget(self, docs):
        final = []
        for doc in docs:
            if(doc=='spam' or doc=='1' or doc==1):
                final.append(1)
            else:
                final.append(0)
        return (final)

    def load_sample_data(self):
        """Load sample data"""
        logging.info("Loading sample data...")

        # Select two categories for binary classification
        categories = ['alt.atheism', 'soc.religion.christian']

        # Load train and test data
        newsgroup_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
        newsgroup_test  = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)

        # Merge train and test to after make a new division
        X = np.concatenate([newsgroup_train.data, newsgroup_test.data])
        y = np.concatenate([newsgroup_train.target, newsgroup_test.target])

        return X, y, newsgroup_train.target_names
    
    def load_real_data(self):
        documents = pd.read_csv(f"{dirname}/../input/{self.input_file_name}", encoding='utf-8')

        X = documents['text']
        y = documents['label']

        X = self.clean_docs(X)
        y = self.booleanizerTaget(y)
        
        return X, y, ['0', '1']

    def create_pipeline(self):
        """Create full pipeline with TF-IDF and Logistic Regression"""
        models = {
            'logistic-regression': LogisticRegression(
                                        random_state=42,
                                        C=20.0,
                                        class_weight='balanced',
                                        max_iter=1000,
                                        penalty='l2',
                                        solver='liblinear'
                                    ),
            'naive-bayes': MultinomialNB(
                                alpha=0.1,
                                fit_prior=True,
                                force_alpha=True
                            ),
            'decision-tree': DecisionTreeClassifier(random_state=42)
        }
        
        if self.algorithm_enum.value not in models:
            raise ValueError(f"Modelo '{self.algorithm_enum.value}' não suportado.")

        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                        max_features=20000,
                        min_df=0.001,
                        max_df=0.7,
                        ngram_range=(1, 1),
                        norm='l2',
                        stop_words='english',
                        sublinear_tf=True,
                    )
            ),
            ('classifier', models[self.algorithm_enum.value])
        ])
        
        return pipeline

    def create_basic_pipeline(self):
        """Create basic pipeline with TF-IDF and Logistic Regression"""
        models = {
            'logistic-regression': LogisticRegression(random_state=42),
            'naive-bayes': MultinomialNB(),
            'decision-tree': DecisionTreeClassifier(random_state=42)
        }
        
        if self.algorithm_enum.value not in models:
            raise ValueError(f"Modelo '{self.algorithm_enum.value}' não suportado.")
        
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('classifier', models[self.algorithm_enum.value])
        ])
        
        return pipeline
    
    def define_param_grid(self, _10percent):
        """Define a params grid optimized for binary classification"""

        # IF-IDF params
        tfidf_params = {
            'tfidf__max_features': [20000], # More features to capture patterns
            'tfidf__ngram_range':  [(1,1), (1,2)], # Important bigrams
            'tfidf__min_df':       [0.0005, 0.001, _10percent], # Smaller min_df for rare spam words
            'tfidf__max_df':       [0.7], # Remove very common words
            'tfidf__stop_words':   ['english'], # Remove or not Stopwords
            'tfidf__sublinear_tf': [True], # Always True to spam (replace tf with 1 + log(tf)) | Default: False
            'tfidf__norm':         ['l2'], # Normalization
        }
        
        # Parâmetros específicos por modelo
        model_params = {
            'logistic-regression': {
                'classifier__C':            [15.0, 20.0],
                'classifier__solver':       ['liblinear'],
                'classifier__penalty':      ['l2'],
                'classifier__class_weight': ['balanced'],
                'classifier__max_iter':     [1000],
            },
            'naive-bayes': {
                'classifier__alpha':        [0.1, 3.0],
                'classifier__force_alpha':  [True],
                'classifier__fit_prior':    [True],
            },
            # 'decision-tree': {
            #     'classifier__criterion':                ['gini', 'entropy'],
            #     'classifier__max_depth':                [8, 10, 15, 20, None],
            #     'classifier__min_samples_split':        [10, 20, 50],
            #     'classifier__min_samples_leaf':         [5, 10, 20],
            #     'classifier__max_features':             ['sqrt', 'log2', 0.3, 0.5, None],
            #     'classifier__min_impurity_decrease':    [0.0, 0.001, 0.01],
            #     'classifier__class_weight':             ['balanced', None, {0: 1, 1: 3}],
            #     'classifier__ccp_alpha':                [0.0, 0.01, 0.1],
            # },
        }

        # # IF-IDF params
        # tfidf_params = {
        #     'tfidf__max_features': [1000,3500,5000,10000], # More features to capture patterns
        #     'tfidf__ngram_range':  [(1,1), (1,2),], # Important bigrams
        #     'tfidf__min_df':       [1, 2], # Smaller min_df for rare spam words
        #     'tfidf__max_df':       [0.7, 0.8, 0.9], # Remove very common words
        #     'tfidf__stop_words':   ['english'], # Remove or not Stopwords
        #     'tfidf__sublinear_tf': [True], # Always True to spam (replace tf with 1 + log(tf)) | Default: False
        #     'tfidf__norm':         ['l2','l1'], # Normalization
        # }
        
        # # Parâmetros específicos por modelo
        # model_params = {
        #     'logistic-regression': {
        #         'classifier__C':            [0.1, 1.0, 5.0, 10.0, 15.0],
        #         'classifier__solver':       ['liblinear'],
        #         'classifier__penalty':      ['l1', 'l2'],
        #         'classifier__class_weight': [{0: 1, 1: 5}, 'balanced'],
        #         'classifier__max_iter':     [1000, 1500],
        #     },
        #     'naive-bayes': {
        #         'classifier__alpha':        [0.1, 3.0, 5.0, 10.0],
        #         'classifier__force_alpha':  [True, False],
        #         'classifier__fit_prior':    [True, False],
        #     },
        #     # 'decision-tree': {
        #     #     'classifier__criterion':                ['gini', 'entropy'],
        #     #     'classifier__max_depth':                [8, 10, 15, 20, None],
        #     #     'classifier__min_samples_split':        [10, 20, 50],
        #     #     'classifier__min_samples_leaf':         [5, 10, 20],
        #     #     'classifier__max_features':             ['sqrt', 'log2', 0.3, 0.5, None],
        #     #     'classifier__min_impurity_decrease':    [0.0, 0.001, 0.01],
        #     #     'classifier__class_weight':             ['balanced', None, {0: 1, 1: 3}],
        #     #     'classifier__ccp_alpha':                [0.0, 0.01, 0.1],
        #     # },
        # }
        
        if self.algorithm_enum.value not in model_params:
            raise ValueError(f"Modelo '{self.algorithm_enum.value}' não suportado")
        
        # Combina parâmetros do TF-IDF com parâmetros do modelo
        param_grid = {**tfidf_params, **model_params[self.algorithm_enum.value]}
        
        return param_grid
    
    def train_without_grid_search(self, X_train, y_train):
        """Training the binary classification model"""
        logging.info("Starting...")
        self.pipeline = self.create_pipeline()
        self.pipeline.fit(X_train, y_train)

    def train_with_grid_search(self, X_train, y_train, cv_folds=5):
        """Training the model using GridSearchCV optimized for binary classification"""
        logging.info("Starting Grid Search with Cross-Validation...")

        # Create pipeline
        pipeline = self.create_basic_pipeline()

        _10percent = int(len(X_train)*0.03)
        logging.info(F"_10percent: {_10percent}")

        # Define params grid
        param_grid = self.define_param_grid(_10percent)

        # Set up GridSearchCV with multiple metrics
        grid_search = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            cv=cv_folds,
            scoring='accuracy',
            n_jobs=10,
            verbose=1,
            return_train_score=True
        )

        # Execute the search
        grid_search.fit(X_train, y_train)

        # Store results
        self.pipeline = grid_search.best_estimator_
        self.best_params = grid_search.best_params_

        logging.info(f"Best score CV (precision): {grid_search.best_score_:.4f}")
        logging.info("Best params")
        for param, value in self.best_params.items():
            logging.info(f"   {param}: {value}")

        return grid_search
    
    def cross_validate(self, X, y, cv_folds=5):
        """Execute cross validation with multiple metrics for binary classification"""
        if self.pipeline is None:
            raise ValueError("A model must be trained first!")
        
        logging.info("\n")
        logging.info(f"Executing cross validation with {cv_folds} folds...")

        # Multiple metrics for binary classification
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

        train_sizes = np.linspace(0.1, 1.0, 5)

        results = {}

        for metric in metrics:
            results[metric] = {}
            # scores = cross_val_score(self.pipeline, X, y, cv=cv_folds, scoring=metric)

            train_sizes_abs, train_scores, val_scores = learning_curve(
                self.pipeline, X, y, cv=cv_folds, train_sizes=train_sizes, scoring=metric, n_jobs=10
            )

            for i, train_size in enumerate(train_sizes):
                results[metric][f"{train_size}"] = {}
                results[metric][f"{train_size}"]['train_scores']    = train_scores[i]
                results[metric][f"{train_size}"]['val_scores']      = val_scores[i]
                results[metric][f"{train_size}"]['train_sizes_abs'] = train_sizes_abs[i]
            
            logging.info(f"{metric.upper()}: {val_scores[len(train_sizes)-1].mean():.4f} (+/- {val_scores[len(train_sizes)-1].std() * 2:.4f})")

        self.cv_scores = results
        return results
    
    def _plot_evaluation_results(self, y_test, y_pred, y_pred_proba, target_names):
        """Create some specific views for binary classification"""
        fig, axes = plt.subplots(1, 2)

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names, ax=axes[0])

        axes[0].set_title('Confusion Matrix')
        axes[0].set_ylabel('True')
        axes[0].set_xlabel('Predicted')

        # ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc         = roc_auc_score(y_test, y_pred_proba)

        axes[1].plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {auc:.4f})')
        axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].set_title('ROC Curve')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.suptitle(f"{self.algorithm_enum.get_name()}", fontsize=16, fontweight='bold')
        
        plt.savefig(f"{self.output_base_path}/heatmap.png")
        # plt.show()

    def _plot_threshold_comparison(self, results):
        """Plots comparison between different thresholds"""
        thresholds = list(results.keys())
        metrics = ['precision', 'recall', 'f1', 'accuracy']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        for i, metric in enumerate(metrics):
            values = [results[t][metric] for t in thresholds]
            axes[i].bar(range(len(thresholds)), values, alpha=0.7)
            axes[i].set_xlabel('Threshold')
            axes[i].set_ylabel(metric.upper())
            axes[i].set_title(f'Comparison of {metric.upper()} by Threshold')
            axes[i].set_xticks(range(len(thresholds)))
            axes[i].set_xticklabels([f'{t:.1f}' for t in thresholds])
            axes[i].grid(True, alpha=0.3)
            
            # Adds values to the bars
            for j, v in enumerate(values):
                axes[i].text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.suptitle(f"{self.algorithm_enum.get_name()}", fontsize=16, fontweight='bold')

        plt.savefig(f"{self.output_base_path}/thresholds-comparison-per-metric.png")
        
        # Additional graph: FPR vs FNR
        plt.figure(figsize=(10, 6))
        fpr_values = [results[t]['fpr'] for t in thresholds]
        fnr_values = [results[t]['fnr'] for t in thresholds]
        
        x = np.arange(len(thresholds))
        width = 0.35
        
        plt.bar(x - width/2, fpr_values, width, label='False Positive Rate', alpha=0.7)
        plt.bar(x + width/2, fnr_values, width, label='False Negative Rate', alpha=0.7)
        
        plt.xlabel('Threshold')
        plt.ylabel('Error rate')
        plt.title('Comparison: False Positive vs False Negative rate')
        plt.xticks(x, [f'{t:.1f}' for t in thresholds])
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Adds values to the bars
        for i, (fpr, fnr) in enumerate(zip(fpr_values, fnr_values)):
            plt.text(i - width/2, fpr + 0.01, f'{fpr:.3f}', ha='center', va='bottom')
            plt.text(i + width/2, fnr + 0.01, f'{fnr:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.suptitle(f"{self.algorithm_enum.get_name()}", fontsize=16, fontweight='bold')

        plt.savefig(f"{self.output_base_path}/thresholds-comparison-fp-fn-rate.png")

    def evaluate_with_multiple_thresholds(self, y_test, y_pred_proba, thresholds=[0.3, 0.5, 0.7]):
        """Evaluate the model with multiple thresholds to compare"""
        if self.pipeline is None:
            raise("Model must be trained first!")
        
        logging.info("\n")
        logging.info("="*60)
        logging.info("Evaluation with multiple Thresholds")
        logging.info("="*60)

        results = {}

        # fig, axes = plt.subplots(1, len(thresholds), figsize=(5 * len(thresholds), 4))

        for i, threshold in enumerate(thresholds):
            logging.info("\n")
            logging.info(f"--- Threshold: {threshold:.1f} ---")

            # Predictions with custom threshold
            y_pred = (y_pred_proba >= threshold).astype(int)
            # logging.info("\n" + "="*60)
            # logging.info("y_pred_proba")
            # logging.info(y_pred)
            # logging.info("="*60)

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)

            fig, ax = plt.subplots(figsize=(5, 4))

            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'],
                   ax=ax
                )
            ax.set_title(f'{self.algorithm_enum.get_name()} - Threshold = {threshold}')
            ax.set_xlabel('Predito')
            ax.set_ylabel('Real')

            # Save individual heatmap
            plt.tight_layout()
            plt.savefig(f"{self.output_base_path}/heatmap-threshold-{threshold:.1f}.png")
            plt.close()
            
            # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            #             xticklabels=['Positive', 'Negative'],
            #             yticklabels=['Positive', 'Negative'],
            #             ax=axes[i])
            # axes[i].set_title(f'Threshold = {threshold}')
            # axes[i].set_xlabel('Predito')
            # axes[i].set_ylabel('Real')

            tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

            # False positive/negative rate
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

            # logging.info(f"Accuracy:  {accuracy:.4f}")
            # logging.info(f"Precision: {precision:.4f} (False Positives: {fp})")
            # logging.info(f"Recall:    {recall:.4f} (False Negatives: {fn})")
            # logging.info(f"F1-Score:  {f1:.4f}")
            # logging.info(f"FPR:       {fpr:.4f} (False Positives Rate)")
            # logging.info(f"FNR:       {fnr:.4f} (False Negatives Rate)")
            
            results[threshold] = {
                'accuracy': accuracy, 'precision': precision, 'recall': recall,
                'f1': f1, 'fpr': fpr, 'fnr': fnr, 'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp
            }

        # plt.tight_layout()
        # plt.suptitle(f"{self.algorithm_enum.get_name()}", fontsize=16, fontweight='bold')

        # plt.savefig(f"{self.output_base_path}/heatmap-per-threshold.png")

        # Comparative view
        self._plot_threshold_comparison(results)

        return results
    
    def evaluate_model(self, X_test, y_test, target_names):
        """Evaluates the model on the test set with specific metrics for binary classification"""
        if self.pipeline is None:
            raise ValueError("Model must be trained first!")
        
        logging.info("\n")
        logging.info("="*50)
        logging.info("Evaluation on test set")
        logging.info("="*50)

        # Predictions
        y_pred       = self.pipeline.predict(X_test)
        y_pred_proba = self.pipeline.predict_proba(X_test)[:, 1]

        # Basic metrics
        accuracy  = accuracy_score(X_test, y_pred)
        auc_score = roc_auc_score(y_test, y_pred_proba)

        logging.info(f"Accuracy: {accuracy:.4f}")
        logging.info(f"AUC-ROC: {auc_score:.4f}")
        
        # Detailed report
        logging.info("\n")
        logging.info("Detailed report")
        logging.info(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))

        # Views
        self._plot_evaluation_results(y_test, y_pred, y_pred_proba, target_names)

        return accuracy, auc_score, y_pred, y_pred_proba

    def plot_cv_scores(self, X_train, X_test, y_train, y_test, feature_names=None, class_names=['0', '1']):
        """Plot cross validation scored for multiple metrics"""
        if self.cv_scores is None:
            logging.info("Execute cross validation first!")
            return
            
        n_metrics = len(self.cv_scores)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols  # Ceiling division
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
        
        # Handle case where there's only one subplot
        if n_metrics == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if n_cols > 1 else [axes]
        else:
            axes = axes.flatten()
        
        for i, (metric, scores) in enumerate(self.cv_scores.items()):
            if(metric.lower() == 'accuracy'):
                logging.info(f"accuracy scores {scores['1.0']['val_scores']}")
                
            axes[i].boxplot(scores['1.0']['val_scores'])
            axes[i].set_title(f"{metric.upper()}\nAverage: {scores['1.0']['val_scores'].mean():.4f}")
            axes[i].set_ylabel('Score')
            axes[i].grid(True, alpha=0.3)

        # Remove extra subplot
        if len(self.cv_scores) < len(axes):
            fig.delaxes(axes[-1])

        plt.tight_layout()
        plt.suptitle(f"{self.algorithm_enum.get_name()}", fontsize=16, fontweight='bold')

        plt.savefig(f"{self.output_base_path}/boxplot-per-metric.png")        
        # plt.show()

        # Generating reports
        logging.info("Generating analyses...")
        self.plot.plot_roc_curve()
        self.plot.plot_precision_recall_curve()
        self.plot.plot_class_distribution()
        self.plot.plot_learning_curves(self.cv_scores)
        self.plot.generate_complete_report()
        self.plot.analyze_tfidf_features(32)

    def predict_with_confidence(self, texts):
        """Make predictions with confidence intervals"""
        if self.pipeline is None:
            raise ValueError("Model must be trained first!")
        
        predictions   = self.pipeline.predict(texts)
        probabilities = self.pipeline.predict_proba(texts)

        results = []
        for i, (text, pred, prob) in enumerate(zip(texts, predictions, probabilities)):
            confidence   = max(prob)
            prob_class_1 = prob[1]

            results.append({
                'text':         text[:100] + '...' if len(text) > 100 else text,
                'precition':    int(pred),
                'confidence':   confidence,
                'prob_class_1': prob_class_1
            })

        return results

def string_to_boolean(s):
    bool_map = {"true": True, "false": False}

    return bool_map.get(s.strip().lower(), False)

def main():
    """Main function to demonstrate full use"""
    # Read  arguments
    args = sys.argv
    args.pop(0)

    new_args = dict()
    for (i, value) in enumerate(args):
        if(i%2 != 0):
            continue
        
        new_args[value] = args[i+1]

    # Read algorithm setting (Default: 'logistic-regression')
    w_grid_search = new_args.get('-g')
    if(w_grid_search == None or (w_grid_search != None and w_grid_search not in ['True', 'False'])):
        w_grid_search = 'True'

    w_grid_search = string_to_boolean(w_grid_search)

    # Read algorithm setting (Default: 'logistic-regression')
    algorithm = new_args.get('-a')
    if(algorithm == None or (algorithm != None and algorithm not in ['logistic-regression', 'naive-bayes', 'decision-tree'])):
        algorithm = 'logistic-regression'

    # Read scoring setting (Default: 'accuracy')
    scoring = new_args.get('-s')
    if(scoring == None or (scoring != None and scoring not in ['accuracy', 'precision', 'reacll', 'f1', 'roc_auc'])):
        scoring = 'accuracy'

    input_file_name = new_args.get('-i')

    # Initiates the classifier
    classifier = SpamClassifier(AlgorithmEnum(algorithm), scoring, input_file_name)

    os.makedirs(classifier.output_base_path, exist_ok=True)
    # sys.stdout = open(f"{classifier.output_base_path}/stdout.log", 'w', encoding='utf-8')

    # Configurar o logging
    logging.basicConfig(
        filename=f"{classifier.output_base_path}/stdout.log",
        filemode='w',  # 'w' para sobrescrever, 'a' para adicionar
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO,
        encoding='utf-8'
    )

    if(w_grid_search):
        logging.info('With GridSearchCV')

    logging.info("="*60)
    logging.info(f"Binary classification with TF-IDF and {classifier.algorithm_enum.get_name()}")
    logging.info("="*60)

    # Data loading
    # X, y, target_names = classifier.load_sample_data()
    X, y, target_names = classifier.load_real_data()
    logging.info(f"Dataset loaded: {len(X)} samples")
    logging.info(f"Classes: {target_names}")
    
    logging.info(f"Distribuição das classes: {np.bincount(y)}")

    # Data split
    # - Split arrays or matrices into random train and test subsets.
    # - random_state: Controls the shuffling applied to the data before applying the split. 
    #    - Pass an int for reproducible output across multiple function calls.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    logging.info(f"Train: {len(X_train)} samples")
    logging.info(f"Test: {len(X_test)} samples")

    grid_search = None

    if(w_grid_search):
        # Train with Grid Search
        grid_search = classifier.train_with_grid_search(X_train, y_train, cv_folds=5)
    else:
        # Train without Grid Search
        classifier.train_without_grid_search(X_train, y_train)

    # Complete cross validation
    cv_results  = classifier.cross_validate(X_train, y_train, cv_folds=5)

    # Final evaluation (evaluate_model|evaluate_with_multiple_thresholds)
    threshold_results = None
    accuracy = None
    auc = None
    
    # Get probabilities
    y_pred_proba = classifier.pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    # accuracy, auc, y_pred, y_pred_proba = classifier.evaluate_model(
    #     X_test, y_test, target_names
    # )
    threshold_results = classifier.evaluate_with_multiple_thresholds(
        y_test, y_pred_proba, thresholds=[0.3, 0.5, 0.6, 0.7]
    )

    feature_names = classifier.pipeline[:-1].get_feature_names_out()

    classifier.plot = BinaryClassificationPlot(
                        classifier.pipeline,
                        classifier.algorithm_enum,
                        X_train,
                        X_test,
                        y_train,
                        y_test,
                        classifier.output_base_path,
                        feature_names,
                        ['0', '1'],
                        y_pred,
                        y_pred_proba
                    )

    # Views
    classifier.plot_cv_scores(X_train, X_test, y_train, y_test, feature_names, ['0', '1'])

    # # Salvar modelo completo
    joblib.dump(classifier.pipeline, f"{classifier.output_base_path}/model.pkl")
    
    logging.info("\n")
    logging.info("="*50)
    logging.info("Summary of Results")
    logging.info("="*50)

    if(w_grid_search):
        logging.info(f"Best CV Score (AUC): {grid_search.best_score_:.4f}")
    
    logging.info(f"AVG CV Accuracy: {cv_results['accuracy']['1.0']['val_scores'].mean():.4f}")
    logging.info(f"AVG CV F1: {cv_results['f1']['1.0']['val_scores'].mean():.4f}")
    logging.info(f"AVG CV AUC: {cv_results['roc_auc']['1.0']['val_scores'].mean():.4f}")

    if threshold_results is None:
        logging.info(f"Average: {accuracy:.4f}")
        logging.info(f"AUC: {auc:.4f}")
    else:
        logging.info("\n")
        logging.info("Threshold Results:")
        for result in threshold_results:
            logging.info("\n")
            logging.info(f"Threshold: {result}")
            logging.info(f"   - Average: {threshold_results[result]['accuracy']:.4f}")
            logging.info(f"   - Precision: {threshold_results[result]['precision']:.4f}")
            logging.info(f"   - Recall: {threshold_results[result]['recall']:.4f}")
            logging.info(f"   - F1: {threshold_results[result]['f1']:.4f}")
            logging.info(f"   - fpr: {threshold_results[result]['fpr']:.4f}")
            logging.info(f"   - fnr: {threshold_results[result]['fnr']:.4f}")
            logging.info(f"   - tn: {threshold_results[result]['tn']:.4f}")
            logging.info(f"   - fp: {threshold_results[result]['fp']:.4f}")
            logging.info(f"   - fn: {threshold_results[result]['fn']:.4f}")
            logging.info(f"   - tp: {threshold_results[result]['tp']:.4f}")

    return classifier, grid_search



def __old():
    np.set_printoptions(threshold=sys.maxsize)

    # Vamos supor que temos um arquivo 'textos.csv' com colunas 'texto' e 'categoria'
    # Leitura do arquivo CSV
    # documents = pd.read_csv('./input/spam_ham_dataset.csv', encoding='utf-8')
    documents = pd.read_csv('./input/spam.csv', encoding='utf-8')


    # Talvez fazer o encode de carcteres HTML(ex: &lt; (<)) !!!
    def remove_stops(text, stops):       
        words = text.split()
        final = []
        for word in words:
            if word not in stops:
                final.append(word)
        final = " ".join(final)
        final = final.translate(str.maketrans("", "", string.punctuation))
        final = "".join([i for i in final if not i.isdigit()])
        while "  " in final:
            final = final.replace("  ", " ")
        return (final)

    def clean_docs(docs):
        stops = stopwords.words("english")
        final = []
        for doc in docs:
            clean_doc = remove_stops(doc, stops)
            final.append(clean_doc)
        return (final)

    def booleanizerTaget(docs):
        final = []
        for doc in docs:
            if(doc=='spam' or doc=='1'):
                final.append(1)
            else:
                final.append(0)
        return (final)


    corpus = documents['text']
    target = documents['label']

    cleaned_corpus = clean_docs(corpus)
    target =booleanizerTaget(target)

    vectorizer = TfidfVectorizer(
        lowercase=True,
        max_df=0.7,
        max_features=5000,
        min_df=2,
        ngram_range=(1, 1),
        norm='l2',
        stop_words='english',
        sublinear_tf=True
    )

    vectors = vectorizer.fit_transform(cleaned_corpus)

    feature_names = vectorizer.get_feature_names_out()

    dense = vectors.todense()
    denselist = dense.tolist()

    all_keywords = []

    for description in denselist:
        x=0
        keywords = []
        for word in description:
            if word > 0:
                keywords.append(feature_names[x])
            x=x+1
        all_keywords.append(keywords)

    # logging.info(corpus[442])
    # logging.info(all_keywords[442])


    # Dividir em treino e teste
    corpus_train, corpus_test, target_train, target_test = train_test_split(
        vectors, target , test_size=0.2 #, random_state=42, stratify=target
    )

    # É recomendável definir fit_intercept=True e aumentar o intercept_scaling.
    # Com SVMs e regressão logística, o parâmetro C controla a esparsidade: 
    #   - quanto menor C, menos recursos selecionados. 
    #   - Com Lasso, quanto maior o parâmetro alfa, menos recursos selecionados.
    # https://scikit-learn.org/stable/_images/grid_search_workflow.png
    model = LogisticRegression(
        penalty="l2",
        C=10.0,
        class_weight={0: 1, 1: 5},
        max_iter=10000,
        solver='liblinear'
    )

    # model = MultinomialNB(
    #         alpha=1.0,  # Smoothing parameter (Laplace/Lidstone smoothing)
    #         fit_prior=True,  # Learn class prior probabilities
    #         class_prior=None  # Prior probabilities of the classes
    #     )

    model.fit(corpus_train, target_train)

    # eli5.explain_weights(model)

    # eli5.show_weights(model, vec=vectorizer)

    explaining_pred_html = eli5.formatters.html.format_as_html(eli5.explain_prediction(model, cleaned_corpus[34], vec=vectorizer))

    with open("explaining.html", "w", encoding="utf-8") as f:
        f.write(explaining_pred_html)

    explaining_weights_html = eli5.formatters.html.format_as_html(eli5.explain_weights(model, feature_names=feature_names))
    # explaining_weights_html = eli5.formatters.html.format_as_html(eli5.explain_weights(model, vec=vectorizer))

    with open("explaining_w.html", "w", encoding="utf-8") as f:
        f.write(explaining_weights_html)

    # explaining_preddf_html = eli5.formatters.html.format_as_html(eli5.explain_prediction_df(estimator=model, doc=cleaned_corpus))

    # with open("explaining_preddf.html", "w", encoding="utf-8") as f:
    #     f.write(explaining_preddf_html)


    prediction = model.predict(corpus_test)

    # Avaliar o modelo
    logging.info("Relatório de Classificação: (classification_report)")
    logging.info(classification_report(target_test, prediction))

    logging.info("Relatório de Classificação (confusion_matrix):")
    logging.info(confusion_matrix(target_test, prediction))

    # Validação cruzada
    cv_scores = cross_val_score(model, corpus_train, target_train, cv=5)
    logging.info("\n")
    logging.info("Scores da Validação Cruzada:")
    logging.info(f"Acurácia média: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

    # # Salvar o modelo treinado
    # import joblib
    # joblib.dump(model, './output/modelo_tfidf_logreg.joblib')



if __name__ == "__main__":
    main()
    # __old()