import streamlit as st
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

dirname = os.path.dirname(__file__)

# Configuração da página
st.set_page_config(
    page_title="Detector de Spam/Ham",
    page_icon="📧",
    layout="centered"
)

MODELS_CONFIG = {
    "Naive Bayes DT1": {
        "path": "./output/spam/naive-bayes/scoring_accuracy/20250707-2342/model.pkl",
        "description": "Modelo Naive Bayes - Dataset 1"
    },
    "Naive Bayes DT2": {
        "path": "./output/spam_ham_dataset/naive-bayes/scoring_accuracy/20250715-2336/model.pkl", 
        "description": "Modelo Naive Bayes - Dataset 2"
    },
    "Naive Bayes DT3": {
        "path": "./output/spamassassin/naive-bayes/scoring_accuracy/20250707-2338/model.pkl",
        "description": "Modelo Naive Bayes - Dataset 3"
    },
    "Logistic Regression DT1": {
        "path": "./output/spam/logistic-regression/scoring_accuracy/20250707-2345/model.pkl",
        "description": "Regressão Logística - Dataset 1"
    },
    "Logistic Regression DT2": {
        "path": "./output/spam_ham_dataset/logistic-regression/scoring_accuracy/20250715-2328/model.pkl",
        "description": "Regressão Logística - Dataset 2"
    },
    "Logistic Regression DT3": {
        "path": "./output/spamassassin/logistic-regression/scoring_accuracy/20250715-2249/model.pkl",
        "description": "Regressão Logística - Dataset 3"
    }
}


# Seletor de modelo
selected_model = st.sidebar.selectbox(
    "Escolha o modelo:",
    options=list(MODELS_CONFIG.keys()),
    index=0,
    help="Selecione qual modelo usar para classificação"
)

# Título da aplicação
st.title("📧 Detector de Spam/Ham")
st.markdown(f"**Modelo ativo:** {selected_model}")
st.markdown("---")

# Sidebar com informações e controles
st.sidebar.header("🤖 Seleção de Modelo")

# Mostrar descrição do modelo selecionado
st.sidebar.info(MODELS_CONFIG[selected_model]["description"])

st.sidebar.markdown("---")
st.sidebar.header("ℹ️ Sobre a Aplicação")
st.sidebar.write("Esta aplicação utiliza machine learning para classificar mensagens como SPAM ou HAM (mensagem legítima).")
st.sidebar.write("**HAM**: Mensagem legítima")
st.sidebar.write("**SPAM**: Mensagem indesejada")

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Configurações")

# Controle do limiar de decisão
threshold = st.sidebar.slider(
    "🎯 Limiar de Decisão",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.01,
    help="Probabilidade mínima para classificar como SPAM. Valores menores = mais sensível a SPAM. Valores maiores = mais conservador."
)

# Explicação do limiar
st.sidebar.markdown("**Como funciona o limiar:**")
if threshold < 0.3:
    st.sidebar.warning("🔴 **Muito Sensível**: Pode classificar HAMs como SPAM")
elif threshold < 0.7:
    st.sidebar.info("🟡 **Balanceado**: Boa precisão geral")
else:
    st.sidebar.success("🟢 **Conservador**: Evita falsos positivos de SPAM")

st.sidebar.write(f"Atual: {threshold:.2f} ({threshold*100:.0f}%)")

# Função para carregar um modelo específico
@st.cache_resource
def load_model(model_name):
    """
    Carrega o modelo escolhido pelo usuário.
    """
    try:
        model_path = MODELS_CONFIG[model_name]["path"]

        # Opção 2: Modelo salvo with joblib
        model = joblib.load(model_path)

        return model, None
    except FileNotFoundError:
         print('aaa')

# Função principal de predição
def predict_spam(text, model, threshold=0.5):
    """
    Função para fazer a predição se o texto é spam ou ham.
    O modelo já faz todo o preprocessamento e vetorização internamente.
    
    Args:
        text: Texto a ser classificado
        model: Modelo treinado
        threshold: Limiar de decisão (padrão: 0.5)
    """
    try:
        # Apenas remove espaços extras
        text = text.strip()
        
        # O modelo recebe o texto bruto e faz todo o processamento
        probability = model.predict_proba([text])[0]
        
        # Aplicar o limiar customizado
        # probability[1] é a probabilidade de ser SPAM
        prediction = 1 if probability[1] >= threshold else 0
        
        return prediction, probability
        
    except Exception as e:
        st.error(f"Erro na predição: {e}")
        return None, None

# Carregar modelo selecionado
try:
    result = load_model(selected_model)
    if result is None:
        model, warning_msg = None, "Erro ao carregar modelo"
    else:
        model, warning_msg = result
except Exception as e:
    model, warning_msg = None, f"Erro ao carregar modelo: {str(e)}"

# Mostrar aviso se modelo não foi encontrado
if warning_msg:
    st.warning(warning_msg)

# Interface principal
if model is not None:
    st.subheader("✍️ Digite sua mensagem:")
    
    # Campo de texto para entrada
    user_input = st.text_area(
        "Insira o texto da mensagem que deseja classificar:",
        height=120,
        placeholder="Digite aqui a mensagem que você quer verificar se é spam ou ham..."
    )
    
    # Botão para fazer a predição
    if st.button("🔍 Analisar Mensagem", type="primary"):
        if user_input.strip():
            with st.spinner("Analisando a mensagem..."):
                prediction, probability = predict_spam(user_input, model, threshold)
                
                if prediction is not None:
                    # Mostrar resultado
                    st.markdown("---")
                    st.subheader("📊 Resultado da Análise:")
                    
                    # Criar duas colunas para mostrar o resultado
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if prediction == 1:  # SPAM
                            st.error("🚨 **SPAM**")
                            st.write("Esta mensagem foi classificada como spam (mensagem indesejada).")
                        else:  # HAM
                            st.success("✅ **HAM**")
                            st.write("Esta mensagem foi classificada como ham (mensagem legítima).")
                    
                    with col2:
                        # Mostrar probabilidades
                        if len(probability) == 2:
                            prob_ham = probability[0] * 100
                            prob_spam = probability[1] * 100
                            
                            st.metric("Probabilidade HAM", f"{prob_ham:.1f}%")
                            st.metric("Probabilidade SPAM", f"{prob_spam:.1f}%")
                    
                    # Informação sobre o limiar usado
                    st.info(f"🎯 **Limiar usado**: {threshold:.2f} ({threshold*100:.0f}%) - Probabilidade SPAM: {probability[1]*100:.1f}%")
                    
                    # Barra de progresso visual para probabilidade de SPAM
                    st.markdown("**Probabilidade de ser SPAM:**")
                    spam_prob = probability[1]
                    
                    # Criar uma barra colorida baseada na probabilidade
                    if spam_prob >= threshold:
                        st.progress(spam_prob, text=f"SPAM: {spam_prob*100:.1f}%")
                        st.markdown(f"<div style='background-color: #ff4444; padding: 5px; border-radius: 5px; text-align: center; color: white;'><b>Acima do limiar ({threshold*100:.0f}%) → SPAM</b></div>", unsafe_allow_html=True)
                    else:
                        st.progress(spam_prob, text=f"SPAM: {spam_prob*100:.1f}%")
                        st.markdown(f"<div style='background-color: #44ff44; padding: 5px; border-radius: 5px; text-align: center; color: white;'><b>Abaixo do limiar ({threshold*100:.0f}%) → HAM</b></div>", unsafe_allow_html=True)
        else:
            st.warning("⚠️ Por favor, digite uma mensagem para analisar.")
    
    # Exemplos de uso com comparação
    st.markdown("---")
    st.subheader("💡 Teste Rápido dos Modelos:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Exemplo de HAM:**")
        example_ham = "Oi, tudo bem? Vamos nos encontrar amanhã no cinema às 19h?"
        if st.button("Testar HAM", key="ham_example"):
            prediction, probability = predict_spam(example_ham, model, threshold)
            if prediction is not None:
                result = "SPAM" if prediction == 1 else "HAM"
                prob = probability[1] * 100
                st.write(f"**{selected_model}**: {result} (Prob. SPAM: {prob:.1f}%)")
    
    with col2:
        st.markdown("**Exemplo de SPAM:**")
        example_spam = "PARABÉNS! Você ganhou um prêmio de R$ 10.000! Clique aqui urgentemente para resgatar sua oferta grátis!"
        if st.button("Testar SPAM", key="spam_example"):
            prediction, probability = predict_spam(example_spam, model, threshold)
            if prediction is not None:
                result = "SPAM" if prediction == 1 else "HAM"
                prob = probability[1] * 100
                st.write(f"**{selected_model}**: {result} (Prob. SPAM: {prob:.1f}%)")
    
    # Comparação entre todos os modelos
    st.markdown("---")
    st.subheader("🔍 Comparar Todos os Modelos")
    
    test_text = st.text_input("Teste uma mensagem em todos os modelos:", placeholder="Digite uma mensagem para comparar...")
    
    if st.button("🚀 Comparar Modelos", type="secondary") and test_text.strip():
        comparison_results = []
        
        with st.spinner("Testando em todos os modelos..."):
            for model_name in MODELS_CONFIG.keys():
                try:
                    temp_result = load_model(model_name)
                    if temp_result is None:
                        temp_model, temp_warning = None, "Erro"
                    else:
                        temp_model, temp_warning = temp_result
                        
                    if temp_model:
                        pred, prob = predict_spam(test_text, temp_model, threshold)
                        if pred is not None and prob is not None:
                            result = "SPAM" if pred == 1 else "HAM"
                            comparison_results.append({
                                "Modelo": model_name,
                                "Classificação": result,
                                "Prob. SPAM": f"{prob[1]*100:.1f}%",
                                "Prob. HAM": f"{prob[0]*100:.1f}%"
                            })
                        else:
                            comparison_results.append({
                                "Modelo": model_name,
                                "Classificação": "Erro",
                                "Prob. SPAM": "-",
                                "Prob. HAM": "-"
                            })
                    else:
                        comparison_results.append({
                            "Modelo": model_name,
                            "Classificação": "Erro",
                            "Prob. SPAM": "-",
                            "Prob. HAM": "-"
                        })
                except Exception as ex:
                    comparison_results.append({
                        "Modelo": model_name,
                        "Classificação": f"Erro: {str(ex)}",
                        "Prob. SPAM": "-",
                        "Prob. HAM": "-"
                    })
        
        # Mostrar comparação em tabela
        if comparison_results:
            df_comparison = pd.DataFrame(comparison_results)
            st.dataframe(df_comparison, use_container_width=True)
            
            # Estatísticas da comparação
            spam_count = sum(1 for result in comparison_results if result["Classificação"] == "SPAM")
            ham_count = sum(1 for result in comparison_results if result["Classificação"] == "HAM")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Classificaram como SPAM", f"{spam_count}/{len(comparison_results)}")
            with col2:
                st.metric("Classificaram como HAM", f"{ham_count}/{len(comparison_results)}")
            with col3:
                consensus = "Alta" if max(spam_count, ham_count) >= len(comparison_results) * 0.8 else "Baixa"
                st.metric("Consenso", consensus)

else:
    st.error("❌ Não foi possível carregar o modelo selecionado.")
    st.info("📝 **Instruções para configurar seus modelos:**")