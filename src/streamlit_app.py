import streamlit as st
import tiktoken
import re

# Streamlit page configuration
st.set_page_config(
    page_title="TokenShrink - Multilingual Prompt Compressor", 
    page_icon="🗜️", 
    layout="wide"
)

# Initialize universal token counter (cl100k_base used by GPT-4o, Claude 3, and others)
def contar_tokens(texto):
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(texto))
    except:
        # Simple fallback method if the tokenization library fails
        return len(texto.split())

# Compression function supporting multiple languages
def comprimir_prompt(texto, remover_vazios, remover_stop_words, forçar_direto, idioma):
    novo_texto = texto
    
    # Dictionary for cleaning rules by language
    regras = {
        "Portuguese": {
            "saudações": [
                r"olá[!.,]?\s*", r"por favor[!.,]?\s*", r"você poderia\s*", 
                r"gostaria de saber se\s*", r"bom dia[!.,]?\s*", r"boa tarde[!.,]?\s*",
                r"por gentileza[!.,]?\s*"
            ],
            "stops": [" de o ", " de a ", " para o ", " para a ", " que nós ", " com os ", " com as "]
        },
        "English": {
            "saudações": [
                r"hello[!.,]?\s*", r"hi[!.,]?\s*", r"please[!.,]?\s*", r"could you\s*", 
                r"would you mind\s*", r"good morning[!.,]?\s*", r"kindly[!.,]?\s*",
                r"i would like to know if\s*"
            ],
            "stops": [" of the ", " to the ", " that we ", " with the ", " in order to ", " as well as "]
        }
    }
    
    # Fetch language-specific rules based on user selection
    config = regras[idioma]
            
    # 1. Remove greetings and courtesies if enabled
    if forçar_direto:
        for saudacao in config["saudações"]:
            novo_texto = re.sub(saudacao, "", novo_texto, flags=re.IGNORECASE)
            
    # 2. Remove excessive whitespaces and empty lines (Universal rule)
    if remover_vazios:
        novo_texto = re.sub(r'\n\s*\n', '\n', novo_texto)
        novo_texto = re.sub(r' +', ' ', novo_texto)
        
    # 3. Remove language-specific redundant stop words
    if remover_stop_words:
        for word in config["stops"]:
            novo_texto = novo_texto.replace(word, " ")
            
    # --- EDGE CLEANUP & FORMATTING ---
    
    # Strip any leading punctuation or stray symbols left at the very beginning
    novo_texto = re.sub(r'^[!.,;:?\s]+', '', novo_texto)
    
    # Capitalize the first letter of the prompt if the string is not empty
    if novo_texto:
        novo_texto = novo_texto[0].upper() + novo_texto[1:]
            
    return novo_texto.strip()

# --- LOCALIZATION DICTIONARY (DYNAMIC UI TEXTS) ---
ui_strings = {
    "Portuguese": {
        "subheader": "Economize tokens e reduza seus custos limpando prompts antes de enviar para a IA",
        "sidebar_header": "⚙️ Configurações de Otimização",
        "lang_label": "Idioma do Prompt:",
        "filters_label": "**Filtros de Compartilhamento / Compressão:**",
        "greetings_cb": "Cortar cortesias e saudações",
        "spaces_cb": "Remover espaços e linhas vazias",
        "connectives_cb": "Remover conectivos redundantes",
        "placeholder": "Ex: Olá! Por favor, você poderia fazer um resumo...",
        "input_label": "Cole seu prompt original em Português aqui:",
        "optimized_header": "⚡ Otimizado",
        "metric_label": "Tokens Economizados",
        "delta_label": "de eficiência",
        "tip": "💡 Dica: Copie o texto da caixa verde (Otimizado) e use direto no seu modelo de IA."
    },
    "English": {
        "subheader": "Save tokens and reduce costs by cleaning prompts before sending them to the AI",
        "sidebar_header": "⚙️ Optimization Settings",
        "lang_label": "Prompt Language:",
        "filters_label": "**Compression Filters:**",
        "greetings_cb": "Trim courtesies and greetings",
        "spaces_cb": "Remove spaces and empty lines",
        "connectives_cb": "Remove redundant connectives",
        "placeholder": "Ex: Hello! Please, could you summarize...",
        "input_label": "Paste your original prompt in English here:",
        "optimized_header": "⚡ Optimized",
        "metric_label": "Saved Tokens",
        "delta_label": "efficiency",
        "tip": "💡 Tip: Copy the text from the green box (Optimized) and use it directly in your AI model."
    }
}

# --- STREAMLIT GRAPHICAL INTERFACE ---

st.title("🗜️ TokenShrink")

# SIDEBAR CONFIGURATIONS
# Language selector sets the interface language and the ruleset simultaneously
idioma_selecionado = st.sidebar.selectbox(
    "Language / Idioma:", 
    ["Portuguese", "English"],
    index=1 # Defaults to English layout
)

# Fetch current UI text dictionary based on language selection
current_ui = ui_strings[idioma_selecionado]

# Render dynamic headers and sidebar options
st.subheader(current_ui["subheader"])
st.markdown("---")

st.sidebar.header(current_ui["sidebar_header"])
st.sidebar.markdown("---")
st.sidebar.markdown(current_ui["filters_label"])

limpar_saudacoes = st.sidebar.checkbox(current_ui["greetings_cb"], value=True)
limpar_espacos = st.sidebar.checkbox(current_ui["spaces_cb"], value=True)
limpar_conectivos = st.sidebar.checkbox(current_ui["connectives_cb"], value=False)

# MAIN INPUT AREA
prompt_original = st.text_area(
    current_ui["input_label"], 
    height=220, 
    placeholder=current_ui["placeholder"]
)

# PROCESSING AND RESULTS DISPLAY
if prompt_original:
    # Execute the compression function
    prompt_comprimido = comprimir_prompt(
        prompt_original, 
        limpar_espacos, 
        limpar_conectivos, 
        limpar_saudacoes, 
        idioma_selecionado
    )
    
    # Measure tokens
    tokens_antes = contar_tokens(prompt_original)
    tokens_depois = contar_tokens(prompt_comprimido)
    
    # Calculate efficiency percentage
    if tokens_antes > 0:
        economia_pct = ((tokens_antes - tokens_depois) / tokens_antes) * 100
    else:
        economia_pct = 0

    # Split layout into two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"### 📄 Original ({tokens_antes} tokens)")
        st.info(prompt_original)
        
    with col2:
        st.markdown(f"### {current_ui['optimized_header']} ({tokens_depois} tokens)")
        st.success(prompt_comprimido)
        
    # Economy metrics panel
    st.markdown("---")
    
    st.metric(
        label=current_ui["metric_label"], 
        value=f"{tokens_antes - tokens_depois} tokens", 
        delta=f"{economia_pct:.1f}% {current_ui['delta_label']}"
    )
    
    st.caption(current_ui["tip"])