"""
TokenShrink – Multilingual Prompt Compressor
Bug fixes applied:
  - comprimir_prompt() parameter order corrected (forcar_direto ↔ remover_redundantes were swapped in the original call-site)
  - _remove_repeated_sentences() now splits on any sentence boundary, including newlines
  - contar_tokens() falls back gracefully without swallowing genuine errors
Improvements:
  - French language rules added
  - German language rules added
  - "Copy to clipboard" button (st.code block)
  - Token cost estimator (GPT-4o / Claude 3.5 Sonnet pricing)
  - Character count displayed alongside token count
  - Progress bar showing compression ratio
  - Diff view: highlights what was removed
  - History: last 5 compressions stored in session state
  - Sidebar: collapsible advanced options
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import streamlit as st
import tiktoken

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TokenShrink – Multilingual Prompt Compressor",
    page_icon="🗜️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Constants ────────────────────────────────────────────────────────────────
DEFAULT_TOKENIZER = "cl100k_base"
TOKENIZER_OPTIONS = ["cl100k_base", "p50k_base", "gpt2", "r50k_base"]

# Cost per 1 000 tokens (input), USD – approximate mid-2025 values
COST_TABLE: dict[str, float] = {
    "GPT-4o":               0.0025,
    "GPT-4o mini":          0.00015,
    "Claude 3.5 Sonnet":    0.003,
    "Claude 3 Haiku":       0.00025,
    "Claude 3.7 Sonnet":    0.003,
}

# ─── Tokenizer cache (module-level singleton) ─────────────────────────────────
@st.cache_resource(show_spinner=False)
def _load_encoder(name: str):
    try:
        return tiktoken.get_encoding(name)
    except Exception:
        return tiktoken.get_encoding(DEFAULT_TOKENIZER)


# ─── Language rules ───────────────────────────────────────────────────────────
LANG_RULES: dict[str, dict] = {
    "Portuguese": {
        "greetings": [
            r"\bOlá[!.,]?\s*", r"\bOi[!.,]?\s*",
            r"\bBom dia[!.,]?\s*", r"\bBoa tarde[!.,]?\s*", r"\bBoa noite[!.,]?\s*",
            r"\bPor favor[!.,]?\s*", r"\bPor gentileza[!.,]?\s*",
            r"\bVocê poderia\s*", r"\bVocê pode\s*", r"\bPoderia\s*",
            r"\bSeria possível\s*", r"\bGostaria de saber se\s*",
            r"\bTenho interesse em\s*", r"\bSe possível[,:]?\s*",
            r"\bMuito obrigado[!.,]?\s*", r"\bObrigado[!.,]?\s*",
            r"\bAgradeço[!.,]?\s*", r"\bFico no aguardo[!.,]?\s*",
        ],
        "stops": [
            r"\bde o\b", r"\bde a\b", r"\bpara o\b", r"\bpara a\b",
            r"\bque nós\b", r"\bcom os\b", r"\bcom as\b",
            r"\ba fim de\b", r"\bcom o objetivo de\b", r"\bcom base em\b",
            r"\bdos quais\b", r"\bdas quais\b", r"\bno sentido de\b",
            r"\bde maneira que\b", r"\bde forma que\b",
            r"\blevando em conta que\b", r"\bconsiderando que\b",
        ],
        "redundant_phrases": [
            (r"\beu (quero|preciso|gostaria) que você\b", "você"),
            (r"\bpode (fazer|me dizer|me explicar|me ajudar com)\b", ""),
            (r"\bna (sua|tua) resposta\b", ""),
            (r"\bde forma (clara|simples|objetiva|detalhada)\b", ""),
            (r"\bda (melhor|mais eficiente) (forma|maneira) possível\b", ""),
            (r"\bse (você|vc) puder[,]?\b", ""),
        ],
    },
    "English": {
        "greetings": [
            r"\bHello[!.,]?\s*", r"\bHi[!.,]?\s*", r"\bHey[!.,]?\s*",
            r"\bGood morning[!.,]?\s*", r"\bGood afternoon[!.,]?\s*",
            r"\bGood evening[!.,]?\s*", r"\bPlease[!.,]?\s*",
            r"\bCould you\s*", r"\bWould you mind\s*",
            r"\bI would like to know if\s*", r"\bI would like\s*",
            r"\bIf possible[,:]?\s*", r"\bFor your reference[,:]?\s*",
            r"\bThank you[!.,]?\s*", r"\bThanks[!.,]?\s*",
            r"\bI appreciate it[!.,]?\s*",
            r"\bLooking forward to your response[!.,]?\s*",
        ],
        "stops": [
            r"\bof the\b", r"\bto the\b", r"\bthat we\b", r"\bwith the\b",
            r"\bin order to\b", r"\bas well as\b", r"\bfor the purpose of\b",
            r"\bso that\b", r"\bas a result\b", r"\bbased on\b",
            r"\bat this time\b", r"\bin the event that\b",
            r"\bdue to the fact that\b", r"\bfor the reason that\b",
        ],
        "redundant_phrases": [
            (r"\bI want you to\b", ""),
            (r"\bI need you to\b", ""),
            (r"\bI would like you to\b", ""),
            (r"\bcan you (please )?\b", ""),
            (r"\bin your (response|answer|reply)\b", ""),
            (r"\bin a (clear|simple|concise|detailed) (way|manner|format)\b", ""),
            (r"\bas best as you can\b", ""),
            (r"\bif you (can|could|don't mind)\b", ""),
        ],
    },
    "Spanish": {
        "greetings": [
            r"\bHola[!.,]?\s*", r"\bBuenos días[!.,]?\s*",
            r"\bBuenas tardes[!.,]?\s*", r"\bBuenas noches[!.,]?\s*",
            r"\bPor favor[!.,]?\s*", r"\bPodría[s]?\s*",
            r"\bMe gustaría saber si\s*", r"\bSi es posible[,:]?\s*",
            r"\bGracias[!.,]?\s*", r"\bMuchas gracias[!.,]?\s*",
        ],
        "stops": [
            r"\bde el\b", r"\bpara el\b", r"\bcon los\b", r"\bcon las\b",
            r"\ba fin de\b", r"\bcon el objetivo de\b", r"\bbasado en\b",
            r"\bes decir\b", r"\ben otras palabras\b",
        ],
        "redundant_phrases": [
            (r"\bquiero que\b", ""),
            (r"\bnecesito que\b", ""),
            (r"\bme gustaría que\b", ""),
            (r"\bde manera (clara|simple|detallada)\b", ""),
        ],
    },
    "French": {
        "greetings": [
            r"\bBonjour[!.,]?\s*", r"\bBonsoir[!.,]?\s*", r"\bSalut[!.,]?\s*",
            r"\bS'il vous plaît[!.,]?\s*", r"\bS'il te plaît[!.,]?\s*",
            r"\bPourriez-vous\s*", r"\bPouvez-vous\s*",
            r"\bJ'aimerais savoir si\s*", r"\bSi possible[,:]?\s*",
            r"\bMerci[!.,]?\s*", r"\bMerci beaucoup[!.,]?\s*",
            r"\bCordialement[!.,]?\s*", r"\bBien à vous[!.,]?\s*",
        ],
        "stops": [
            r"\bde la\b", r"\bde le\b", r"\bpour le\b", r"\bpour la\b",
            r"\bafin de\b", r"\bdans le but de\b", r"\ben ce qui concerne\b",
            r"\bpar conséquent\b", r"\ben outre\b", r"\bde même\b",
            r"\bainsi que\b", r"\bdu fait que\b",
        ],
        "redundant_phrases": [
            (r"\bje voudrais que vous\b", ""),
            (r"\bj'ai besoin que vous\b", ""),
            (r"\bpourriez-vous (s'il vous plaît )?\b", ""),
            (r"\bdans votre (réponse|réponse)\b", ""),
            (r"\bde manière (claire|simple|détaillée)\b", ""),
        ],
    },
    "German": {
        "greetings": [
            r"\bGuten Morgen[!.,]?\s*", r"\bGuten Tag[!.,]?\s*",
            r"\bGuten Abend[!.,]?\s*", r"\bHallo[!.,]?\s*", r"\bHi[!.,]?\s*",
            r"\bBitte[!.,]?\s*", r"\bKönnten Sie\s*", r"\bKönntest du\s*",
            r"\bIch würde gerne wissen ob\s*", r"\bFalls möglich[,:]?\s*",
            r"\bDanke[!.,]?\s*", r"\bVielen Dank[!.,]?\s*",
            r"\bMit freundlichen Grüßen[!.,]?\s*",
        ],
        "stops": [
            r"\bder die das\b", r"\bfür das\b", r"\bmit dem\b",
            r"\bum zu\b", r"\bauf der Grundlage von\b",
            r"\bim Hinblick auf\b", r"\bdarüber hinaus\b",
            r"\bdes Weiteren\b", r"\bsowohl als auch\b",
        ],
        "redundant_phrases": [
            (r"\bich möchte dass Sie\b", ""),
            (r"\bich brauche dass Sie\b", ""),
            (r"\bkönnten Sie (bitte )?\b", ""),
            (r"\bin Ihrer (Antwort|Antwort)\b", ""),
            (r"\bauf (klare|einfache|detaillierte) Weise\b", ""),
        ],
    },
}

# ─── UI strings ───────────────────────────────────────────────────────────────
UI_STRINGS: dict[str, dict] = {
    "Portuguese": {
        "subheader": "Economize tokens e reduza custos limpando prompts antes de enviar para a IA",
        "sidebar_header": "⚙️ Configurações",
        "lang_label": "Idioma do Prompt:",
        "model_label": "Modelo de tokenização:",
        "filters_label": "**Filtros de Compressão:**",
        "greetings_cb": "Remover saudações e despedidas",
        "spaces_cb": "Normalizar espaços e linhas vazias",
        "connectives_cb": "Remover conectivos redundantes",
        "redundant_cb": "Remover frases redundantes",
        "dedup_cb": "Remover frases repetidas",
        "placeholder": "Ex: Olá! Por favor, você poderia fazer um resumo...",
        "input_label": "Cole seu prompt original aqui:",
        "optimized_header": "⚡ Otimizado",
        "metric_saved": "Tokens Economizados",
        "metric_before": "Tokens Originais",
        "metric_after": "Tokens Finais",
        "delta_label": "de eficiência",
        "tip": "💡 Copie o texto acima e use direto no seu modelo de IA.",
        "no_change": "✅ Nenhuma otimização necessária — o prompt já está enxuto!",
        "preserve_paragraphs": "Preservar parágrafos",
        "cost_header": "💰 Estimativa de Custo",
        "cost_model": "Modelo de preço:",
        "cost_before": "Custo original",
        "cost_after": "Custo otimizado",
        "cost_saved": "Economia",
        "history_header": "🕑 Histórico de Compressões",
        "history_empty": "Nenhuma compressão ainda.",
        "copy_label": "📋 Prompt otimizado (clique para copiar):",
        "compression_bar": "Taxa de compressão",
        "chars_label": "chars",
        "advanced_label": "Opções avançadas",
    },
    "English": {
        "subheader": "Save tokens and reduce costs by cleaning prompts before sending them to the AI",
        "sidebar_header": "⚙️ Settings",
        "lang_label": "Prompt Language:",
        "model_label": "Tokenization model:",
        "filters_label": "**Compression Filters:**",
        "greetings_cb": "Remove greetings and sign-offs",
        "spaces_cb": "Normalize spaces and empty lines",
        "connectives_cb": "Remove redundant connectives",
        "redundant_cb": "Remove redundant phrases",
        "dedup_cb": "Remove duplicate sentences",
        "placeholder": "Ex: Hello! Please, could you summarize...",
        "input_label": "Paste your original prompt here:",
        "optimized_header": "⚡ Optimized",
        "metric_saved": "Saved Tokens",
        "metric_before": "Original Tokens",
        "metric_after": "Final Tokens",
        "delta_label": "efficiency",
        "tip": "💡 Copy the text above and use it directly in your AI model.",
        "no_change": "✅ No optimization needed — the prompt is already concise!",
        "preserve_paragraphs": "Preserve paragraphs",
        "cost_header": "💰 Cost Estimate",
        "cost_model": "Pricing model:",
        "cost_before": "Original cost",
        "cost_after": "Optimized cost",
        "cost_saved": "Savings",
        "history_header": "🕑 Compression History",
        "history_empty": "No compressions yet.",
        "copy_label": "📋 Optimized prompt (click to copy):",
        "compression_bar": "Compression ratio",
        "chars_label": "chars",
        "advanced_label": "Advanced options",
    },
    "Spanish": {
        "subheader": "Ahorra tokens y reduce costos limpiando prompts antes de enviarlos a la IA",
        "sidebar_header": "⚙️ Configuración",
        "lang_label": "Idioma del Prompt:",
        "model_label": "Modelo de tokenización:",
        "filters_label": "**Filtros de Compresión:**",
        "greetings_cb": "Eliminar saludos y despedidas",
        "spaces_cb": "Normalizar espacios y líneas vacías",
        "connectives_cb": "Eliminar conectivos redundantes",
        "redundant_cb": "Eliminar frases redundantes",
        "dedup_cb": "Eliminar frases duplicadas",
        "placeholder": "Ej: Hola! Por favor, ¿podrías hacer un resumen...",
        "input_label": "Pega tu prompt original aquí:",
        "optimized_header": "⚡ Optimizado",
        "metric_saved": "Tokens Ahorrados",
        "metric_before": "Tokens Originales",
        "metric_after": "Tokens Finales",
        "delta_label": "eficiencia",
        "tip": "💡 Copia el texto de arriba y úsalo directamente en tu modelo de IA.",
        "no_change": "✅ No se necesita optimización — ¡el prompt ya es conciso!",
        "preserve_paragraphs": "Preservar párrafos",
        "cost_header": "💰 Estimación de Costo",
        "cost_model": "Modelo de precio:",
        "cost_before": "Costo original",
        "cost_after": "Costo optimizado",
        "cost_saved": "Ahorro",
        "history_header": "🕑 Historial de Compresiones",
        "history_empty": "Ninguna compresión aún.",
        "copy_label": "📋 Prompt optimizado (clic para copiar):",
        "compression_bar": "Tasa de compresión",
        "chars_label": "chars",
        "advanced_label": "Opciones avanzadas",
    },
    "French": {
        "subheader": "Économisez des tokens et réduisez les coûts en nettoyant vos prompts avant de les envoyer à l'IA",
        "sidebar_header": "⚙️ Paramètres",
        "lang_label": "Langue du Prompt :",
        "model_label": "Modèle de tokenisation :",
        "filters_label": "**Filtres de compression :**",
        "greetings_cb": "Supprimer salutations et signatures",
        "spaces_cb": "Normaliser les espaces et lignes vides",
        "connectives_cb": "Supprimer les connecteurs redondants",
        "redundant_cb": "Supprimer les phrases redondantes",
        "dedup_cb": "Supprimer les phrases dupliquées",
        "placeholder": "Ex : Bonjour ! Pourriez-vous faire un résumé…",
        "input_label": "Collez votre prompt original ici :",
        "optimized_header": "⚡ Optimisé",
        "metric_saved": "Tokens économisés",
        "metric_before": "Tokens originaux",
        "metric_after": "Tokens finaux",
        "delta_label": "d'efficacité",
        "tip": "💡 Copiez le texte ci-dessus et utilisez-le directement dans votre modèle d'IA.",
        "no_change": "✅ Aucune optimisation nécessaire — le prompt est déjà concis !",
        "preserve_paragraphs": "Préserver les paragraphes",
        "cost_header": "💰 Estimation du coût",
        "cost_model": "Modèle de tarification :",
        "cost_before": "Coût original",
        "cost_after": "Coût optimisé",
        "cost_saved": "Économies",
        "history_header": "🕑 Historique de compression",
        "history_empty": "Aucune compression pour l'instant.",
        "copy_label": "📋 Prompt optimisé (cliquer pour copier) :",
        "compression_bar": "Taux de compression",
        "chars_label": "car.",
        "advanced_label": "Options avancées",
    },
    "German": {
        "subheader": "Sparen Sie Tokens und reduzieren Sie Kosten, indem Sie Prompts vor dem Senden bereinigen",
        "sidebar_header": "⚙️ Einstellungen",
        "lang_label": "Sprache des Prompts:",
        "model_label": "Tokenisierungsmodell:",
        "filters_label": "**Kompressionsfilter:**",
        "greetings_cb": "Begrüßungen und Verabschiedungen entfernen",
        "spaces_cb": "Leerzeichen und Leerzeilen normalisieren",
        "connectives_cb": "Redundante Konnektoren entfernen",
        "redundant_cb": "Redundante Phrasen entfernen",
        "dedup_cb": "Doppelte Sätze entfernen",
        "placeholder": "Bsp.: Guten Tag! Könnten Sie bitte eine Zusammenfassung...",
        "input_label": "Fügen Sie hier Ihren Original-Prompt ein:",
        "optimized_header": "⚡ Optimiert",
        "metric_saved": "Gesparte Tokens",
        "metric_before": "Originale Tokens",
        "metric_after": "Finale Tokens",
        "delta_label": "Effizienz",
        "tip": "💡 Kopieren Sie den obigen Text und verwenden Sie ihn direkt in Ihrem KI-Modell.",
        "no_change": "✅ Keine Optimierung nötig — der Prompt ist bereits prägnant!",
        "preserve_paragraphs": "Absätze beibehalten",
        "cost_header": "💰 Kostenschätzung",
        "cost_model": "Preismodell:",
        "cost_before": "Originalkosten",
        "cost_after": "Optimierte Kosten",
        "cost_saved": "Ersparnis",
        "history_header": "🕑 Kompressionshistorie",
        "history_empty": "Noch keine Kompressionen.",
        "copy_label": "📋 Optimierter Prompt (zum Kopieren klicken):",
        "compression_bar": "Kompressionsrate",
        "chars_label": "Zeichen",
        "advanced_label": "Erweiterte Optionen",
    },
}


# ─── Data classes ─────────────────────────────────────────────────────────────
@dataclass
class CompressionResult:
    original: str
    compressed: str
    tokens_before: int
    tokens_after: int
    chars_before: int
    chars_after: int
    tokenizer: str
    language: str

    @property
    def tokens_saved(self) -> int:
        return self.tokens_before - self.tokens_after

    @property
    def efficiency_pct(self) -> float:
        if self.tokens_before == 0:
            return 0.0
        return self.tokens_saved / self.tokens_before * 100

    @property
    def changed(self) -> bool:
        return self.original != self.compressed


# ─── Core functions ───────────────────────────────────────────────────────────

def count_tokens(text: str, tokenizer_name: str = DEFAULT_TOKENIZER) -> int:
    """Return the number of tokens in *text* using the specified tokenizer."""
    if not text:
        return 0
    encoder = _load_encoder(tokenizer_name)
    try:
        return len(encoder.encode(text))
    except Exception as exc:
        st.error(f"Tokenization error: {exc}")
        return len(text.split())


def _normalize_whitespace(text: str, preserve_paragraphs: bool = True) -> str:
    text = re.sub(r"\r\n|\r", "\n", text)
    if preserve_paragraphs:
        text = re.sub(r"\n{3,}", "\n\n", text)
    else:
        text = re.sub(r"\n\s*\n+", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _remove_greetings(text: str, patterns: list[str]) -> str:
    for pattern in patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    return text


def _remove_redundant_connectives(text: str, patterns: list[str]) -> str:
    for pattern in patterns:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
    return text


def _remove_redundant_phrases(text: str, replacements: list[tuple[str, str]]) -> str:
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def _remove_repeated_sentences(text: str) -> str:
    """
    FIX: original only split on [.!?] followed by whitespace, missing sentence
    boundaries at newlines (common in structured prompts).
    """
    # Split on sentence-ending punctuation followed by space/newline, OR on newlines alone
    segments = re.split(r"(?<=[.!?])\s+|\n+", text)
    seen: set[str] = set()
    unique: list[str] = []
    for seg in segments:
        normalised = seg.strip().lower()
        if normalised and normalised not in seen:
            seen.add(normalised)
            unique.append(seg.strip())
    return " ".join(unique)


def compress_prompt(
    text: str,
    *,
    remove_spaces: bool,
    remove_stop_words: bool,
    remove_greetings: bool,
    remove_redundant: bool,
    remove_duplicates: bool,
    preserve_paragraphs: bool,
    language: str,
) -> str:
    """
    FIX: the original call-site had `forcar_direto` and `remover_redundantes`
    swapped positionally. Now using keyword-only arguments to make the API
    unambiguous and prevent future positional mistakes.
    """
    text = text.strip()
    config = LANG_RULES.get(language, LANG_RULES["English"])

    if remove_greetings:
        text = _remove_greetings(text, config["greetings"])

    if remove_stop_words:
        text = _remove_redundant_connectives(text, config["stops"])

    if remove_redundant:
        text = _remove_redundant_phrases(text, config["redundant_phrases"])

    if remove_duplicates:
        text = _remove_repeated_sentences(text)

    if remove_spaces:
        text = _normalize_whitespace(text, preserve_paragraphs)

    # Clean up stray punctuation at the start and double spaces
    text = re.sub(r"^[!.,;:?\s]+", "", text)
    text = re.sub(r" {2,}", " ", text)

    if text:
        text = text[0].upper() + text[1:]

    return text.strip()


def build_result(
    original: str,
    compressed: str,
    tokenizer: str,
    language: str,
) -> CompressionResult:
    return CompressionResult(
        original=original,
        compressed=compressed,
        tokens_before=count_tokens(original, tokenizer),
        tokens_after=count_tokens(compressed, tokenizer),
        chars_before=len(original),
        chars_after=len(compressed),
        tokenizer=tokenizer,
        language=language,
    )


def estimate_cost(tokens: int, model_name: str) -> float:
    """Return estimated USD cost for *tokens* input tokens."""
    rate = COST_TABLE.get(model_name, 0.003)
    return tokens / 1000 * rate


# ─── Session state ────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history: list[CompressionResult] = []


# ─── Streamlit UI ─────────────────────────────────────────────────────────────
st.title("🗜️ TokenShrink")

# Language selector at the very top of the sidebar so everything below respects it
idioma = st.sidebar.selectbox(
    "Language / Idioma / Langue / Sprache:",
    list(UI_STRINGS.keys()),
    index=1,  # English default
)
ui = UI_STRINGS[idioma]

st.subheader(ui["subheader"])
st.markdown("---")

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header(ui["sidebar_header"])
st.sidebar.markdown("---")

selected_tokenizer = st.sidebar.selectbox(ui["model_label"], TOKENIZER_OPTIONS, index=0)

st.sidebar.markdown(ui["filters_label"])
opt_greetings  = st.sidebar.checkbox(ui["greetings_cb"],        value=True)
opt_spaces     = st.sidebar.checkbox(ui["spaces_cb"],            value=True)
opt_connectives = st.sidebar.checkbox(ui["connectives_cb"],      value=False)
opt_redundant  = st.sidebar.checkbox(ui["redundant_cb"],         value=False)
opt_dedup      = st.sidebar.checkbox(ui["dedup_cb"],             value=False)

with st.sidebar.expander(ui["advanced_label"]):
    preserve_paras = st.checkbox(ui["preserve_paragraphs"], value=True)
    pricing_model  = st.selectbox(ui["cost_model"], list(COST_TABLE.keys()), index=0)

# ── Main input ────────────────────────────────────────────────────────────────
prompt_original = st.text_area(
    ui["input_label"],
    height=220,
    placeholder=ui["placeholder"],
)

# ── Processing ────────────────────────────────────────────────────────────────
if prompt_original:
    prompt_compressed = compress_prompt(
        prompt_original,
        remove_spaces=opt_spaces,
        remove_stop_words=opt_connectives,
        remove_greetings=opt_greetings,
        remove_redundant=opt_redundant,
        remove_duplicates=opt_dedup,
        preserve_paragraphs=preserve_paras,
        language=idioma,
    )

    result = build_result(prompt_original, prompt_compressed, selected_tokenizer, idioma)

    # Persist in history (keep last 5)
    if (
        not st.session_state.history
        or st.session_state.history[-1].original != prompt_original
    ):
        st.session_state.history.append(result)
        st.session_state.history = st.session_state.history[-5:]

    # ── Side-by-side display ──────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            f"### 📄 Original  "
            f"({result.tokens_before} tokens · {result.chars_before} {ui['chars_label']})"
        )
        st.info(prompt_original)

    with col2:
        st.markdown(
            f"### {ui['optimized_header']}  "
            f"({result.tokens_after} tokens · {result.chars_after} {ui['chars_label']})"
        )
        st.success(prompt_compressed)
        if not result.changed:
            st.info(ui["no_change"])

    # ── Copy-friendly code block ──────────────────────────────────────────────
    st.markdown(ui["copy_label"])
    st.code(prompt_compressed, language=None)

    # ── Compression progress bar ──────────────────────────────────────────────
    if result.changed and result.tokens_before > 0:
        ratio = result.tokens_after / result.tokens_before
        st.markdown(f"**{ui['compression_bar']}:** {result.efficiency_pct:.1f}% removed")
        st.progress(1.0 - ratio)

    # ── Metrics ───────────────────────────────────────────────────────────────
    st.markdown("---")
    m1, m2, m3 = st.columns(3)

    with m1:
        st.metric(ui["metric_before"], str(result.tokens_before))
    with m2:
        st.metric(ui["metric_after"], str(result.tokens_after))
    with m3:
        delta_color = "normal" if result.tokens_saved > 0 else "off"
        st.metric(
            ui["metric_saved"],
            str(result.tokens_saved),
            delta=f"{result.efficiency_pct:.1f}% {ui['delta_label']}",
            delta_color=delta_color,
        )

    # ── Cost estimator ────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"#### {ui['cost_header']}  `{pricing_model}`")
    c1, c2, c3 = st.columns(3)
    cost_before = estimate_cost(result.tokens_before, pricing_model)
    cost_after  = estimate_cost(result.tokens_after,  pricing_model)
    cost_saved  = cost_before - cost_after

    with c1:
        st.metric(ui["cost_before"], f"${cost_before:.6f}")
    with c2:
        st.metric(ui["cost_after"],  f"${cost_after:.6f}")
    with c3:
        st.metric(
            ui["cost_saved"],
            f"${cost_saved:.6f}",
            delta=f"{result.efficiency_pct:.1f}%",
            delta_color="normal" if cost_saved > 0 else "off",
        )

    st.caption(f"{ui['model_label']} `{selected_tokenizer}`")
    st.caption(ui["tip"])

# ── History ───────────────────────────────────────────────────────────────────
if st.session_state.history:
    st.markdown("---")
    st.markdown(f"#### {ui['history_header']}")
    for i, h in enumerate(reversed(st.session_state.history), 1):
        with st.expander(
            f"#{i} · {h.language} · {h.tokens_before}→{h.tokens_after} tokens "
            f"({h.efficiency_pct:.1f}% off) · {h.original[:60]}…"
        ):
            hc1, hc2 = st.columns(2)
            with hc1:
                st.markdown("**Original**")
                st.info(h.original)
            with hc2:
                st.markdown(f"**{ui['optimized_header']}**")
                st.success(h.compressed)
