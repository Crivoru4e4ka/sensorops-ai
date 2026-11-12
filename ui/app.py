"""SensorOps AI — Streamlit Dashboard (демо-интерфейс).

Мультистраничное приложение с тремя разделами:
1. Smart Search   — гибридный семантический поиск по каталогу датчиков;
2. Voice-to-CRM   — разбор аудиозвонка (или текста) в карточку CRM + скоринг;
3. Lead Scoring   — B2B-профилирование и скоринг лида.

UI общается с FastAPI по адресу ``API_URL`` (по умолчанию http://localhost:8000).

Запуск:
    streamlit run ui/app.py
"""
from __future__ import annotations

import json
import os

import httpx
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="SensorOps AI",
    page_icon="📡",
    layout="wide",
)

PRIORITY_STYLES = {
    "High Priority": ("🔴 High Priority", "#ff4b4b"),
    "Medium": ("🟡 Medium", "#f6c343"),
    "Low": ("🟢 Low", "#21c354"),
}


def api() -> httpx.Client:
    return httpx.Client(base_url=API_URL, timeout=300)


def health_badge() -> None:
    """Показывает статус подключения к API и бэкендов."""
    try:
        with api() as client:
            response = client.get("/api/health")
        response.raise_for_status()
        data = response.json()
        backends = data.get("backends", {})
        st.sidebar.success(f"API: {API_URL} — ok")
        st.sidebar.caption(
            f"embedding: {backends.get('embedding')} · vector: {backends.get('vector_db')} · "
            f"llm: {backends.get('llm')} · asr: {backends.get('asr')} · "
            f"catalog: {data.get('catalog_size')} позиций"
        )
    except Exception as exc:  # noqa: BLE001
        st.sidebar.error(f"API недоступен ({API_URL}): {exc}")
        st.sidebar.caption("Запустите бэкенд: uvicorn src.main:app --port 8000")


def score_badge(probability: float, priority: str) -> None:
    label, color = PRIORITY_STYLES[priority]
    st.markdown(
        f"<span style='background:{color};color:#111;padding:4px 14px;border-radius:12px;"
        f"font-weight:600'>{label} · {probability:.0%}</span>",
        unsafe_allow_html=True,
    )


# ─── Страница 1: Smart Search ─────────────────────────────────────────────────

def page_search() -> None:
    st.title("📡 Smart Search — гибридный поиск датчиков")
    st.caption(
        "Dense Retrieval (векторы) + Sparse/Exact Filter (жёсткие атрибуты) + RAG-подсказка менеджеру"
    )

    with st.form("search_form"):
        query = st.text_input(
            "Запрос клиента (обычным языком)",
            value="Нужен оптический датчик приближения с защитой от влаги, резьба М18, выход PNP, питание 24 вольта",
        )
        col1, col2, col3 = st.columns(3)
        with col1:
            category = st.selectbox("Категория (жёсткий фильтр)", [None, "индуктивный", "оптический", "давления", "температуры"])
            voltage = st.selectbox("Напряжение", [None, "24В", "10-30В", "12В", "220В", "12-24В"])
        with col2:
            output_type = st.selectbox("Тип выхода", [None, "PNP", "NPN", "PNP/NPN", "аналоговый 4-20мА", "RS-485 Modbus"])
            ip_rating = st.selectbox("Класс защиты", [None, "IP65", "IP67", "IP68", "IP69K"])
        with col3:
            thread = st.selectbox("Резьба", [None, "М8", "М12", "М18", "М30", "G1/2", "G1/4", "М20×1.5"])
            max_price = st.number_input("Макс. цена, ₽", min_value=0.0, value=0.0, step=500.0)
        top_k = st.slider("Количество результатов", 3, 10, 5)
        submitted = st.form_submit_button("🔍 Найти", type="primary")

    if not submitted:
        return

    payload = {
        "query": query,
        "top_k": top_k,
        "filters": {
            "category": category,
            "voltage": voltage,
            "output_type": output_type,
            "ip_rating": ip_rating,
            "thread": thread,
            "max_price": max_price or None,
            "min_stock": None,
        },
    }
    with st.spinner("Гибридный поиск + RAG-генерация..."):
        with api() as client:
            response = client.post("/api/search", json=payload)
    if response.status_code != 200:
        st.error(f"Ошибка API: {response.status_code} {response.text}")
        return
    result = response.json()

    st.subheader("Результаты поиска")
    if not result["hits"]:
        st.warning("По заданным критериям ничего не найдено. Смягчите фильтры.")
    else:
        rows = []
        for hit in result["hits"]:
            specs = "; ".join(f"{key}: {value}" for key, value in hit["specifications"].items())
            rows.append(
                {
                    "Артикул": hit["id"],
                    "Название": hit["name"],
                    "Категория": hit["category"],
                    "Рейтинг": f"{hit['score']:.3f}",
                    "Остаток": hit["in_stock"],
                    "Цена, ₽": f"{hit['price']:,.0f}",
                    "Параметры": specs,
                }
            )
        st.dataframe(rows, use_container_width=True, hide_index=True)

    st.subheader("💡 Сравнительная подсказка для менеджера")
    source = result.get("generated_by", "mock")
    st.caption("Источник генерации: " + ("LLM (LiteLLM)" if source == "llm" else "mock-генератор (детерминированный)"))
    st.markdown(result["comparison"])


# ─── Страница 2: Voice-to-CRM ─────────────────────────────────────────────────

def page_voice_to_crm() -> None:
    st.title("🎙️ Voice-to-CRM — разбор звонка в карточку CRM")
    st.caption("faster-whisper транскрибация → Instructor + Pydantic v2 экстракция → скоринг лида")

    uploaded = st.file_uploader(
        "Загрузите аудиозапись звонка (.mp3, .wav)",
        type=["mp3", "wav", "m4a", "ogg", "flac"],
    )
    st.divider()
    use_text = st.checkbox("Или вставьте текст разговора")
    text = ""
    if use_text:
        text = st.text_area(
            "Текст разговора",
            height=160,
            placeholder="Менеджер: Компания ПромТех, слушаю вас.\nКлиент: Здравствуйте, это Дмитрий из ПромСтройМаш...",
        )

    if st.button("🚀 Разобрать звонок", type="primary", disabled=not (uploaded or text.strip())):
        with st.spinner("Транскрибация + экстракция сущностей + скоринг..."):
            with api() as client:
                if uploaded is not None:
                    response = client.post(
                        "/api/calls/analyze",
                        files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                    )
                else:
                    response = client.post("/api/calls/analyze-text", json={"text": text})
        if response.status_code != 200:
            st.error(f"Ошибка API: {response.status_code} {response.text}")
            return
        result = response.json()

        col_left, col_right = st.columns([3, 2])
        with col_left:
            st.subheader("📝 Транскрипт")
            with st.expander("Показать транскрипт", expanded=False):
                st.write(result["transcript"])

            st.subheader("🗂️ Карточка лида (CRMLeadCard)")
            card = result["lead_card"]
            st.json(card, expanded=True)
            st.caption(f"Экстракция: {'LLM (Instructor)' if result['extraction_source'] == 'llm' else 'rule-based (fallback)'}")

        with col_right:
            st.subheader("🎯 Скоринг лида")
            score = result["score"]
            score_badge(score["probability"], score["priority"])
            st.progress(score["probability"], text=f"Вероятность конверсии: {score['probability']:.1%}")
            st.markdown(f"**Рекомендация:** {score['recommendation']}")
            st.caption(f"ASR: {result.get('asr_model') or 'текстовый ввод'}")


# ─── Страница 3: Lead Scoring ─────────────────────────────────────────────────

def page_lead_scoring() -> None:
    st.title("🏭 B2B Profiling & Lead Scoring")
    st.caption("Data Enrichment Mock → ML-скоринг (CatBoost/Scikit-learn) → бейдж приоритета")

    with st.form("lead_form"):
        company = st.text_input("Название компании", value="ПромСтройМаш")
        client_name = st.text_input("Контактное лицо", value="Дмитрий Петров")
        contact = st.text_input("Телефон / email", value="d.petrov@promstroymash.ru")
        products = st.text_area(
            "Запрошенные продукты (каждый с новой строки)",
            value="датчик индуктивный\nдатчик индуктивный Autonics PR18-08DN",
        )
        col1, col2 = st.columns(2)
        with col1:
            urgency = st.selectbox("Срочность", ["критическая", "средняя", "низкая"])
            deal_stage = st.selectbox("Статус сделки", ["запрос КП", "повторный звонок", "отказ"])
        with col2:
            summary = st.text_area(
                "Суть разговора",
                value="Клиенту срочно нужны индуктивные датчики М18 PNP 24В, 200 штук на этой неделе.",
            )
        submitted = st.form_submit_button("🎯 Оценить лид", type="primary")

    if not submitted:
        return

    card = {
        "company_name": company or None,
        "client_name": client_name or None,
        "contact_info": contact or None,
        "requested_products": [item.strip() for item in products.splitlines() if item.strip()],
        "urgency": urgency,
        "deal_stage": deal_stage,
        "summary": summary,
        "action_items": [],
    }
    with st.spinner("Обогащение профиля + ML-скоринг..."):
        with api() as client:
            response = client.post("/api/calls/score", json=card)
    if response.status_code != 200:
        st.error(f"Ошибка API: {response.status_code} {response.text}")
        return
    result = response.json()

    features = result["features"]
    score = result["score"]
    st.subheader("Профиль компании (Data Enrichment Mock)")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Индустрия", features["industry"])
    col2.metric("Масштаб", features["business_scale"])
    col3.metric("Наличие на складе", f"{features['stock_availability']:.0%}")
    col4.metric("Объём партии", f"{features['order_volume']} шт.")

    st.subheader("Оценка сделки")
    score_badge(score["probability"], score["priority"])
    st.progress(score["probability"], text=f"Вероятность конверсии: {score['probability']:.1%}")
    st.markdown(f"**Рекомендация менеджеру:** {score['recommendation']}")


# ─── Навигация ────────────────────────────────────────────────────────────────

def main() -> None:
    st.sidebar.title("📡 SensorOps AI")
    st.sidebar.caption("Enterprise PoC: поиск · звонки · скоринг")
    page = st.sidebar.radio(
        "Раздел",
        ["🔍 Smart Search", "🎙️ Voice-to-CRM", "🏭 Lead Scoring"],
    )
    st.sidebar.divider()
    health_badge()
    st.sidebar.divider()
    st.sidebar.caption("Стек: FastAPI · LiteLLM · Instructor · Qdrant/ChromaDB · faster-whisper · CatBoost/Scikit-learn")

    if page == "🔍 Smart Search":
        page_search()
    elif page == "🎙️ Voice-to-CRM":
        page_voice_to_crm()
    else:
        page_lead_scoring()


if __name__ == "__main__":
    main()
