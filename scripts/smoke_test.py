"""Полный smoke-тест работающего API SensorOps AI (эндпоинт за эндпоинтом)."""
import sys
import time

import httpx

BASE = "http://127.0.0.1:8000"
client = httpx.Client(base_url=BASE, timeout=120)

# дождаться старта сервера
for _ in range(30):
    try:
        if client.get("/api/health").status_code == 200:
            break
    except Exception:
        time.sleep(1)
else:
    sys.exit("Сервер не поднялся")

print("=" * 72)
print("1. /api/health")
health = client.get("/api/health").json()
print("   status:", health["status"], "| catalog:", health["catalog_size"], "| backends:", health["backends"])

print("=" * 72)
print("2. /api/search — пример из ТЗ (оптический, М18, PNP, 24В, IP67)")
r = client.post(
    "/api/search",
    json={
        "query": "Нужен оптический датчик приближения с защитой от влаги, резьба М18, выход PNP, питание 24 вольта",
        "filters": {"category": "оптический", "voltage": "24В", "output_type": "PNP", "thread": "М18", "ip_rating": "IP67"},
        "top_k": 3,
    },
).json()
print("   hits:", len(r["hits"]), "| generated_by:", r["generated_by"])
for h in r["hits"]:
    print(f"   - {h['name']} | {h['category']} | score={h['score']} | stock={h['in_stock']} | price={h['price']} ₽")
print("   --- RAG-сравнение ---")
print("   " + r["comparison"].replace("\n", "\n   ")[:600])

print("=" * 72)
print("3. /api/search — семантический поиск БЕЗ фильтров («датчик для пищевого производства с паром»)")
r = client.post("/api/search", json={"query": "нужен датчик для пищевого производства, работает с паром и высокой температурой", "top_k": 3}).json()
for h in r["hits"]:
    print(f"   - {h['name']} | {h['category']} | score={h['score']}")

print("=" * 72)
print("4. /api/calls/analyze-text — разбор текста звонка")
text = (
    "Менеджер: Компания ПромТех, менеджер Сергей, слушаю вас. "
    "Клиент: Здравствуйте, это Дмитрий из ПромСтройМаш. Нам срочно нужны индуктивные датчики, "
    "резьба М18, выход PNP, питание двадцать четыре вольта, двести штук на этой неделе. "
    "Менеджер: Хорошо, сегодня отправлю вам коммерческое предложение на почту. "
    "Клиент: Отлично, мой адрес: Дмитрий точка Петров, собака промстроймаш точка ру."
)
r = client.post("/api/calls/analyze-text", json={"text": text}).json()
card = r["lead_card"]
print("   компания:", card["company_name"], "| клиент:", card["client_name"])
print("   контакт:", card["contact_info"])
print("   продукты:", card["requested_products"])
print("   срочность:", card["urgency"], "| статус:", card["deal_stage"])
print("   action_items:", card["action_items"])
print("   скоринг: p=", r["score"]["probability"], "|", r["score"]["priority"])
print("   рекомендация:", r["score"]["recommendation"][:120], "...")

print("=" * 72)
print("5. /api/calls/analyze — аудиофайл (faster-whisper пока не установлен локально)")
with open("data/audio_samples/call_01_urgent_inductive.wav", "rb") as f:
    r = client.post("/api/calls/analyze", files={"file": ("call_01.wav", f, "audio/wav")})
print("   status:", r.status_code, "| ответ:", r.json().get("detail", "")[:120])

print("=" * 72)
print("6. /api/calls/score — скоринг карточки лида")
r = client.post(
    "/api/calls/score",
    json={
        "company_name": "ООО НефтеГазСервис",
        "client_name": "Алексей",
        "contact_info": "+7 900 123-45-67",
        "requested_products": ["датчик давления", "датчик температуры"],
        "urgency": "критическая",
        "deal_stage": "запрос КП",
        "summary": "Нужны датчики давления 50 штук и температуры 100 штук, срочно.",
        "action_items": ["Выслать КП"],
    },
).json()
print("   probability:", r["score"]["probability"], "| priority:", r["score"]["priority"])
print("   features:", {k: v for k, v in r["features"].items()})
print("   рекомендация:", r["score"]["recommendation"][:120], "...")

print("=" * 72)
print("7. /api/calls/enrich — обогащение профиля компании")
for company in ["ООО НефтеГазСервис", "АгроПищеПром", "РитейлГрупп"]:
    r = client.post("/api/calls/enrich", data={"company_name": company}).json()
    print(f"   {company} -> {r}")

print("=" * 72)
print("SMOKE TEST OK")
