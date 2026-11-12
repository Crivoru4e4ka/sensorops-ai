"""E2E Voice-to-CRM: реальная транскрибация 3 аудиозвонков + экстракция + скоринг."""
import sys
import time

import httpx

BASE = "http://127.0.0.1:8000"
client = httpx.Client(base_url=BASE, timeout=600)

for _ in range(40):
    try:
        if client.get("/api/health").status_code == 200:
            break
    except Exception:
        time.sleep(1)
else:
    sys.exit("Сервер не поднялся")

health = client.get("/api/health").json()
print("health:", health["backends"])

files = [
    "data/audio_samples/call_01_urgent_inductive.wav",
    "data/audio_samples/call_02_medium_temperature_pressure.wav",
    "data/audio_samples/call_03_refusal_optical.wav",
]
for path in files:
    print("=" * 72)
    print("Файл:", path)
    with open(path, "rb") as fh:
        r = client.post("/api/calls/analyze", files={"file": (path.split("/")[-1], fh, "audio/wav")})
    if r.status_code != 200:
        print("   ОШИБКА:", r.status_code, r.json().get("detail"))
        continue
    data = r.json()
    print("   ASR:", data["asr_model"], "| экстракция:", data["extraction_source"])
    print("   Транскрипт:", data["transcript"][:220].replace("\n", " "))
    card = data["lead_card"]
    print("   Компания:", card["company_name"], "| Клиент:", card["client_name"], "| Контакт:", card["contact_info"])
    print("   Продукты:", card["requested_products"])
    print("   Срочность:", card["urgency"], "| Статус:", card["deal_stage"])
    print("   Action items:", card["action_items"])
    s = data["score"]
    print(f"   Скоринг: p={s['probability']} | {s['priority']}")
print("=" * 72)
print("E2E AUDIO PIPELINE OK")
