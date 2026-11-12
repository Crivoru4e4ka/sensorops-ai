"""Генератор синтетических аудиозвонков для демо Voice-to-CRM.

Использует системный TTS (pyttsx3) для озвучивания диалогов «менеджер — клиент»
на русском языке и сохраняет результат в data/audio_samples/ как .wav.
Если установлен пакет ``lameenc``, дополнительно конвертирует в .mp3.

Запуск:
    python scripts/generate_audio.py
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIO_DIR = PROJECT_ROOT / "data" / "audio_samples"

# Диалоги: (роль, реплика). Роль «m» — менеджер, «c» — клиент.
DIALOGUES: dict[str, list[tuple[str, str]]] = {
    "call_01_urgent_inductive": [
        ("m", "Компания ПромТех, менеджер Сергей, слушаю вас."),
        ("c", "Здравствуйте, это Дмитрий из ПромСтройМаш. Нам срочно нужны индуктивные датчики, резьба М18, выход PNP, питание двадцать четыре вольта."),
        ("m", "Понял вас, Дмитрий. Какое количество требуется?"),
        ("c", "Двести штук, желательно с запасом на складе. Поставка нужна на этой неделе."),
        ("m", "Хорошо. Посмотрю наличие и сегодня отправлю вам коммерческое предложение на почту."),
        ("c", "Отлично. Мой адрес: Дмитрий точка Петров, собака промстроймаш точка ру."),
        ("m", "Записал. До связи!"),
    ],
    "call_02_medium_temperature_pressure": [
        ("m", "Добрый день, компания ПромТех, Елена слушает."),
        ("c", "Здравствуйте, меня зовут Марина, компания АгроПищеПром. Подбираем датчики температуры и давления для пищевого производства. Защита IP шестьдесят семь минимум."),
        ("m", "Для каких сред будут использоваться датчики?"),
        ("c", "Температура до ста пятидесяти градусов, давление до десяти бар. Партия пятьдесят штук. Сроки не горят."),
        ("m", "Я подготовлю варианты и перезвоню вам в четверг."),
        ("c", "Договорились, спасибо."),
    ],
    "call_03_refusal_optical": [
        ("m", "Компания ПромТех, Олег слушает."),
        ("c", "Добрый день, Игорь из компании МойкаПро. Мы искали оптические датчики с защитой IP шестьдесят девять для влажной среды."),
        ("m", "У нас есть такие позиции. Сколько штук вам нужно?"),
        ("c", "Честно говоря, ваш конкурент предложил цену ниже. Мы пока отказываемся, не нужно."),
        ("m", "Жаль. Если что, обращайтесь — пришлю наши условия."),
        ("c", "Хорошо, до свидания."),
    ],
}


def render_dialogue_wav(dialogue: list[tuple[str, str]], out_path: Path) -> None:
    """Озвучивает диалог системным TTS в WAV (менеджер и клиент — разный тембр)."""
    import pyttsx3

    engine = pyttsx3.init()
    voices = engine.getProperty("voices")
    russian = [voice for voice in voices if "ru" in str(getattr(voice, "id", "")).lower()] or voices
    manager_voice = russian[0].id

    full_text = ""
    for role, line in dialogue:
        prefix = "Менеджер. " if role == "m" else "Клиент. "
        full_text += prefix + line + " "

    engine.setProperty("voice", manager_voice)
    engine.setProperty("rate", 150)
    engine.save_to_file(full_text, str(out_path))
    engine.runAndWait()


def to_mp3(wav_path: Path) -> Path | None:
    try:
        import lameenc
    except ImportError:
        return None
    import wave

    with wave.open(str(wav_path), "rb") as wav:
        params = wav.getparams()
        frames = wav.readframes(wav.getnframes())
    encoder = lameenc.Encoder()
    encoder.set_bit_rate(64)
    encoder.set_in_sample_rate(params.framerate)
    encoder.set_channels(params.nchannels)
    encoder.set_quality(2)
    mp3_path = wav_path.with_suffix(".mp3")
    mp3_path.write_bytes(encoder.encode(frames) + encoder.flush())
    return mp3_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Генерация демо-аудиозвонков")
    parser.add_argument("--out", type=str, default=str(AUDIO_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, dialogue in DIALOGUES.items():
        wav_path = out_dir / f"{name}.wav"
        render_dialogue_wav(dialogue, wav_path)
        transcript = "\n".join(f"{'Менеджер' if role == 'm' else 'Клиент'}: {line}" for role, line in dialogue)
        (out_dir / f"{name}.txt").write_text(transcript + "\n", encoding="utf-8")
        mp3_path = to_mp3(wav_path)
        print(f"Создан {wav_path.name}" + (f" и {mp3_path.name}" if mp3_path else " (mp3 пропущен — нет lameenc)"))


if __name__ == "__main__":
    main()
