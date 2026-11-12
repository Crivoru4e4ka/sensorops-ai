# Генерация демо-аудиозвонков для SensorOps AI через системный TTS (ru-RU).
# Менеджер и клиент озвучиваются с разной скоростью, чтобы диалог читался.
Add-Type -AssemblyName System.Speech

$outDir = "C:\Users\mira\Documents\Default Project\sensorops-ai\data\audio_samples"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

$calls = @(
    @{
        Name = "call_01_urgent_inductive"
        Lines = @(
            @("m", "Компания ПромТех, менеджер Сергей, слушаю вас."),
            @("c", "Здравствуйте, это Дмитрий из ПромСтройМаш. Нам срочно нужны индуктивные датчики, резьба М18, выход PNP, питание двадцать четыре вольта."),
            @("m", "Понял вас, Дмитрий. Какое количество требуется?"),
            @("c", "Двести штук, желательно с запасом на складе. Поставка нужна на этой неделе."),
            @("m", "Хорошо. Посмотрю наличие и сегодня отправлю вам коммерческое предложение на почту."),
            @("c", "Отлично. Мой адрес: Дмитрий точка Петров, собака промстроймаш точка ру."),
            @("m", "Записал. До связи.")
        )
    },
    @{
        Name = "call_02_medium_temperature_pressure"
        Lines = @(
            @("m", "Добрый день, компания ПромТех, Елена слушает."),
            @("c", "Здравствуйте, меня зовут Марина, компания АгроПищеПром. Подбираем датчики температуры и давления для пищевого производства. Защита IP шестьдесят семь минимум."),
            @("m", "Для каких сред будут использоваться датчики?"),
            @("c", "Температура до ста пятидесяти градусов, давление до десяти бар. Партия пятьдесят штук. Сроки не горят."),
            @("m", "Я подготовлю варианты и перезвоню вам в четверг."),
            @("c", "Договорились, спасибо.")
        )
    },
    @{
        Name = "call_03_refusal_optical"
        Lines = @(
            @("m", "Компания ПромТех, Олег слушает."),
            @("c", "Добрый день, Игорь из компании МойкаПро. Мы искали оптические датчики с защитой IP шестьдесят девять для влажной среды."),
            @("m", "У нас есть такие позиции. Сколько штук вам нужно?"),
            @("c", "Честно говоря, ваш конкурент предложил цену ниже. Мы пока отказываемся, не нужно."),
            @("m", "Жаль. Если что, обращайтесь - пришлю наши условия."),
            @("c", "Хорошо, до свидания.")
        )
    }
)

foreach ($call in $calls) {
    $synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
    $voice = $synth.GetInstalledVoices() | Where-Object { $_.VoiceInfo.Culture.Name -eq "ru-RU" } | Select-Object -First 1
    if ($voice) { $synth.SelectVoice($voice.VoiceInfo.Name) }

    $wavPath = Join-Path $outDir ($call.Name + ".wav")
    $synth.SetOutputToWaveFile($wavPath)

    $transcript = New-Object System.Text.StringBuilder
    foreach ($line in $call.Lines) {
        $role = $line[0]
        $text = $line[1]
        if ($role -eq "m") {
            $synth.Rate = 1
            $prefix = "Менеджер. "
        } else {
            $synth.Rate = -2
            $prefix = "Клиент. "
        }
        $roleName = if ($role -eq "m") { "Менеджер" } else { "Клиент" }
        [void]$transcript.AppendLine($roleName + ": " + $text)
        $synth.Speak($prefix + $text)
    }
    $synth.SetOutputToNull()
    $synth.Dispose()

    $txtPath = Join-Path $outDir ($call.Name + ".txt")
    [System.IO.File]::WriteAllText($txtPath, $transcript.ToString(), [System.Text.Encoding]::UTF8)
    Write-Output ("Создан " + $wavPath + " (" + (Get-Item $wavPath).Length + " байт)")
}
