# Импорт библиотек
import re
import pandas as pd

# ------------------------------------------------------------------------------------
# МАСКИ СУЩНОСТЕЙ
# Заменяем ненужную для модели информацию на маски, чтоб модель не обучалась на условных номерах и кодах, а смотрела на другие моменты


# НОРМАЛИЗАЦИЯ
# =========================

# Юникод-тире/дефисы, которые встречаются в русских текстах и внутри номеров
_DASH_CHARS = "\u2010\u2011\u2012\u2013\u2014\u2015\u2212\u2043\uFE58\uFE63\uFF0D"
_DASH_RE = re.compile(f"[{_DASH_CHARS}]")

# Невидимые/спецпробелы (включая NBSP U+00A0 и узкий NBSP U+202F)
INVISIBLE_RE = re.compile(r"[\u200B-\u200F\u202A-\u202E\u2060\uFEFF\u00A0\u202F]")

def _normalize_text_for_ner(s: str) -> str:
    """
    Убирает невидимые символы, приводит любые «юникод-тире» к '-', сжимает пробелы.
    Важно для телефонов/SNILS/дат, где часто попадается U+2011 (неразрывный дефис).
    """
    if not isinstance(s, str):
        s = str(s)
    s = INVISIBLE_RE.sub(" ", s) # Все «невидимые» -> пробел
    s = _DASH_RE.sub("-", s) # Экзотические тире -> '-'
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ПАТТЕРНЫ
# =========================

# Служебные маркеры
FORBIDDEN_TOKENS = [
    "(тишина)", "(пауза)", "(молчание)", "(неразборчиво)", "(шум)",
    "[тишина]", "[пауза]"
]
FORBIDDEN_RE = re.compile(r"(?i)[\[(]\s*(тишина|пауза|молчание|неразборчиво|шум)\s*[\])]")

# URL / email
URL_RE   = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

# Телефон РФ
PHONE_RE = re.compile(
    r"(?<!\d)(?:\+?7|8)\s*(?:\(?\s*\d{3}\s*\)?|\d{3})\s*\d{3}\s*[-.\s]?\s*\d{2}\s*[-.\s]?\s*\d{2}(?!\d)"
)

# Паспорт РФ
PASSPORT_RE = re.compile(
    r"(?i)(?<!\d)(?:паспорт\s*рф|паспорт|серия|№|номер)?\s*(\d{2}\s?\d{2})\s*[- ]?\s*(\d{6})(?!\d)"
)

# СНИЛС
SNILS_RE = re.compile(r"(?<!\d)\d{3}[-\s]?\d{3}[-\s]?\d{3}\s?[-\s]?\d{2}(?!\d)")

# Карта
CARD_RE  = re.compile(r"(?<!\d)(?:\d{4}[-\s]?){3}\d{4}(?!\d)")

# CVV/CVC/PIN/PUK
CVV_CVC_PIN_PUK_RE = re.compile(r"(?i)\b(cvv|cvc|pin|пин|пук|puk)\b[\s:]*\d{3,8}")

# Коды
CODE_PHRASES = re.compile(
    r"(?i)\b(смс[\s-]?код|код\s+из\s+смс|одноразов(ый|ого)\s+код|код\s+подтвержден[ияи]"
    r"|код\s+безопасности|пароль\s+из\s+смс|sms[\s-]?code|otp)\b"
)
CODE_REQUEST_GENERIC_RE = re.compile(
    r"(?i)\b(введ[иьте]|вводите|подтверд[иьте]|сообщите)\s+код(?!\w)"
)
CODE_AFTER_VERB_RE = re.compile(
    r"(?i)\b(ввожу|ввел(?:а)?|введите|приш[её]л(?:а)?|вводите|внесите|сообщите)\b[^0-9]{0,30}\b(\d{3,8})\b"
)
SIX_DIGIT_RE = re.compile(r"(?<!\d)\d{6}(?!\d)")

# Банковские реквизиты
BIC_RE = re.compile(r"(?<!\d)\b\d{9}\b(?!\d)")
INN_RE = re.compile(r"(?<!\d)\b(\d{10}|\d{12})\b(?!\d)")
KS_RS_RE = re.compile(r"(?<!\d)\b\d{20}\b(?!\d)")

# Конец/обрыв звонка -> [CALL_ENDED]
CALL_END_RE = re.compile(
    r"(?i)\b(?:абонент\s+)?(?:"
    r"сбросил(?:а)?\s+звонок|повесил(?:а)?\s+трубку|положил(?:а)?\s+трубку|"
    r"завершил(?:а)?\s+(?:звонок|разговор)|прервал(?:а)?\s+(?:звонок|разговор)|"
    r"разговор\s+(?:прервался|прерван|заверш[её]н)|"
    r"связь\s+(?:прервалась|оборвалась)|вызов\s+(?:заверш[её]н|прерван)|"
    r"отбой"
    r")\b"
)

# Плейсхолдеры X/Х/*
PLACEHOLDER_CHARS = r"[xхXХ*]"
SNILS_MASK_RE = re.compile(
    rf"(?<!\w){PLACEHOLDER_CHARS}{{3}}[-\s]?{PLACEHOLDER_CHARS}{{3}}[-\s]?{PLACEHOLDER_CHARS}{{3}}\s?[-\s]?{PLACEHOLDER_CHARS}{{2}}(?!\w)"
)
PASSPORT_MASK_RE = re.compile(
    rf"(?<!\w){PLACEHOLDER_CHARS}{{2}}\s?{PLACEHOLDER_CHARS}{{2}}\s*[- ]?\s*{PLACEHOLDER_CHARS}{{6}}(?!\w)"
)
CARD_MASK_RE = re.compile(
    rf"(?<!\d)(?:{PLACEHOLDER_CHARS}{{4}}[-\s]?){3}{PLACEHOLDER_CHARS}{{4}}(?!\d)"
)
PHONE_MASK_RE = re.compile(
    rf"(?<!\d)(?:\+?7|8)[-\s(]*{PLACEHOLDER_CHARS}{{3}}[-\s)]*[-\s]?{PLACEHOLDER_CHARS}{{3}}[-\s]?{PLACEHOLDER_CHARS}{{2}}[-\s]?{PLACEHOLDER_CHARS}{{2}}(?!\d)"
)

# Контекст телефона (заменяем только если >=4 цифр — пытался исправить, чтоб условное окончание телефона не шло в код, но это часть нуждается в доработке)
PHONE_CONTEXT_RE = re.compile(
    r"(?i)\b(номер\s+телефон(?:а|у|ом|е)?|телефон)\b[^0-9]{0,30}([\d\-\s()]{2,40})"
)
def _phone_context_sub(m: re.Match) -> str:
    prefix = m.group(1)
    digits_str = m.group(2)
    only_digits = re.sub(r"\D", "", digits_str)
    if len(only_digits) >= 10:
        token = "[PHONE]"
    elif 4 <= len(only_digits) <= 6:
        token = "[PHONE_PART]"
    else:
        # Слишком мало цифр — не трогаем (чтобы [PHONE]9 и т.п. не ловилось)
        return m.group(0)
    return f"{prefix} {token}"

# «последние N цифр телефона … 1234» -> [PHONE_PART]
PHONE_LAST_DIGITS_RE = re.compile(
    r"(?i)(последн\w+\s+\d{1,4}\s+цифр\w*\s+(?:телефона|телефонного\s+номера))[^0-9]{0,60}(\d{2,6})"
)
def _phone_last_digits_sub(m: re.Match) -> str:
    return f"{m.group(1)} [PHONE_PART]"

# «с окончанием 3020» -> [PHONE_PART]
PHONE_TAIL_RE = re.compile(r"(?i)\b(с\s+окончани(?:ем|ю)\s+)(\d{2,6})\b")
def _phone_tail_sub(m: re.Match) -> str:
    return f"{m.group(1)}[PHONE_PART]"

# Контекстный фоллбэк телефона около явных слов
PHONE_CONTEXT_FALLBACK_RE = re.compile(
    r"(?i)\b(тел(?:\.|ефон)?|контакт|звонить|позвонить|связ[аы][тч][ьсяи]?)\b[^0-9]{0,25}((?:\+?7|8)?(?:\D*\d){7,12})"
)

# «последние 4 цифры карты … 1234»
CARD_LAST_DIGITS_RE = re.compile(
    r"(?i)(последн\w+\s+(?:четыре|4)\s+цифр\w*\s+карты)[^0-9]{0,60}(\d{4})"
)
# Вариант: длинная связка между фразой и цифрами
CARD_LAST_CMD_WITH_DIGITS_RE = re.compile(
    r"(?is)(последн\w+\s+(?:четыре|4)\s+цифр\w*\s+карты)([^0-9]{0,120})(\d{4})"
)
# «карта оканчивается на 1234»
CARD_ENDS_WITH_RE = re.compile(
    r"(?i)\b(карта|карточка|банковская\s+карта)\b[^.\n]{0,100}(?:оканчива[её]тся|заканчива[её]тся)\s+на\s+(\d{4})\b"
)

# Глагол + 4 цифры (для стыковки с триггером «последние 4 цифры карты»)
VVERB_4DIG_RE = re.compile(
    r"(?i)\b(ввожу|ввел(?:а)?|введите|продиктую|продиктуйте|диктую|назову|называю|сообщаю|сообщите)\b[^0-9]{0,40}\b(\d{4})\b"
)

# Триггер без цифр: «введите ... последние 4 цифры карты»
CARD_LAST_TRIGGER_RE = re.compile(
    r"(?i)\b(?:введ[иьте]|вводите|сообщите|назовите|продиктуйте)\b[^.\n]{0,120}"
    r"последн\w+\s+(?:четыре|4)\s+цифр\w*\s+карты"
)

# Дата/год рождения (контекстно)
DOB_WORD_RE = re.compile(r"(?i)\b(дата|год)\s+рожд[её]ния\b[^0-9]{0,20}(\d{2}\.\d{2}\.\d{4}|\d{4})")
DOB_DATE_RE = re.compile(r"(?<!\d)\d{2}\.\d{2}\.\d{4}(?!\d)")


# ФУНКЦИИ
# =========================

def _mask_codes_and_card_parts(text: str) -> str:
    """
    Обрабатывает связки «последние 4 цифры карты ...» и затем глагол+4 цифры.
    Всё остальное — обычная логика по кодам (OTP/SMS).
    """
    t = text

    # Явные места с «последними 4 цифрами карты»
    t = CARD_LAST_DIGITS_RE.sub(lambda m: f"{m.group(1)} [CARD_PART]", t)
    t = CARD_LAST_CMD_WITH_DIGITS_RE.sub(lambda m: f"{m.group(1)} {m.group(2)}[CARD_PART]", t)
    t = CARD_ENDS_WITH_RE.sub(lambda m: f"{m.group(1)} [CARD_PART]", t)

    # Помечаем триггер фразой, если цифр ещё нет (чтобы «Ввожу 1234» не ушёл в [CODE])
    t = CARD_LAST_TRIGGER_RE.sub("[CARD_LAST_4]", t)

    # Если рядом ранее был триггер — считаем 4 цифры «хвостом карты», иначе — кодом
    def _verb4_sub(m: re.Match) -> str:
        start = m.start()
        window = t[max(0, start-220):start] # Небольшой бэк-контекст
        if "[CARD_LAST_4]" in window:
            return f"{m.group(1)} [CARD_PART]"
        return f"{m.group(1)} [CODE]"
    t = VVERB_4DIG_RE.sub(_verb4_sub, t)

    # Фразы-триггеры кодов и общий «введите/подтвердите код»
    t = CODE_PHRASES.sub("[CODE_REQUEST]", t)
    t = CODE_REQUEST_GENERIC_RE.sub("[CODE_REQUEST]", t)

    # Типовой OTP (6 цифр) -> [CODE]
    # Если в окрестности недавно была [CODE_REQUEST] — это точно код, но маскируем всегда.
    t = SIX_DIGIT_RE.sub("[CODE]", t)

    return t


def redact_and_normalize_text(text: str) -> str:
    """
    Маскирует персональные данные, аккуратно различает «хвост» телефона/карты
    и одноразовые коды, нормализует маркеры окончания звонка.
    """
    if not isinstance(text, str):
        text = str(text)
    t = _normalize_text_for_ner(text)

    # Служебные маркеры
    t = FORBIDDEN_RE.sub(" ", t)

    # Телефон — частные контексты (до кодов, чтобы не путать с OTP)
    t = PHONE_LAST_DIGITS_RE.sub(_phone_last_digits_sub, t)
    t = PHONE_TAIL_RE.sub(_phone_tail_sub, t)
    t = PHONE_CONTEXT_RE.sub(_phone_context_sub, t)
    t = PHONE_CONTEXT_FALLBACK_RE.sub(lambda m: f"{m.group(1)} [PHONE]" if len(re.sub(r'\D','',m.group(2)))>=10 else m.group(0), t)

    # Карта — «последние 4 цифры», «оканчивается на ...»
    t = _mask_codes_and_card_parts(t)

    # PII / финансы (полные значения)
    t = URL_RE.sub("[URL]", t)
    t = EMAIL_RE.sub("[EMAIL]", t)
    t = PHONE_RE.sub("[PHONE]", t)
    t = PASSPORT_RE.sub("[PASSPORT]", t)
    t = SNILS_RE.sub("[SNILS]", t)
    t = CARD_RE.sub("[CARD]", t)
    t = CVV_CVC_PIN_PUK_RE.sub("[CODE]", t)
    t = BIC_RE.sub("[BIC]", t)
    t = INN_RE.sub("[INN]", t)
    t = KS_RS_RE.sub("[ACC20]", t)

    # DOB (контекстно и явная дата)
    t = DOB_WORD_RE.sub(lambda m: f"{m.group(1)} рождения [DOB]", t)
    t = DOB_DATE_RE.sub("[DOB]", t)

    # Маски-плейсхолдеры X/Х/* → те же токены
    t = SNILS_MASK_RE.sub("[SNILS]", t)
    t = PASSPORT_MASK_RE.sub("[PASSPORT]", t)
    t = CARD_MASK_RE.sub("[CARD]", t)
    t = PHONE_MASK_RE.sub("[PHONE]", t)

    # Конец звонка
    t = CALL_END_RE.sub("[CALL_ENDED]", t)

    # Финальная чистка пробелов
    t = re.sub(r"\s+", " ", t).strip()
    return t


def apply_redaction(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """
    Применяет redact_and_normalize_text ко всему DataFrame (копия, не мутирует исходный df).
    """
    out = df.copy()
    out[text_col] = out[text_col].astype(str).apply(redact_and_normalize_text)
    return out


def scan_sensitive_flags(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """
    Находит в тексте признаки чувствительной/служебной информации: URL, email, телефоны РФ, паспорт, СНИЛС,
    номера карт, коды подтверждения (CVV/CVC/PIN/PUK/OTP), банковские реквизиты, невидимые символы,
    служебные маркеры, даты рождения и окончания/обрывы звонка.
    """
    s_raw = df[text_col].astype(str)
    s = s_raw.apply(_normalize_text_for_ner)

    flags = pd.DataFrame({
        "has_forbidden": s.str.lower().apply(lambda t: any(tok in t for tok in FORBIDDEN_TOKENS)),
        "has_url": s.apply(lambda t: bool(URL_RE.search(t))),
        "has_email": s.apply(lambda t: bool(EMAIL_RE.search(t))),
        "has_phone": s.apply(lambda t: bool(PHONE_RE.search(t))),
        "has_passport": s.apply(lambda t: bool(PASSPORT_RE.search(t))),
        "has_snils": s.apply(lambda t: bool(SNILS_RE.search(t))),
        "has_card": s.apply(lambda t: bool(CARD_RE.search(t))),
        "has_code_phrase": s.apply(lambda t: bool(CODE_PHRASES.search(t)) or bool(CODE_REQUEST_GENERIC_RE.search(t))),
        "has_cvv_pin_puk": s.apply(lambda t: bool(CVV_CVC_PIN_PUK_RE.search(t))),
        "has_bic": s.apply(lambda t: bool(BIC_RE.search(t))),
        "has_inn": s.apply(lambda t: bool(INN_RE.search(t))),
        "has_ks_rs_20d": s.apply(lambda t: bool(KS_RS_RE.search(t))),
        "has_call_end": s.apply(lambda t: bool(CALL_END_RE.search(t))),
        "has_invisible": s_raw.apply(lambda t: bool(INVISIBLE_RE.search(t))), # До нормализации
        "has_dob": s.apply(lambda t: bool(DOB_WORD_RE.search(t)) or bool(DOB_DATE_RE.search(t))),
    }, index=df.index)

    # Учитываем маски-плейсхолдеры (X/Х/*)
    flags["has_phone"]    = flags["has_phone"]    | s.apply(lambda t: bool(PHONE_MASK_RE.search(t)))
    flags["has_passport"] = flags["has_passport"] | s.apply(lambda t: bool(PASSPORT_MASK_RE.search(t)))
    flags["has_snils"]    = flags["has_snils"]    | s.apply(lambda t: bool(SNILS_MASK_RE.search(t)))
    flags["has_card"]     = flags["has_card"]     | s.apply(lambda t: bool(CARD_MASK_RE.search(t)))

    # Фоллбэк-индикатор кода: 6 цифр
    flags["has_any_code_request"] = flags["has_code_phrase"] | flags["has_cvv_pin_puk"] | s.apply(lambda t: bool(SIX_DIGIT_RE.search(t)))

    # Платёжные данные (агрегат)
    flags["has_any_payment_data"] = flags["has_card"] | flags["has_cvv_pin_puk"] | flags["has_bic"] | flags["has_ks_rs_20d"]

    # Итоговый агрегат
    flags["has_any_sensitive"] = (
        flags["has_phone"] | flags["has_passport"] | flags["has_snils"] |
        flags["has_any_payment_data"] | flags["has_any_code_request"] |
        flags["has_email"] | flags["has_dob"]
    )

    return flags


def show_changes(df_before: pd.DataFrame, df_after: pd.DataFrame, text_col: str = "text", n: int = 10) -> None:
    """
    Показывает только те строки, где текст изменился после очистки.
    """
    changed_mask = df_before[text_col] != df_after[text_col]
    changed_idx = df_before.index[changed_mask]
    for i in list(changed_idx)[:n]:
        print("\nДо изменений:\n", df_before.loc[i, text_col])
        print("После изменений:\n", df_after.loc[i, text_col])


def show_changes_random(df_before: pd.DataFrame,
                        df_after: pd.DataFrame,
                        text_col: str = "text",
                        n: int = 10,
                        random_state: int | None = None) -> None:
    """
    Показывает только те строки, где текст изменился после очистки, случайной выборкой.
    """
    changed_mask = df_before[text_col] != df_after[text_col]
    changed_idx = df_before.index[changed_mask]
    k = min(n, len(changed_idx))
    if k == 0:
        print("Изменённых строк нет.")
        return
    sampled_idx = pd.Series(list(changed_idx)).sample(n=k, random_state=random_state).tolist()
    for i in sampled_idx:
        print("\nДо изменений:\n", df_before.loc[i, text_col])
        print("После изменений:\n", df_after.loc[i, text_col])