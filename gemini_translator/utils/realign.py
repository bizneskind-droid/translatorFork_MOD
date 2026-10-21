"""
Realign module for recovering HTML structure (paragraph count matching 1:1)
when model translation finishes but fails validate_html_structure,
or when raw output lacks HTML tags.
"""
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)


def validate_html_structure(
    original_html: str,
    translated_html: str,
    strict: bool = True,
) -> tuple[bool, str, dict]:
    """
    Validates HTML structure and paragraph alignment between original and translation.

    Returns: (is_valid: bool, reason: str, details: dict)
    details contains:
      'src_p': int,
      'ru_p': int,
      'diff': int (ru_p - src_p),
      'matched': bool (ru_p == src_p),
      'missing_p_tags': bool,
      'has_content': bool,
    """
    details = {
        "src_p": 0,
        "ru_p": 0,
        "diff": 0,
        "matched": False,
        "missing_p_tags": False,
        "has_content": False,
    }

    if not translated_html or not translated_html.strip():
        return False, "Пустой контент перевода", details

    details["has_content"] = True

    # Count <p> tags
    src_p = len(re.findall(r"<p[ >]", original_html, re.I))
    ru_p = len(re.findall(r"<p[ >]", translated_html, re.I))
    details["src_p"] = src_p
    details["ru_p"] = ru_p
    details["diff"] = ru_p - src_p
    details["matched"] = (src_p == ru_p)

    if src_p > 0 and ru_p == 0:
        details["missing_p_tags"] = True
        return False, f"Отсутствуют теги <p> в переводе (в оригинале: {src_p})", details

    # Critical tags balance check
    for tag in ["table", "ul", "ol"]:
        opens = len(re.findall(rf"<{tag}[ >]", translated_html, re.I))
        closes = len(re.findall(rf"</{tag}>", translated_html, re.I))
        if opens != closes:
            return False, f"Несбалансированный тег <{tag}> ({opens} открыто, {closes} закрыто)", details

    # Length ratio check (ensure not truncated)
    if len(original_html) > 500 and len(translated_html) < len(original_html) * 0.35:
        return False, f"Слишком короткий перевод ({len(translated_html)} симв. vs {len(original_html)} в оригинале)", details

    # If strict check is requested, paragraph count must match 1:1
    if strict and src_p > 0 and src_p != ru_p:
        diff_str = f"{details['diff']:+d}"
        return False, f"Расхождение по числу абзацев: EN={src_p}, RU={ru_p} (разница: {diff_str})", details

    # If not strict, allow non-critical ratio deviations
    if not strict and src_p > 0:
        ratio = ru_p / src_p
        if ratio < 0.5 or ratio > 2.0:
            return False, f"Критическое расхождение <p> ({src_p} vs {ru_p})", details

    return True, "", details


def wrap_plain_text_to_paragraphs(text: str) -> str:
    """
    Wrap plain text or markdown without <p> tags into <p>...</p> elements.
    """
    if not text or not text.strip():
        return ""

    cleaned = re.sub(r"^```(?:html|markdown|text)?\s*", "", text.strip(), flags=re.MULTILINE)
    cleaned = re.sub(r"```\s*$", "", cleaned.strip(), flags=re.MULTILINE).strip()

    if re.search(r"<p\b", cleaned, re.I):
        return cleaned

    # Split by double newlines into paragraphs
    blocks = re.split(r"\n\s*\n+", cleaned)
    p_tags = []
    for b in blocks:
        b_clean = b.strip()
        if not b_clean:
            continue
        # Replace internal line breaks with spaces
        b_clean = re.sub(r"\s*\n\s*", " ", b_clean)
        p_tags.append(f"<p>{b_clean}</p>")

    return "\n".join(p_tags)


def try_realign_translated_html(
    original_html: str,
    translated_html: str,
    worker=None,
    model: str = "gemini-3.8-flash-high",
    timeout: int = 300,
) -> str | None:
    """
    Attempts to align translated HTML paragraph structure 1:1 with original HTML.
    Uses hundred_reigns_realign_prompt.txt and the agy CLI.
    Returns realigned HTML if successful, or None on failure.
    """
    if not translated_html or not translated_html.strip():
        return None

    # Pre-wrap if <p> tags are completely missing
    working_html = translated_html
    if not re.search(r"<p\b", working_html, re.I):
        logger.info("[REALIGN] Входной текст без тегов <p>, предварительно оборачиваем абзацы.")
        working_html = wrap_plain_text_to_paragraphs(working_html)

    prompt_file = Path(__file__).resolve().parent.parent.parent / "config" / "hundred_reigns_realign_prompt.txt"
    if not prompt_file.is_file():
        logger.warning(f"[REALIGN] Prompt file not found: {prompt_file}")
        return None

    system_prompt = prompt_file.read_text(encoding="utf-8")
    full_prompt = (
        f"{system_prompt}\n\n"
        f"# ОРИГИНАЛ (ENGLISH HTML)\n{original_html}\n\n"
        f"# ПЕРЕВОД (RUSSIAN HTML)\n{working_html}\n"
    )

    realigned_text = ""

    # Сначала пробуем через воркера, если он передан
    if worker:
        try:
            logger.info(f"[REALIGN] Вызов realign через worker ({len(full_prompt)} символов)...")
            realigned_text = worker._call_model(full_prompt, timeout=timeout)
        except Exception as e:
            logger.warning(f"[REALIGN] Вызов через worker завершился с ошибкой: {e}")
            realigned_text = ""

    if not realigned_text or not realigned_text.strip():
        try:
            from gemini_translator.review import _call_model_agy
            logger.info(f"[REALIGN] Calling agy CLI for structure realignment ({len(full_prompt)} chars)...")
            realigned_text = _call_model_agy(model, full_prompt, timeout=timeout)
        except Exception as e:
            logger.warning(f"[REALIGN] agy realignment call failed: {e}")
            realigned_text = ""

    if not realigned_text or not realigned_text.strip():
        logger.warning("[REALIGN] Model returned empty realignment output.")
        return None

    # Clean markdown fences if model wrapped in ```html ... ```
    cleaned = re.sub(r"^```(?:html)?\s*", "", realigned_text.strip(), flags=re.MULTILINE)
    cleaned = re.sub(r"```\s*$", "", cleaned.strip(), flags=re.MULTILINE).strip()

    return cleaned


def validate_and_realign(
    original_html: str,
    translated_html: str,
    worker=None,
    model: str = "gemini-3.8-flash-high",
    timeout: int = 300,
) -> tuple[bool, str, str]:
    """
    Validates HTML structure and paragraph count. If validation fails or
    paragraphs do not match 1:1, runs realignment via model and re-validates.

    Returns: (is_valid: bool, reason: str, final_html: str)
    """
    is_valid, reason, stats = validate_html_structure(original_html, translated_html, strict=True)
    if is_valid:
        return True, "", translated_html

    logger.info(f"[REALIGN] Запуск этапа realign из-за валидации: {reason}")
    realigned = try_realign_translated_html(
        original_html, translated_html, worker=worker, model=model, timeout=timeout
    )
    if not realigned:
        return False, f"Realign не вернул результат. Исходная причина: {reason}", translated_html

    is_valid_after, reason_after, stats_after = validate_html_structure(original_html, realigned, strict=True)
    if is_valid_after:
        logger.info(f"[REALIGN] Структура успешно выровнена 1:1 ({stats_after['ru_p']}/{stats_after['src_p']} <p>)")
        return True, "", realigned

    logger.warning(f"[REALIGN] После realign: {reason_after}")
    return False, reason_after, realigned
