"""
Review stage: verify and repair a finished translation against the CN source.

Design notes
------------
This is a separate pipeline stage, not a translation mode. It consumes what
`translate` produced and emits a corrected HTML file per chapter, so a package of
freshly translated chapters can be verified before it is inserted into the book.

Why a dedicated prompt instead of reusing the translation prompt: the translation
prompt is a 900-line literary brief that tells the model how to *write*. Review
needs the opposite framing -- minimal intervention, four defect classes, and an
explicit ban on stylistic rewriting. Mixing the two makes the model re-translate
the chapter, which destroys already-approved wording.

Glossary scoping: only terms whose CN key occurs in the chapter are injected.
Measured on this book, that is a median of 47 entries out of 1773, i.e. ~1.2k
tokens instead of ~60k, and it keeps unrelated `НЕ "..."` prohibitions from
misfiring on words the chapter never uses.

Safety invariant: the paragraph count must survive. A review reply with a
different number of <p> elements is rejected and the chapter is left untouched,
because a merged or split paragraph breaks EPUB assembly and silently changes
the text the user already read.
"""

from __future__ import annotations

import html
import json
import os
import re
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

DEFAULT_TIMEOUT = 900
ENV_FILE = Path.home() / ".hermes" / ".env"


# ------------------------------------------------------------------ source text

def cn_chapter_text(epub_path: str, chapter_rel: str) -> str:
    """Plain text of one chapter of the CN source EPUB."""
    with zipfile.ZipFile(epub_path) as z:
        names = z.namelist()
        if chapter_rel in names:
            target = chapter_rel
        else:
            base = os.path.basename(chapter_rel)
            matches = [n for n in names if os.path.basename(n) == base]
            if not matches:
                raise FileNotFoundError(f"нет {chapter_rel} в {epub_path}")
            target = matches[0]
        raw = z.read(target).decode("utf-8", "replace")

    paras = []
    for p in re.findall(r"<p[^>]*>(.*?)</p>", raw, re.S):
        t = html.unescape(re.sub(r"<[^>]+>", "", p))
        t = re.sub(r"\s+", " ", t).strip()
        if t:
            paras.append(t)
    return "\n".join(paras)


def count_paragraphs(html_text: str) -> int:
    return len(re.findall(r"<p[^>]*>", html_text, re.I))


def source_paragraph_count(epub_path: str, chapter_rel: str) -> int:
    """Paragraph count of the RAW source chapter.

    This is the count the book assembly needs: the translated chapter is spliced
    against the source chapter paragraph by paragraph. It is therefore the only
    count a stage may legitimately converge to when the incoming file carries
    torn paragraphs (a `<strong>` split across `<p>` boundaries, `…` fragments).
    """
    with zipfile.ZipFile(epub_path) as z:
        names = z.namelist()
        target = chapter_rel if chapter_rel in names else None
        if target is None:
            base = os.path.basename(chapter_rel)
            matches = [n for n in names if os.path.basename(n) == base]
            if not matches:
                raise FileNotFoundError(f"нет {chapter_rel} в {epub_path}")
            target = matches[0]
        raw = z.read(target).decode("utf-8", "replace")
    return count_paragraphs(raw)


# --------------------------------------------------------------------- glossary

def scope_glossary(glossary: dict[str, dict], cn_text: str,
                   max_terms: int = 400) -> list[dict]:
    """Keep only entries whose CN key actually appears in this chapter."""
    hits = []
    for original, entry in glossary.items():
        if not original:
            continue
        if original in cn_text:
            hits.append({
                "original": original,
                "rus": (entry or {}).get("rus", ""),
                "note": (entry or {}).get("note", ""),
            })
    # Longest keys first: compounds should be read before their components.
    hits.sort(key=lambda e: -len(e["original"]))
    return hits[:max_terms]


def format_glossary(entries: list[dict]) -> str:
    if not entries:
        return "(в этой главе глоссарных терминов не найдено)"
    lines = []
    for e in entries:
        note = f"  — {e['note']}" if e.get("note") else ""
        lines.append(f"{e['original']} = {e['rus']}{note}")
    return "\n".join(lines)


# ----------------------------------------------------------------- provider I/O

def _load_env_pairs(prefix: str) -> dict[str, str]:
    if not ENV_FILE.exists():
        return {}
    out = {}
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith(prefix) and "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def resolve_endpoint(provider_id: str, provider_conf: dict) -> str:
    base = (provider_conf or {}).get("base_url") or ""
    if not base:
        raise ValueError(
            f"у провайдера {provider_id} нет base_url — "
            f"стадия review поддерживает только OpenAI-совместимые эндпоинты"
        )
    return base


PLACEHOLDER_KEYS = ("DUMMY", "NONE", "X", "-", "")


def build_headers(provider_id: str, api_key: str | None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    # Placeholder keys are the documented convention for proxy providers
    # (--api-key DUMMY), so they must never win over a real credential.
    real_key = None
    if api_key and api_key.strip().upper() not in PLACEHOLDER_KEYS:
        real_key = api_key.strip()

    if provider_id == "gumloop":
        env = _load_env_pairs("GUMLOOP_")
        key = real_key or env.get("GUMLOOP_API_KEY")
        uid = env.get("GUMLOOP_USER_ID")
        if not key or not uid:
            raise ValueError("нужны GUMLOOP_API_KEY и GUMLOOP_USER_ID "
                             "(настройки или ~/.hermes/.env)")
        headers["Authorization"] = f"Bearer {key}"
        headers["x-auth-key"] = uid
        return headers

    headers["Authorization"] = f"Bearer {real_key or 'DUMMY'}"
    return headers


def call_model(endpoint: str, headers: dict, model_id: str, prompt: str,
               max_tokens: int, timeout: int = DEFAULT_TIMEOUT) -> str:
    # DS4F reasoning drift (2026-09-11): reasoning is now the default upstream
    # and burns the whole max_tokens budget on reasoning_content, leaving an
    # empty content field. Review calls go through this raw path (not the
    # agentrouter handler), so the effort cap must be set here as well.
    # The budget must also cover both reasoning AND content.
    reasoning_off = "deepseek" in str(model_id).lower()
    payload = {
        "model": model_id,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if reasoning_off:
        payload["reasoning_effort"] = "none"

    req = urllib.request.Request(endpoint, data=json.dumps(payload).encode("utf-8"),
                                 headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            payload = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:500]
        raise RuntimeError(f"HTTP {e.code}: {detail}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"нет соединения: {e}")

    choices = payload.get("choices") or []
    if not choices:
        raise RuntimeError(f"пустой ответ: {json.dumps(payload)[:300]}")
    choice = choices[0]
    content = (choice.get("message") or {}).get("content") or ""
    if not content:
        # DS4F drift (2026-09-19): reasoning is emitted even with
        # reasoning_effort=none and consumes the whole max_tokens budget, so
        # content comes back empty with finish_reason=max_tokens. Report that
        # plainly instead of letting it surface as a nameless empty_reply.
        reasoning = (choice.get("message") or {}).get("reasoning_content") or ""
        raise RuntimeError(
            "пустой content (finish_reason=%s, reasoning_chars=%d, max_tokens=%s)"
            % (choice.get("finish_reason"), len(reasoning), max_tokens))
    return content


# --------------------------------------------------------------- reply cleaning

def extract_html(reply: str) -> str:
    """Strip markdown fences and leading prose the model may have added."""
    text = reply.strip()
    fenced = re.search(r"```(?:html)?\s*(.+?)```", text, re.S)
    if fenced:
        text = fenced.group(1).strip()
    # Drop anything before the first tag so a stray preamble cannot leak in.
    first = re.search(r"<(?:\?xml|!DOCTYPE|html|body|div|h1|p)\b", text, re.I)
    if first:
        text = text[first.start():]
    return text.strip()


# ------------------------------------------------------------------ single pass

def paragraph_texts(html_text: str) -> list[str]:
    """Plain text of each <p>, in document order."""
    out = []
    for m in re.findall(r"<p[^>]*>(.*?)</p>", html_text, re.S):
        t = html.unescape(re.sub(r"<[^>]+>", "", m))
        out.append(re.sub(r"\s+", " ", t).strip())
    return out


def diff_paragraphs(before_html: str, after_html: str,
                    max_chars: int = 400) -> list[dict]:
    """Per-paragraph edit list, so the user can rule on each change.

    Compared as plain text rather than raw HTML so that tag-level noise does not
    show up as an edit; the paragraph-count invariant guarantees the two lists
    line up index for index.
    """
    a = paragraph_texts(before_html)
    b = paragraph_texts(after_html)
    edits = []
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            edits.append({
                "paragraph": i,
                "before": x[:max_chars],
                "after": y[:max_chars],
            })
    return edits


def review_chapter(*, cn_epub: str, chapter_rel: str, translated_html: str,
                   glossary: dict[str, dict], prompt_template: str,
                   endpoint: str, headers: dict, model_id: str,
                   project_rules: str = "",
                   timeout: int = DEFAULT_TIMEOUT) -> dict:
    """Review one chapter. Returns a result dict; never raises on model issues."""
    cn_text = cn_chapter_text(cn_epub, chapter_rel)
    entries = scope_glossary(glossary, cn_text)
    src_paras = count_paragraphs(translated_html)
    # The source chapter's own count — the target a torn input may converge to.
    try:
        en_paras = source_paragraph_count(cn_epub, chapter_rel)
    except Exception:
        en_paras = src_paras

    prompt = prompt_template
    for key, value in (("{glossary}", format_glossary(entries)),
                       ("{rules}", project_rules or "(дополнительных правил нет)"),
                       ("{source}", cn_text),
                       ("{input_paras}", str(src_paras)),
                       ("{source_paras}", str(en_paras)),
                       ("{text}", translated_html)):
        if key in prompt:
            prompt = prompt.replace(key, value)

    # Output must fit the whole chapter HTML with headroom; a tight cap
    # truncates the reply and the paragraph guard then rejects a valid review.
    # DS4F (2026-09-19) emits 70k+ chars of reasoning per full chapter even with
    # reasoning_effort=none, i.e. ~20-25k tokens BEFORE any content: at the old
    # cap (len/2 ≈ 23k) the model returns empty content with
    # finish_reason=max_tokens. So the cap must cover reasoning + content.
    # Gl.79 (2026-09-19): polish burned 98k reasoning chars (~30k tokens) on a
    # 34k-char chapter and still hit the 32k cap -> budget raised to 64k (the
    # model's max_output_tokens) instead of len(html).
    budget = max(64000, len(translated_html))

    # One call, no re-ask. A torn input normally never reaches this stage: the hr
    # pipeline repairs it deterministically right after translation
    # (skills/.../hundred-reigns-mt/scripts/repair_tears.py), because a second
    # pass over a full torn chapter burned the whole 64k token budget on
    # reasoning twice in a row and returned empty content
    # (finish_reason=max_tokens, reasoning_chars 207k-212k). The count guard
    # below is the net, not the path.
    try:
        reply = call_model(endpoint, headers, model_id, prompt, budget, timeout)
    except Exception as exc:
        return {"chapter": chapter_rel, "ok": False, "status": "api_error",
                "error": str(exc), "glossary_terms": len(entries)}

    cleaned = extract_html(reply)
    out_paras = count_paragraphs(cleaned)

    if not cleaned or out_paras == 0:
        return {"chapter": chapter_rel, "ok": False, "status": "empty_reply",
                "glossary_terms": len(entries), "reply_head": reply[:200]}

    # Paragraph count is the one invariant EPUB assembly depends on: the count
    # must be the incoming file's, or — when the input arrived torn (extra <p>
    # from split tags, `…` fragments) — the source chapter's own count, which is
    # what assembly splices against. Growth past the input is always rejected.
    merged = src_paras - out_paras if out_paras < src_paras else 0
    if out_paras != src_paras and not (merged > 0 and out_paras == en_paras):
        return {"chapter": chapter_rel, "ok": False,
                "status": "paragraph_mismatch",
                "source_paragraphs": src_paras, "review_paragraphs": out_paras,
                "en_paragraphs": en_paras,
                "glossary_terms": len(entries)}

    counts_match = out_paras == src_paras
    changed = cleaned.strip() != translated_html.strip()
    edits = diff_paragraphs(translated_html, cleaned) if (changed and counts_match) else []
    return {"chapter": chapter_rel, "ok": True,
            "status": "changed" if changed else "unchanged",
            "source_paragraphs": src_paras, "review_paragraphs": out_paras,
            "en_paragraphs": en_paras, "merged_paragraphs": merged,
            "glossary_terms": len(entries),
            "cn_chars": len(cn_text),
            "edit_count": len(edits),
            "edits": edits,
            "html": cleaned}
