"""Prompts for Step 6 document enrichment (structured JSON)."""

from __future__ import annotations

ENRICHMENT_SYSTEM = (
    "You extract structured metadata for a documentation retrieval system.\n"
    "Rules:\n"
    "- Output ONLY a single JSON object. No markdown code fences, no commentary before or after.\n"
    "- Ground every value in the SOURCE EXCERPT. Do not invent facts, policies, or URLs.\n"
    "- If unsupported by the excerpt, use an empty string or empty arrays.\n"
    "- short_summary: 1–3 neutral sentences, max ~400 characters.\n"
    "- tags: short lowercase labels, max 12 items.\n"
    "- entities: names of organizations, products, programmes, or key concepts that appear in the excerpt.\n"
    "- questions_answered: concise questions a reader could answer using this page.\n"
    "- relationships: use relation one of related_to, prerequisite, see_also, part_of; "
    "target_hint must be a short phrase grounded in the excerpt or headings.\n"
    "- content_intents: coarse labels such as overview, how_to, reference, policy, faq, contact, programme_detail."
)


def build_enrichment_user_message(
    markdown_excerpt: str,
    page_title: str,
    source_url: str,
) -> str:
    """User message with fixed JSON shape instruction + excerpt."""
    return (
        f'page_title: {page_title}\nsource_url: {source_url}\n\n'
        "Return JSON with exactly these keys:\n"
        '{"short_summary":"","tags":[],"entities":[{"name":"","type":"concept"}],'
        '"questions_answered":[],"relationships":[{"relation":"","target_hint":""}],'
        '"content_intents":[]}\n\n'
        "Omit array items you cannot ground. Use empty arrays where nothing applies.\n\n"
        "SOURCE EXCERPT (markdown):\n"
        f"{markdown_excerpt}"
    )
