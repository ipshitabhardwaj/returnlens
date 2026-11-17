"""Track C - LLM as EVENT REASONER (not forecaster).
Frozen small LLM -> structured fields -> numeric features. Skip gate follows the ReasonCast idea."""
import json, math

MANDATORY = {
    "Diwali",
    "Dussehra",
    "Holi",
    "big_sale_festive",
    "republic_day_sale",
}  # skip NOT allowed for these

PROMPT = """You help an Indian e-commerce retailer forecast RETURNS. Event: "{event}" on {date}. Category: {category}.
Answer ONLY with JSON: {{"relevance": 0-1, "direction": "up|down|none", "lead_days": int, "lag_days": int,
"amplitude": "low|medium|high", "reason": "<=20 words"}}.
lead_days = days BEFORE the event when orders rise; lag_days = days AFTER when returns peak."""

FIELDS = {"direction": {"up", "down", "none"}, "amplitude": {"low", "medium", "high"}}


def route_event(event, calendar_hit):
    """'skip' = no LLM call, features are exactly 0 (so stable periods are never polluted)."""
    return "basic" if (event in MANDATORY or calendar_hit) else "skip"


def parse(text):
    try:
        d = json.loads(text[text.index("{") : text.rindex("}") + 1])
    except Exception:
        return None
    if d.get("direction") not in FIELDS["direction"] or d.get("amplitude") not in FIELDS["amplitude"]:
        return None
    d["relevance"] = float(min(max(d.get("relevance", 0), 0), 1))
    d["lead_days"] = int(min(max(d.get("lead_days", 0), 0), 30))
    d["lag_days"] = int(min(max(d.get("lag_days", 0), 0), 30))
    return d


AMP = {"low": 0.3, "medium": 0.6, "high": 1.0}


def to_features(d, d2):
    """d2 = date - event_date in days (negative = before; same convention as d2_* calendar features).
    Bump spans [-lead_days, +lag_days]; height = relevance * amplitude; sign from direction."""
    if d is None or d["direction"] == "none":
        return 0.0
    sign = 1 if d["direction"] == "up" else -1
    centre = (d["lag_days"] - d["lead_days"]) / 2
    width = max(3.0, (d["lag_days"] + d["lead_days"]) / 2)
    return sign * d["relevance"] * AMP[d["amplitude"]] * math.exp(-(((d2 - centre) / width) ** 2))


def run_llm(pipe, event, date, category):
    # pipe = transformers.pipeline("text-generation", "Qwen/Qwen2.5-1.5B-Instruct", ...)
    msgs = [{"role": "user", "content": PROMPT.format(event=event, date=date, category=category)}]
    out = pipe(msgs, max_new_tokens=120, do_sample=False)[0]["generated_text"][-1]["content"]
    return parse(out)
