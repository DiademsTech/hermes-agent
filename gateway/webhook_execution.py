"""Per-delivery model choice from trusted route configuration, never event payload."""

def normalize_choice(raw):
    from hermes_constants import parse_reasoning_effort

    if not isinstance(raw, dict):
        raise ValueError("Invalid webhook model configuration")
    result = {}
    for key in ("model", "provider", "reasoning_effort"):
        value = raw.get(key)
        if value is None or value == "":
            continue
        if not isinstance(value, str) or len(value) > 240 or any(ord(c) < 32 for c in value):
            raise ValueError("Invalid webhook model configuration")
        value = value.strip()
        if value:
            result[key] = value
    if bool(result.get("model")) != bool(result.get("provider")):
        raise ValueError("Webhook model and provider must be supplied together")
    if result.get("reasoning_effort") and parse_reasoning_effort(result["reasoning_effort"]) is None:
        raise ValueError("Invalid webhook reasoning effort")
    return result


def source_choice(source):
    if source is None or getattr(getattr(source, "platform", None), "value", None) != "webhook":
        return {}
    return normalize_choice(getattr(source, "webhook_execution", None) or {})
