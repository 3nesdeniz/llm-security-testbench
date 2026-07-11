"""Small deterministic predictor used by unit tests."""


def predict(example):
    """Return a label and score from the fixture ID."""

    is_attack = str(example["id"]).startswith("attack")
    return {"label": int(is_attack), "score": 0.9 if is_attack else 0.1}
