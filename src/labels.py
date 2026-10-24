"""Gold labels for a RealsLab listing audit."""

_FRAUD = (
    "fraud",
    "location mismatch",
    "location fraud",
    "location integrity",
    "location discrepancy",
    "location inconsistency",
    "location claim",
    "locality",
    "coordinates",
    "map pin",
    "map-derived",
)
_TYPING = ("data entry", "decimal", "typo", "mistyped")


def label_for_audit(passed: bool, reason: str) -> str:
    """Map an audit verdict to ok, fraude, error_tipeo, or anomalia_datos.

    A locality contradiction is fraud even when the price is also wrong.
    A number the auditor calls a data-entry mistake is a typing error.
    A price-to-cadastral ratio, a property-type contradiction, or a manual
    rejection is a data anomaly.
    """
    if passed:
        return "ok"
    text = reason.lower()
    if any(mark in text for mark in _FRAUD):
        return "fraude"
    if _is_typing(text):
        return "error_tipeo"
    return "anomalia_datos"


def _is_typing(text: str) -> bool:
    if not any(mark in text for mark in _TYPING):
        return False
    if "property type" in text or "misclassification" in text:
        return False
    ratio = "price-to-cadastral" in text or "price to cadastral" in text
    if ratio and "impossible" not in text and "trillion" not in text:
        return False
    return True
