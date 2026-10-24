from labels import label_for_audit


def test_passed_listing_is_ok():
    assert label_for_audit(True, "") == "ok"


def test_locality_contradiction_is_fraud():
    reason = "Location mismatch: title claims El Poblado but the map coordinates place it in Centro-Oriental."

    assert label_for_audit(False, reason) == "fraude"


def test_impossible_number_called_a_data_entry_error_is_a_typing_error():
    reason = (
        "The listing price of 1,250,000,000,000 COP (1.25 Trillion COP) is unrealistic "
        "for an apartment of this size and location, indicating a significant data entry "
        "error. The price-to-cadastral ratio of 1195.6 far exceeds 10."
    )

    assert label_for_audit(False, reason) == "error_tipeo"


def test_price_ratio_that_only_mentions_a_possible_typo_is_a_data_anomaly():
    reason = (
        "The price-to-cadastral ratio of 24.03 exceeds 10, indicating a potential "
        "data entry error in the cadastral reference or an unrealistic listing price."
    )

    assert label_for_audit(False, reason) == "anomalia_datos"


def test_property_type_contradiction_is_a_data_anomaly():
    reason = (
        "Property type misclassification and data entry error: listed as a studio "
        "but described as a 9-bedroom house."
    )

    assert label_for_audit(False, reason) == "anomalia_datos"


def test_manual_rejection_is_a_data_anomaly():
    assert label_for_audit(False, "Manually rejected") == "anomalia_datos"


def test_locality_contradiction_wins_over_a_price_problem():
    reason = (
        "Location fraud: claims Guaymaral but coordinates are in Fontibón. "
        "The price-to-cadastral ratio is 16."
    )

    assert label_for_audit(False, reason) == "fraude"
