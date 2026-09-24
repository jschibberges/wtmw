from __future__ import annotations

import nlp_utils


def test_clean_description_does_not_mutate_cached_stopwords():
    before = set(nlp_utils.get_german_stopwords())
    assert nlp_utils.TOPIC_ROLE_STOP - before  # sonst würde der Test nichts prüfen

    nlp_utils.clean_description_for_labels("Der Außenpolitiker spricht über den Krieg in der Ukraine.")

    assert nlp_utils.get_german_stopwords() == before
