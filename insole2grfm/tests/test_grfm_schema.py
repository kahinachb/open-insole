import pytest

from grfm_schema import participant_of_subject


def test_the_loaded_subject_and_the_unloaded_one_are_the_same_participant():
    assert participant_of_subject("P01b") == "P01"
    assert participant_of_subject("P01") == "P01"


def test_a_subject_not_named_by_its_participant_code_is_refused():
    # A dataset still named by first names would silently split each Participant in two.
    with pytest.raises(ValueError, match="Alice2"):
        participant_of_subject("Alice2")
