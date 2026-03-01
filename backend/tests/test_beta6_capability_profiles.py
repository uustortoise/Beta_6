from ml.beta6.capability_profiles import (
    DEFAULT_PROFILE,
    infer_room_type,
    select_capability_profile,
)


def test_infer_room_type_uses_room_name_hints():
    assert infer_room_type("Master_Bedroom") == "bedroom"
    assert infer_room_type("family_room_main") == "livingroom"
    assert infer_room_type("toilet_2") == "bathroom"


def test_select_capability_profile_falls_back_to_generic():
    profile = select_capability_profile("balcony")
    assert profile.profile_id == DEFAULT_PROFILE.profile_id
    assert profile.room_type == "generic"


def test_select_capability_profile_by_explicit_room_type():
    profile = select_capability_profile(room="room_01", room_type="livingroom")
    assert profile.room_type == "livingroom"
    assert profile.profile_id == "cap_profile_livingroom_v1"
