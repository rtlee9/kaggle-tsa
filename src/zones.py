"""Threat zone maps."""

# map full threat zone space to left or right half
# Note _left_ means _stage left_
left_only_zones = [1, 2, 6, 8, 11, 13, 15]
right_only_zones = [3, 4, 7, 10, 12, 14, 16]
center_zones = [5, 9, 17]
left_zones = left_only_zones + center_zones
right_zones = right_only_zones + center_zones
left_right_map = {
    1: 3,
    2: 4,
    6: 7,
    8: 10,
    11: 12,
    13: 14,
    15: 16,
}
right_left_map = {v: k for k, v in left_right_map.items()}


def test_zone_maps():
    """Sanity checks for zone maps."""
    assert len(set(left_only_zones + center_zones + right_only_zones)) == 17
    assert len(left_only_zones) == len(right_only_zones)
    assert set(left_zones + right_zones) == set(left_only_zones + right_only_zones + center_zones)
    assert len(left_only_zones) == len(left_right_map)
    for z in left_only_zones:
        assert z in left_right_map
    for z in right_only_zones:
        assert z in left_right_map.values()
    for z in right_only_zones:
        assert z in right_left_map
    for z in left_only_zones:
        assert z in right_left_map.values()
