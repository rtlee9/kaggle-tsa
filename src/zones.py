"""Threat zone maps."""
from .constants import IMAGE_DIM
from .crop import crop_dims

# map full threat zone space to left or right half
# Note _left_ means _stage left_
left_only_zones = [1, 2, 6, 8, 11, 13, 15]
right_only_zones = [3, 4, 7, 10, 12, 14, 16]
arm_zones = [3, 4, 7]  # don't flip left <--> right
irreversible_zones = [16, 5, 17, 14, 12, 10, 7]  # don't flip front <--> back
unflippable_zones = [16, 9, 3, 4, 14, 12, 10, 7]  # don't flip top <--> bottom
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

image_dims = (IMAGE_DIM, IMAGE_DIM, IMAGE_DIM)
common_threat_body_map = {
    16: crop_dims(image_dims, top=20, left=70),
    14: crop_dims(image_dims, top=32, bottom=13, left=73),
    12: crop_dims(image_dims, top=47, bottom=25, left=61),
    10: crop_dims(image_dims, top=75, bottom=30, left=67),
    7: crop_dims(image_dims, top=95, bottom=60, left=57, right=118),
    3: crop_dims(image_dims, top=117, bottom=83, left=76, right=126),
    4: crop_dims(image_dims, bottom=92, left=70, right=126),
    9: crop_dims(image_dims, top=70, bottom=30, left=45, right=83),
    5: crop_dims(image_dims, top=100, bottom=75, left=30, right=98, back=94),
    17: crop_dims(image_dims, top=100, bottom=75, left=30, right=98, front=75),
}


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
