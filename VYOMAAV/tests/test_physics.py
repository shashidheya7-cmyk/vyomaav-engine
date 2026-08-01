from physics.spatial import check_bbox_intersection

def test_bbox_intersection_true():
    assert check_bbox_intersection([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.5, 0.5, 0.5], [1.5, 1.5, 1.5]) is True

def test_bbox_intersection_false():
    assert check_bbox_intersection([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]) is False
