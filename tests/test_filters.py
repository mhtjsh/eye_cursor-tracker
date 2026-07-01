from eye_cursor.filters import OneEuroFilter, PointSmoother


def test_one_euro_filter_starts_at_first_value():
    filt = OneEuroFilter()
    assert filt.filter(10.0, 1.0) == 10.0


def test_point_smoother_returns_pair():
    smoother = PointSmoother()
    x, y = smoother.filter(10.0, 20.0, 1.0)
    assert x == 10.0
    assert y == 20.0
