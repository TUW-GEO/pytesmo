


def test_cdf_matching():

    src = ...
    ref = ...

    matcher = CDFMatching().fit(src, ref)
    transformed = matcher.predict(src)

    # something along those lines
    assert ref == transformed
