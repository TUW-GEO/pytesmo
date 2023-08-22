import unittest

import pandas as pd
import numpy as np

from pytesmo.time_series.grouping import YearlessDatetime, TsDistributor
from datetime import datetime


class Test_YearlessDateTime(unittest.TestCase):

    def setUp(self) -> None:
        self.past = datetime(1900, 1, 2, 3, 4, 5)
        self.future = datetime(2104, 6, 7, 8, 9, 10)
        self.yearless = YearlessDatetime(3, 10, 0, 0, 0)

    def test_comparisons(self):
        assert self.yearless > YearlessDatetime.from_datetime(self.past)
        assert self.yearless < YearlessDatetime.from_datetime(self.future)
        assert self.yearless == self.yearless

    def test_doy(self):
        assert YearlessDatetime.from_datetime(self.future).doy == 159
        assert YearlessDatetime(12, 31).doy == 366
        assert YearlessDatetime(2, 29).doy == 60
        assert YearlessDatetime(1, 1).doy == 1

    def test_to_dt(self):
        assert YearlessDatetime.from_datetime(self.past).to_datetime(
            self.past.year) == self.past
        assert YearlessDatetime.from_datetime(
            self.future).to_datetime(years=[2104, 2111])[0] == self.future


class Test_TimeSeriesDistributionSet(unittest.TestCase):

    def setUp(self) -> None:
        df = pd.DataFrame(
            index=pd.date_range('2000-01-01T12', '2009-12-31T12', freq='D'))
        df['data'] = np.random.rand(df.index.size)
        df.loc[np.isin(df.index.month, [1, 7])] = np.nan
        df = df.dropna()
        self.df = df

    def test_filter_dates_only(self):
        dates = (
            datetime(2005, 6, 6, 12),
            datetime(2005, 5, 5, 12),
            datetime(2005, 4, 4, 12),
            datetime(2005, 3, 3, 12),
            datetime(2005, 2, 2, 12),
            datetime(2005, 2, 2, 1),  # not in input/output !!
            datetime(2005, 1, 5, 12),  # not in input/output !!
        )

        set1 = TsDistributor(dates=dates)

        d = set1.select(self.df)
        assert len(d.index) == 5
        assert np.all(dt in d.index for dt in dates[:5])

    def test_filter_daterange_only(self):
        set2 = TsDistributor(date_ranges=[
            (datetime(2004, 12, 20), datetime(2005, 2, 10, 11)),
            (datetime(2005, 2, 27), datetime(2005, 3, 1, 12)),
        ])

        d = set2.select(self.df)
        assert datetime(2005, 2, 1, 12) in d.index
        assert len(d.index) == (12 + (10 - 1)) + 3

    def test_filter_yearless_dates_only(self):
        yearless_dates = (
            YearlessDatetime(6, 6, 12, 0, 0),
            YearlessDatetime(2, 29, 12, 0, 0),
            YearlessDatetime.from_datetime(datetime(2000, 5, 5, 12)),
            YearlessDatetime(1, 1, 12),  # not in input/output !!
        )
        set3 = TsDistributor(yearless_dates=yearless_dates)

        d = set3.select(self.df)
        assert datetime(2005, 5, 5, 12) in d.index
        assert datetime(2008, 2, 29, 12) in d.index
        assert len(d.index) == 2 * len(np.unique(self.df.index.year)) + 3

    def test_filter_yearless_date_ranges_only(self):
        set4 = TsDistributor(yearless_date_ranges=[
            (YearlessDatetime(12, 20),
             YearlessDatetime(2, 10, 0)),  # 12 + 9 elements
            (YearlessDatetime(2, 27), YearlessDatetime(2, 29, 12))  # 3 or 2
        ])
        d = set4.select(self.df)
        ny = len(np.unique(self.df.index.year))
        assert datetime(2007, 12, 21, 12) in d.index
        assert datetime(2004, 2, 29, 12) in d.index
        assert len(d.index) == (12 * ny) + (9 * (ny - 1)) + (3 * ny - 7)

    def test_filter_all_in_one(self):
        dates = (
            datetime(2005, 4, 4, 12),
            datetime(2005, 5, 5, 12),
        )
        date_ranges = [(datetime(2004, 4, 6), datetime(2004, 4, 8, 12))]
        yearless_dates = [YearlessDatetime(4, 10, 12)]
        yearless_date_ranges = [(YearlessDatetime(2, 27),
                                 YearlessDatetime(2, 29, 23))]

        set = TsDistributor(
            dates=dates,
            yearless_dates=yearless_dates,
            date_ranges=date_ranges,
            yearless_date_ranges=yearless_date_ranges,
        )
        d = set.select(self.df)

        assert np.all(dt in d.index for dt in dates)
        for dt in [
            datetime(2004, 4, 6, 12),
            datetime(2004, 4, 7, 12),
            datetime(2004, 4, 8, 12)
        ]:
            assert dt in d.index

        assert datetime(2008, 4, 10, 12) in d.index

        for dt in [
            datetime(2007, 2, 27, 12),
            datetime(2007, 2, 28, 12),
            datetime(2008, 2, 29, 12)
        ]:
            assert dt in d.index
