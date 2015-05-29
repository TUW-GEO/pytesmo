import itertools


class DataManager(object):

    def __init__(self, datasets, data_prep=None, period=None):

        self.datasets = datasets

        self.other_name = []
        for dataset in datasets.keys():
            if datasets[dataset]['type'] == 'reference':
                self.reference_name = dataset
            else:
                self.other_name.append(dataset)

        self.reference_grid = self.datasets[self.reference_name]['class'].grid

        self.data_prep = data_prep
        self.period = period

    def use_lut(self, other_name):

        if self.datasets[other_name]['use_lut']:
            return self.reference_grid.calc_lut(
                self.datasets[other_name]['class'].grid,
                max_dist=self.datasets[other_name]['lut_max_dist'])
        else:
            return None

    def get_result_names(self):

        result_names = []

        ref_columns = []
        for column in self.datasets[self.reference_name]['columns']:
            ref_columns.append(self.reference_name + '.' + column)

        other_columns = []
        for other in self.other_name:
            for column in self.datasets[other]['columns']:
                other_columns.append(other + '.' + column)

        for comb in itertools.product(ref_columns, other_columns):
            result_names.append(comb)

        return result_names

    def read_reference(self, *args):
        """
        Function to read the reference dataset
        """
        reference = self.datasets[self.reference_name]
        args = list(args)
        args.extend(reference['args'])

        try:
            ref_df = reference['class'].read_ts(*args, **reference['kwargs'])
        except IOError:
            return None

        if self.period is not None:
            ref_df = ref_df[self.period[0]:self.period[1]]

        if len(ref_df) == 0:
            return None

        if self.data_prep is not None:
            ref_df = self.data_prep.prep_reference(ref_df)

        if len(ref_df) == 0:
            return None
        else:
            return ref_df

    def read_other(self, other_name, *args):
        """
        Function to read other dataset
        """
        other = self.datasets[other_name]
        args = list(args)
        args.extend(other['args'])

        try:
            other_df = other['class'].read_ts(*args, **other['kwargs'])
        except IOError:
            return None

        if self.period is not None:
            other_df = other_df[self.period[0]:self.period[1]]

        if len(other_df) == 0:
            return None

        if self.data_prep is not None:
            other_df = self.data_prep.prep_other(other_df, other_name)

        if len(other_df) == 0:
            return None
        else:
            return other_df
