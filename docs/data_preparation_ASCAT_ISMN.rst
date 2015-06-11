
Example data preparation ASCAT - ISMN
=====================================

The validation framework allows users to prepare their data before the
validation is performed by using the **DataPreparation** class which
must contain two methods: \* ***prep\_reference***, has at least one
parameter - the reference dataframe \* ***prep\_other***, has at least
two parameters - the other dataframe and the name of the dataset

.. code:: python

    class DataPreparation(object):
        """
        Class for preparing the data before validation.
        """
        @staticmethod
        def prep_reference(reference):
            """
            Static method used to prepare the reference dataset (ISMN).
    
            Parameters
            ----------
            reference : pandas.DataFrame
                ISMN data.
    
            Returns
            -------
            reference : pandas.DataFrame
                Masked reference.
            """
            return reference
    
        @staticmethod
        def prep_other(other, other_name, mask_snow=80, mask_frozen=80, mask_ssf=[0, 1]):
            """
            Static method used to prepare the other datasets (ASCAT).
    
            Parameters
            ----------
            other : pandas.DataFrame
                Containing at least the fields: sm, frozen_prob, snow_prob, ssf.
            other_name : string
                ASCAT.
            mask_snow : int, optional
                If set, all the observations with snow probability > mask_snow
                are removed from the result. Default: 80.
            mask_frozen : int, optional
                If set, all the observations with frozen probability > mask_frozen
                are removed from the result. Default: 80.
            mask_ssf : list, optional
                If set, all the observations with ssf != mask_ssf are removed from
                the result. Default: [0, 1].
    
            Returns
            -------
            reference : pandas.DataFrame
                Masked reference.
            """
            if other_name == 'ASCAT':
    
                # mask frozen
                if mask_frozen is not None:
                    other = other[other['frozen_prob'] < mask_frozen]
    
                # mask snow
                if mask_snow is not None:
                    other = other[other['snow_prob'] < mask_snow]
    
                # mask ssf
                if mask_ssf is not None:
                    other = other[(other['ssf'] == mask_ssf[0]) |
                                  (other['ssf'] == mask_ssf[1])]
            return other
