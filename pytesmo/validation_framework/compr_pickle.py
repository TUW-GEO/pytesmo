import cPickle
import gzip


def pickle(file_name, obj):
    """
    Saveing data compressed to a file. Code used from
    http://stackoverflow.com/questions/695794/
           more-efficient-way-to-pickle-a-string/

    Parameters
    ----------
    file_name : str
        Filename with path.
    obj : any
        Data to store in file.
    """
    cPickle.dump(obj=obj,
                 file=gzip.open(file_name, "wb", compresslevel=3), protocol=2)


def unpickle(file_name):
    """
    Restoring data from a compressed pickled file.

    Parameters
    ----------
    file_name : str
        Filename with path.

    Returns
    -------
    obj : any
        Stored data in  a file.
    """
    return cPickle.load(gzip.open(file_name, "rb"))
