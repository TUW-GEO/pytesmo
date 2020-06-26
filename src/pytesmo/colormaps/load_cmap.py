import matplotlib.cm as cm
import matplotlib.colors as colors
import os
import json
import glob


def colormaps_path():
    """Returns application's default path for storing user-defined colormaps"""
    return os.path.dirname(__file__)


def get_system_colormaps():
    """Returns the list of colormaps that ship with matplotlib"""
    return [m for m in cm.datad]


def get_user_colormaps(cmap_fldr=colormaps_path()):
    """Returns a list of user-defined colormaps in the specified folder (defaults to
    standard colormaps folder if not specified)."""
    user_colormaps = []
    for root, dirs, files in os.walk(cmap_fldr):
        files = glob.glob(root + '/*.cmap')
        for name in files:
            with open(os.path.join(root, name), "r") as fidin:
                cmap_dict = json.load(fidin)
                user_colormaps.append(cmap_dict.get('name', name))
    return user_colormaps


def load_colormap(json_file):
    """Generates and returns a matplotlib colormap from the specified JSON file,
    or None if the file was invalid."""
    colormap = None
    with open(json_file, "r") as fidin:
        cmap_dict = json.load(fidin)
        if cmap_dict.get('colors', None) is None:
            return colormap
        colormap_type = cmap_dict.get('type', 'linear')
        colormap_name = cmap_dict.get('name', os.path.basename(json_file))
        if colormap_type == 'linear':
            colormap = colors.LinearSegmentedColormap.from_list(name=colormap_name,
                                                                colors=cmap_dict['colors'])
        elif colormap_type == 'list':
            colormap = colors.ListedColormap(name=colormap_name, colors=cmap_dict['colors'])
    return colormap


def load(cmap_name, cmap_folder=colormaps_path()):
    """Returns the matplotlib colormap of the specified name -
    if not found in the predefined
    colormaps, searches for the colormap in the specified
    folder (defaults to standard colormaps
    folder if not specified)."""
    cmap_name_user = cmap_name + '.cmap'
    user_colormaps = get_user_colormaps(cmap_folder)
    system_colormaps = get_system_colormaps()

    if cmap_name_user in user_colormaps:
        cmap_file = os.path.join(cmap_folder, cmap_name_user)
        cmap = load_colormap(cmap_file)
    elif cmap_name in system_colormaps:
        return cm.get_cmap(cmap_name)
    else:
        raise ValueError('Colormap not found')
    return cmap
