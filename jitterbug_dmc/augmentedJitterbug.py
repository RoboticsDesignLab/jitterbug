import xml.etree.ElementTree as ET
import numpy as np
from dm_control import suite
from dm_control import viewer
# Open original Jitterbug file
tree = ET.parse('jitterbug.xml')

root = tree.getroot()

def str2array(string):
    """Convert the string containing coordinates into an array.

     Args:
         string (str): string containing coordinates to be converted"""
    arr_str = string.split(" ")
    N = len(arr_str)
    arr_float = np.zeros(N)
    for i in range(N):
        arr_float[i] = float(arr_str[i])

    return arr_float

def array2str(arr):
    """Convert the array containing coordinates into a string.

    Args:
        arr (np array): array containing the coordinates to be converted"""
    arr_str = []
    for el in arr:
        arr_str.append(str(el))

    return " ".join(arr_str)



for bd in root.iter('body'):
    if bd.attrib['name'] in [f"leg{i}lower" for i in range(1,5)]:
        print(bd.attrib['name'])
        dx = np.random.normal(0,0.01)
        dy = np.random.normal(0,0.01)
        for geom in bd.iter('geom'):
            if geom.attrib['type']=="cylinder":
                fromto_str = geom.attrib['fromto']
                fromto_arr = str2array(fromto_str)
                fromto_arr[0] += dx
                fromto_arr[1] += dy
                geom.attrib['fromto'] = array2str(fromto_arr)
            elif geom.attrib['type']=='sphere':
                pos_str = geom.attrib['pos']
                pos_arr = str2array(pos_str)
                pos_arr[0] += dx
                pos_arr[1] += dy
                geom.attrib['pos'] = array2str(pos_arr)


# Save augmented Jitterbug in a new XML file
tree.write('jitterbug2.xml')

env = suite.load(
        domain_name="jitterbug2",
        task_name="move_from_origin",
        visualize_reward=True,
        task_kwargs=dict(
            #time_limit=float("inf")
            norm_obs=True
        )
    )