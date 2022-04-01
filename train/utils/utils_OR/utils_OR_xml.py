import xml.etree.ElementTree as et
import numpy as np
from pathlib import Path
from utils_OR.utils_OR_mesh import loadMesh, computeBox, computeTransform, writeMesh
import copy
import random, string

def get_XML_root(main_xml_file):
    tree = et.parse(str(main_xml_file)) # L202 of sampleCameraPoseFromScanNet.py
    root  = tree.getroot()
    return root

def parse_XML_for_intrinsics(root):
    sensors = root.findall('sensor')
    assert len(sensors)==1
    sensor = sensors[0]

    film = sensor.findall('film')[0]
    integers = film.findall('integer')
    for integer in integers:
        if integer.get('name' ) == 'width':
            width = int(integer.get('value'))
        if integer.get('name' ) == 'height':
            height = int(integer.get('value'))
    fov_entry = sensor.findall('float')[0]
    assert fov_entry.get('name') == 'fov'
    fov = float(fov_entry.get('value'))
    f_px = width / 2. / np.tan(fov / 180. * np.pi / 2.)
    cam_K = np.array([[-f_px, 0., width/2.], [0., -f_px, height/2.], [0., 0., 1.]])
    return cam_K, {'fov': fov, 'f_px': f_px, 'width': width, 'height': height}

def parse_XML_for_shapes(root, root_uv_mapped):
    shapes = root.findall('shape')
    shapes_list = []

    for shape in shapes:
        shape_dict = {'id': shape.get('id'), 'id_random': ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))}

        shape_dict[shape.findall('string')[0].get('name')] = shape.findall('string')[0].get('value')
        
        assert len(shape.findall('transform')) == 1
        transforms = shape.findall('transform')[0]
        assert transforms.get('name') == 'toWorld'
        transforms_list = []
        for transform in transforms:
            transform_name = transform.tag
            assert transform_name in ['scale', 'rotate', 'translate']
            transform_dict = {transform_name: {key: float(transform.get(key)) for key in transform.keys()}}
            transforms_list.append(transform_dict)
        shape_dict['transforms_list'] = transforms_list
        shape_dict['if_correct_path'] = False

        shapes_list.append(shape_dict)

    return shapes_list