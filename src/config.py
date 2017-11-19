from os import path

path_src = path.dirname(path.abspath(__file__))
path_root = path.dirname(path_src)
path_data = path.join(path_root, 'data')
path_model = path.join(path_root, 'model')
path_cache = path.join(path_root, 'cache')
path_output = path.join(path_root, 'output')
path_aps = path.join(path_data, 'aps')
