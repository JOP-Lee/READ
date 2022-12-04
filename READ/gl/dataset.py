import re

from READ.gl.programs import NNScene


def parse_input_string_obsolete(what):
    config = {}
    if '_pr' in what:
        config['draw_points'] = True
        config['splat_mode'] = True
        config['point_size'] = int(what.split('_pr')[-1])
        config['flat_color'] = True
    elif '_p' in what:
        config['draw_points'] = True
        config['splat_mode'] = False
        config['point_size'] = int(what.split('_p')[-1])
        config['flat_color'] = True
    else:
        config['draw_points'] = False
        config['splat_mode'] = False
        config['point_size'] = 1
        config['flat_color'] = False

    if 'colors' in what:
        config['mode'] = NNScene.MODE_COLOR, 0
    elif 'uv' in what:
        m1 = NNScene.UV_TYPE_1D if config['draw_points'] else NNScene.UV_TYPE_2D
        config['mode'] = NNScene.MODE_UV, m1
    elif 'normals' in what:
        nm = what.split('_')[-2] if '_p' in what else what.split('_')[-1]
        m1 = ['g', 'r', 'l', 'd'].index(nm)
        config['mode'] = NNScene.MODE_NORMALS, m1
    elif 'xyz' in what:
        config['mode'] = NNScene.MODE_XYZ, 0

    return config


def parse_input_string(string):
    config = {}
    
    if re.search('^colors', string):
        config['mode'] = NNScene.MODE_COLOR, None
    elif re.search('^uv', string):
        choices = ['uv_1d', 'uv_2d']
        ch = re.findall('|'.join(choices), string)[-1]
        m1 = choices.index(ch)
        config['mode'] = NNScene.MODE_UV, m1
    elif re.search('^normals', string):
        choices = ['normals_m', 'normals_r', 'normals_l', 'normals_d']
        ch = re.findall('|'.join(choices), string)[-1]
        m1 = choices.index(ch)
        config['mode'] = NNScene.MODE_NORMALS, m1
    elif re.search('^xyz', string):
        config['mode'] = NNScene.MODE_XYZ, None
    elif re.search('^depth', string):
        config['mode'] = NNScene.MODE_DEPTH, None
    elif re.search('^labels', string):
        config['mode'] = NNScene.MODE_LABEL, None
    else:
        raise ValueError(string)
        
    res = re.findall('ps[0-9]+|p[0-9]+', string)
    if res:
        res = res[-1]
        config['draw_points'] = True
        config['flat_color'] = True
        config['point_size'] = int(re.search('[0-9]+', res).group())
        config['splat_mode'] = re.search('^ps', res) is not None
    else:
        config['draw_points'] = False
        config['splat_mode'] = False
        config['point_size'] = 1
        config['flat_color'] = False
    
    res = re.findall('ds[0-5]+', string)
    if res:
        res = res[-1]
        config['downscale'] = int(re.search('[0-9]+', res).group())
            

    return config


def generate_input_string(config):
    s = ''
    m0, m1 = config['mode']
    if m0 == NNScene.MODE_COLOR:
        s += 'colors'
    elif m0 == NNScene.MODE_UV:
        s += 'uv'
        if m1 == NNScene.UV_TYPE_1D:
            s += '_1d'
        elif m1 == NNScene.UV_TYPE_2D:
            s += '_2d'
        else:
            raise ValueError
    elif m0 == NNScene.MODE_NORMALS:
        s += 'normals'
        if m1 == NNScene.NORMALS_MODE_MODEL:
            s += '_m'
        elif m1 == NNScene.NORMALS_MODE_REFLECTION:
            s += '_r'
        elif m1 == NNScene.NORMALS_MODE_LOCAL:
            s += '_l'
        elif m1 == NNScene.NORMALS_MODE_DIRECTION:
            s += '_d'
    elif m0 == NNScene.MODE_XYZ:
        s += 'xyz'
    elif m0 == NNScene.MODE_DEPTH:
        s += 'depth'

    if config['draw_points']:
        s += '_p'
        if config['splat_mode']:
            s += 's'
        s += str(config['point_size'])

    if 'downscale' in config:
        s += f"_ds{config['downscale']}"

    return s



def test_generate_parse():
    configs = [
        {
        'draw_points': True,
        'splat_mode': True,
        'point_size': 20,
        'flat_color': True,

        'mode': (NNScene.MODE_UV, NNScene.UV_TYPE_1D)
        },

        {
        'draw_points': True,
        'splat_mode': True,
        'point_size': 20,
        'flat_color': True,

        'mode': (NNScene.MODE_UV, NNScene.UV_TYPE_2D)
        },

        {
        'draw_points': True,
        'splat_mode': False,
        'point_size': 20,
        'flat_color': True,

        'mode': (NNScene.MODE_COLOR, None)
        },

        {
        'draw_points': False,
        'splat_mode': False,
        'point_size': 1,
        'flat_color': False,

        'mode': (NNScene.MODE_NORMALS, NNScene.NORMALS_MODE_REFLECTION)
        },

        {
        'draw_points': False,
        'splat_mode': False,
        'point_size': 1,
        'flat_color': False,

        'mode': (NNScene.MODE_XYZ, None)
        },

        {
        'draw_points': True,
        'splat_mode': False,
        'point_size': 1,
        'flat_color': True,

        'mode': (NNScene.MODE_XYZ, None),

        'downscale': 2
        },
        ]

    def check(config):
        print('config:\n', config)
        s = generate_input_string(config)
        print('string:\n', s)
        c = parse_input_string(s)
        print('generated config:\n', c)
        print('-- OK' if config == c else '-- FAIL')

    for c in configs:
        check(c)


if __name__ == '__main__':
    test_generate_parse()