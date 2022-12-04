import os, sys
import cv2
import numpy as np
import pickle
import time
import yaml
import trimesh

import torch

import xml.etree.ElementTree as ET

from sklearn.decomposition import IncrementalPCA

from READ.gl.programs import NNScene


class TicToc:
    def __init__(self):
        self.tic_toc_tic = None

    def tic(self):
        self.tic_toc_tic = time.time()

    def toc(self):
        assert self.tic_toc_tic, 'You forgot to call tic()'
        return (time.time() - self.tic_toc_tic) * 1000

    def tocp(self, str):
        print(f"{str} took {self.toc():.4f}ms")

    @staticmethod
    def print_timing(timing, name=''):
        print(f'\n=== {name} Timimg ===')
        for fn, times in timing.items():
            min, max, mean, p95 = np.min(times), np.max(times), np.mean(times), np.percentile(times, 95)
            print(f'{fn}:\tmin: {min:.4f}\tmax: {max:.4f}\tmean: {mean:.4f}ms\tp95: {p95:.4f}ms')


class FastRand:
    def __init__(self, shape, tform, bank_size):
        bank = []
        for i in range(bank_size):
            p = np.random.rand(*shape)
            p = tform(p)
            bank.append(p)

        self.bank = bank

    def toss(self):
        i = np.random.randint(0, len(self.bank))
        return self.bank[i]


def cv2_write(fn, x):
    x = np.clip(x, 0, 1) * 255
    x = x[..., :3][..., ::-1]
    cv2.imwrite(fn, x.astype(np.uint8))


def to_numpy(x, float16=False, flipv=True):
    if not isinstance(x, np.ndarray):
        x = x.detach().cpu().numpy()

    if float16:
        x = x.astype(np.float16)

    if flipv:
        x = x[::-1].copy()

    return x


def pca_color(tex, save='', load=''):
    tex = tex[0].transpose(1, 0)
    if load and os.path.exists(load):
        print('loading...')
        with open(load,'rb') as f:
            pca=pickle.load(f)
        print('applying...')
        res=pca.transform(tex)
    else:
        pca=IncrementalPCA(n_components=3, batch_size=64)
        print('applying...')
        res=pca.fit_transform(tex)
    if save and save != load:
        with open(save,'wb') as f:
            pickle.dump(pca,f)
    # pca_color_n=(pca_color - pca_color.min()) / (pca_color.max() - pca_color.min())
    # return res.transpose(1, 0)[None]
    return res


def crop_proj_matrix(pm, old_w, old_h, new_w, new_h):
    # NOTE: this is not precise
    old_cx = old_w / 2
    old_cy = old_h / 2
    new_cx = new_w / 2
    new_cy = new_h / 2

    pm_new = pm.copy()
    pm_new[0,0] = pm[0,0]*old_w/new_w
    pm_new[0,2] = (pm[0,2]-1)*old_w*new_cx/old_cx/new_w + 1
    pm_new[1,1] = pm[1,1]*old_h/new_h
    pm_new[1,2] = (pm[0,2]+1)*old_h*new_cy/old_cy/new_h - 1
    return pm_new


def recalc_proj_matrix_planes(pm, new_near=.01, new_far=1000.):
    depth = float(new_far - new_near)
    q = -(new_far + new_near) / depth
    qn = -2 * (new_far * new_near) / depth

    out = pm.copy()

    # Override near and far planes
    out[2, 2] = q
    out[2, 3] = qn

    return out


def get_proj_matrix(K, image_size, znear=.01, zfar=1000.):
    fx = K[0,0]
    fy = K[1,1]
    cx = K[0,2]
    cy = K[1,2]
    width, height = image_size
    m = np.zeros((4, 4))
    m[0][0] = 2.0 * fx / width
    m[0][1] = 0.0
    m[0][2] = 0.0
    m[0][3] = 0.0

    m[1][0] = 0.0
    m[1][1] = 2.0 * fy / height
    m[1][2] = 0.0
    m[1][3] = 0.0

    m[2][0] = 1.0 - 2.0 * cx / width
    m[2][1] = 2.0 * cy / height - 1.0
    m[2][2] = (zfar + znear) / (znear - zfar)
    m[2][3] = -1.0

    m[3][0] = 0.0
    m[3][1] = 0.0
    m[3][2] = 2.0 * zfar * znear / (znear - zfar)
    m[3][3] = 0.0

    return m.T


def rescale_K(K_, sx, sy, keep_fov=True):
    K = K_.copy()
    K[0, 2] = sx * K[0, 2]
    K[1, 2] = sy * K[1, 2]
    if keep_fov:
        K[0, 0] = sx * K[0, 0]
        K[1, 1] = sy * K[1, 1]
    return K


def crop_intrinsic_matrix(K, old_size, new_size):
    K = K.copy()
    K[0, 2] = new_size[0] * K[0, 2] / old_size[0]
    K[1, 2] = new_size[1] * K[1, 2] / old_size[1]
    return K


def intrinsics_from_xml(xml_file):
    root = ET.parse(xml_file).getroot()
    calibration = root.find('chunk/sensors/sensor/calibration')
    resolution = calibration.find('resolution')
    width = float(resolution.get('width'))
    height = float(resolution.get('height'))
    f = float(calibration.find('f').text)
    cx = width/2
    cy = height/2

    K = np.array([
        [f, 0, cx],
        [0, f, cy],
        [0, 0,  1]
        ], dtype=np.float32)

    return K, (width, height)


def extrinsics_from_xml(xml_file, verbose=False):
    root = ET.parse(xml_file).getroot()
    transforms = {}
    for e in root.findall('chunk/cameras')[0].findall('camera'):
        label = e.get('label')
        try:
            transforms[label] = e.find('transform').text
        except:
            if verbose:
                print('failed to align camera', label)

    view_matrices = []
    # labels_sort = sorted(list(transforms), key=lambda x: int(x))
    labels_sort = list(transforms)
    for label in labels_sort:
        extrinsic = np.array([float(x) for x in transforms[label].split()]).reshape(4, 4)
        extrinsic[:, 1:3] *= -1
        view_matrices.append(extrinsic)

    return view_matrices, labels_sort


def extrinsics_from_view_matrix(path):
    vm = np.loadtxt(path).reshape(-1,4,4)
    vm, ids = get_valid_matrices(vm)

    # we want consistent camera label data type, as cameras from xml
    ids = [str(i) for i in ids]

    return vm, ids



def setup_scene(scene, data, use_mesh=False, use_texture=False):
    
    if data['mesh'] is not None and data['pointcloud'] is None or use_mesh or use_texture:
        assert 'mesh' in data, 'use pointcloud or set mesh'
        model3d = data['mesh']
    else:
        assert 'pointcloud' in data, 'use mesh or set pointcloud'
        model3d = data['pointcloud']
        print("model3d",model3d)
    scene.set_vertices(
        positions=model3d['xyz'],
        colors=model3d['rgb'],
        normals=model3d['normals'],
        uv1d=model3d['uv1d'],
        uv2d=model3d['uv2d'],
        texture=data['texture'])

    if data['proj_matrix'] is not None:
        scene.set_proj_matrix(data['proj_matrix'])
    else:
        print('proj_matrix was not set')

    if data['view_matrix'] is not None or len(data['view_matrix']) > 0:
        scene.set_camera_view(data['view_matrix'][0])
    else:
        print('view_matrix was not set')

    scene.set_model_view(data['model3d_origin'])
    scene.set_indices(model3d['faces'])

    if data['point_sizes'] is not None:
        scene.set_point_sizes(data['point_sizes'])

    scene.set_use_texture(use_texture)


def load_scene_data(path):
    with open(path, 'r') as f:
        config = yaml.load(f,Loader=yaml.FullLoader)

    if 'pointcloud' in config:
        print('loading pointcloud...')
        pointcloud = import_model3d(fix_relative_path(config['pointcloud'], path))
    else:
        pointcloud = None

    if 'mesh' in config and config['mesh']:
        print('loading mesh...')
        uv_order = config['uv_order'] if 'uv_order' in config else 's,t'
        mesh = import_model3d(fix_relative_path(config['mesh'], path), uv_order=uv_order.split(','), is_mesh=True)
    else:
        mesh = None

    if config.get('texture'):
        print('loading texture...')
        texture = cv2.imread(fix_relative_path(config['texture'], path))
        assert texture is not None
        texture = texture[..., ::-1].copy()
    else:
        texture = None

    if 'intrinsic_matrix' in config:
        apath = fix_relative_path(config['intrinsic_matrix'], path)
        if apath[-3:] == 'xml':
            intrinsic_matrix, (width, height) = intrinsics_from_xml(apath)
            assert tuple(config['viewport_size']) == (width, height), f'calibration width, height: ({width}, {height})'
        else:
            intrinsic_matrix = np.loadtxt(apath)[:3, :3]
    else:
        intrinsic_matrix = None

    if 'proj_matrix' in config:
        proj_matrix = np.loadtxt(fix_relative_path(config['proj_matrix'], path))
        proj_matrix = recalc_proj_matrix_planes(proj_matrix)
    else:
        proj_matrix = None

    if 'view_matrix' in config:
        apath = fix_relative_path(config['view_matrix'], path)
        if apath[-3:] == 'xml':
            view_matrix, camera_labels = extrinsics_from_xml(apath)
        else:
            view_matrix, camera_labels = extrinsics_from_view_matrix(apath)
    else:
        view_matrix = None
    # print(camera_labels)

    if 'model3d_origin' in config:
        model3d_origin = np.loadtxt(fix_relative_path(config['model3d_origin'], path))
    else:
        model3d_origin = np.eye(4)

    if 'point_sizes' in config:
        point_sizes = np.load(fix_relative_path(config['point_sizes'], path))
    else:
        point_sizes = None

    config['viewport_size'] = tuple(config['viewport_size'])

    # if 'use_mesh' in config:
    #     use_mesh = config['use_mesh']
    # elif pointcloud is None and mesh is not None:
    #     use_mesh = True
    # else:
    #     use_mesh = False

    if 'net_path' in config:
        net_ckpt = os.path.join(config['net_path'], 'checkpoints', config['ckpt'])
        net_ckpt = fix_relative_path(net_ckpt, path)

        tex_ckpt = os.path.join(config['net_path'], 'checkpoints', config['texture_ckpt'])
        tex_ckpt = fix_relative_path(tex_ckpt, path)
    else:
        net_ckpt = None
        tex_ckpt = None

    return {
    'pointcloud': pointcloud,
    'point_sizes': point_sizes,
    'mesh': mesh,
    # 'use_mesh': use_mesh,
    'texture': texture,
    'proj_matrix': proj_matrix,
    'intrinsic_matrix': intrinsic_matrix,
    'view_matrix': view_matrix,
    'camera_labels': camera_labels,
    'model3d_origin': model3d_origin,
    'config': config,

    'net_ckpt': net_ckpt,
    'tex_ckpt': tex_ckpt
    }


def load_scene(config_path):
    scene_data = load_scene_data(config_path)

    scene = NNScene()
    setup_scene(scene, scene_data)

    return scene, scene_data


def fix_relative_path(path, config_path):
    if not os.path.exists(path) and not os.path.isabs(path):
        root = os.path.dirname(config_path)
        abspath = os.path.join(root, path)
        if os.path.exists(abspath):
            return abspath
    return path


def get_valid_matrices(mlist):
    ilist = []
    vmlist = []
    for i, m in enumerate(mlist):
        if np.isfinite(m).all():
            ilist.append(i)
            vmlist.append(m)

    return vmlist, ilist


def get_xyz_colors(xyz, r=8):
    mmin, mmax = xyz.min(axis=0), xyz.max(axis=0)
    color = (xyz - mmin) / (mmax - mmin)
    # color = 0.5 + 0.5 * xyz / r
    return np.clip(color, 0., 1.).astype(np.float32)

def get_normal_colors(normals):
    # [-1,+1]->[0,1]
    return (normals * 0.5 + 0.5).astype(np.float32)


def import_model3d(model_path, uv_order=None, is_mesh=False):
    data = trimesh.load(model_path)

    n_pts = data.vertices.shape[0]

    model = {
        'rgb': None,
        'normals': None,
        'uv2d': None,
        'faces': None
    }

    if is_mesh:
        if hasattr(data.visual, 'vertex_colors'):
            model['rgb'] = data.visual.vertex_colors[:, :3] / 255.
        elif hasattr(data.visual, 'to_color'):
            try:
                # for some reason, it may fail (happens on blender exports)
                model['rgb'] = data.visual.to_color().vertex_colors[:, :3] / 255.
            except:
                print('data.visual.to_color failed')

        model['normals'] = data.vertex_normals

        if hasattr(data.visual, 'uv'):
            model['uv2d'] = data.visual.uv
        # elif model_path[-3:] == 'ply':
        #     mdata = data.metadata['ply_raw']['vertex']['data']
        #     if 's' in mdata and 't' in mdata:
        #         print('using s,t texture coords')
        #         model['uv2d'] = np.hstack([mdata['s'], mdata['t']])
        #         print(model['uv2d'].shape)

        model['faces'] = data.faces.flatten().astype(np.uint32)
    else:
        if hasattr(data, 'colors'):
            model['rgb'] = data.colors[:, :3] / 255.
        else:
            try:
                model['rgb'] = data.visual.vertex_colors[:, :3] / 255.
            except:
                pass

        if 'ply_raw' in data.metadata:
            normals = np.zeros((n_pts, 3), dtype=np.float32)
            normals[:, 0] = data.metadata['ply_raw']['vertex']['data']['nx']
            normals[:, 1] = data.metadata['ply_raw']['vertex']['data']['ny']
            normals[:, 2] = data.metadata['ply_raw']['vertex']['data']['nz']
            model['normals'] = normals
        elif hasattr(data, 'vertex_normals'):
            model['normals'] = data.vertex_normals

        model['uv2d'] = np.zeros((n_pts, 2), dtype=np.float32)

    model['xyz'] = data.vertices
    model['xyz_c'] = get_xyz_colors(data.vertices)
    model['uv1d'] = np.arange(n_pts)

    if model['rgb'] is None:
        print(f'no colors in {model_path}')
        model['rgb'] = np.zeros((n_pts, 3), dtype=np.float32)

    if model['normals'] is None:
        print(f'no normals in {model_path}')
        model['rgb'] = np.zeros((n_pts, 3), dtype=np.float32)

    if model['uv2d'] is None:
        if is_mesh:
            print(f'no uv in {model_path}')
        model['uv2d'] = np.zeros((n_pts, 2), dtype=np.float32)

    if model['faces'] is None:
        if is_mesh:
            print(f'no faces in {model_path}')
        model['faces'] = np.array([0, 1, 2], dtype=np.uint32)

    print('=== 3D model ===')
    print('VERTICES: ', n_pts)
    print('EXTENT: ', model['xyz'].min(0), model['xyz'].max(0))
    print('================')

    return model


def get_vec(view_mat):
    view_mat = view_mat.copy()
    rvec0 = cv2.Rodrigues(view_mat[:3, :3])[0].flatten()
    t0 = view_mat[:3, 3]
    return rvec0, t0


def nearest_train(view_mat, test_pose, p=0.05):
    dists = []
    angs = []
    test_rvec, test_t = get_vec(test_pose)
    for i in range(len(view_mat)):
        rvec, t = get_vec(view_mat[i])
        dists.append(
            np.linalg.norm(test_t - t)
        )
        angs.append(
            np.linalg.norm(test_rvec - rvec)
        )
    angs_sort = np.argsort(angs)
    angs_sort = angs_sort[:int(len(angs_sort) * p)]
    dists_pick = [dists[i] for i in angs_sort]
    ang_dist_i = angs_sort[np.argmin(dists_pick)]
    return ang_dist_i #, angs_sort[0]

