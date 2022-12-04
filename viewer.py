import argparse
import threading
import yaml
import re

from glumpy import app, gloo, glm, gl, transforms
from glumpy.ext import glfw

from READ.gl.render import OffscreenRender, create_shared_texture, cpy_tensor_to_buffer, cpy_tensor_to_texture
from READ.gl.programs import NNScene
from READ.gl.utils import load_scene_data, get_proj_matrix, crop_intrinsic_matrix, crop_proj_matrix, \
    setup_scene, rescale_K, FastRand, nearest_train, pca_color, extrinsics_from_view_matrix, extrinsics_from_xml
from READ.gl.nn import OGL
from READ.gl.camera import Trackball,linePlaneCollision,dot_product,norm,project_onto_plane,normalize


import os, sys
import time
import numpy as np
import torch
import cv2
import math
import quaternion

import trimesh.transformations as transformations


def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-c', '--config', type=str, default=None, required=True, help='config path')
    parser.add_argument('--viewport', type=str, default='', help='width,height')
    parser.add_argument('--keep-fov', action='store_true', help='keep field of view when resizing viewport')
    parser.add_argument('--init-view', type=str, help='camera label for initial view or path to 4x4 matrix')
    parser.add_argument('--use-mesh', action='store_true')
    parser.add_argument('--use-texture', action='store_true')
    parser.add_argument('--rmode', choices=['trackball', 'fly'], default='trackball')
    parser.add_argument('--fps', action='store_true', help='show fps')
    parser.add_argument('--light-position', type=str, default='', help='x,y,z')
    #parser.add_argument('--replay-camera', type=str, default='Data/kitti6/camera.xml', help='path to view_matrix to replay at given fps')
    parser.add_argument('--replay-camera', type=str, default='', help='path to view_matrix to replay at given fps')
    parser.add_argument('--replay-fps', type=float, default=25., help='view_matrix replay fps')
    parser.add_argument('--supersampling', type=int, default=1, choices=[1, 2])
    parser.add_argument('--clear-color', type=str)
    parser.add_argument('--nearest-train', action='store_true')
    parser.add_argument('--gt', help='like /path/to/images/*.JPG. * will be replaced with nearest camera label.')
    parser.add_argument('--pca', action='store_true')
    parser.add_argument('--origin-view', action='store_true')
    parser.add_argument('--temp-avg', action='store_true')
    parser.add_argument('--checkpoint')
    args = parser.parse_args()

    args.viewport = tuple([int(x) for x in args.viewport.split(',')]) if args.viewport else None
    args.light_position = [float(x) for x in args.light_position.split(',')] if args.light_position else None
    args.clear_color = [float(x) for x in args.clear_color.split(',')] if args.clear_color else None

    return args


def get_screen_program(texture):
    vertex = '''
    attribute vec2 position;
    attribute vec2 texcoord;
    varying vec2 v_texcoord;
    void main()
    {
        gl_Position = <transform>;
        v_texcoord = texcoord;
    } '''
    fragment = '''
    uniform sampler2D texture;
    varying vec2 v_texcoord;
    void main()
    {
        gl_FragColor = texture2D(texture, v_texcoord);
    } '''

    quad = gloo.Program(vertex, fragment, count=4)
    quad["transform"] = transforms.OrthographicProjection(transforms.Position("position"))
    quad['texcoord'] = [( 0, 0), ( 0, 1), ( 1, 0), ( 1, 1)]
    quad['texture'] = texture

    return quad


def start_fps_job():
    def job():
        print(f'FPS {app.clock.get_fps():.1f}')

    threading.Timer(1.0, job).start()


def load_camera_trajectory(path):
    if path[-3:] == 'xml':
        view_matrix, camera_labels = extrinsics_from_xml(path)
    else:
        view_matrix, camera_labels = extrinsics_from_view_matrix(path)
    return view_matrix


def fix_viewport_size(viewport_size, factor=16):
    viewport_w = factor * (viewport_size[0] // factor)
    viewport_h = factor * (viewport_size[1] // factor)
    return viewport_w, viewport_h


class MyApp():
    def __init__(self, args):
        with open(args.config) as f:
            _config = yaml.load(f,Loader=yaml.FullLoader)
            # support two types of configs
            # 1 type - config with scene data
            # 2 type - config with model checkpoints and path to scene data config
            if 'scene' in _config: # 1 type
                self.scene_data = load_scene_data(_config['scene'])
                net_ckpt = _config.get('net_ckpt')
                texture_ckpt = _config.get('texture_ckpt') 
            else:
                self.scene_data = load_scene_data(args.config)
                net_ckpt = self.scene_data['config'].get('net_ckpt')
                texture_ckpt = self.scene_data['config'].get('texture_ckpt')

        self.viewport_size = args.viewport if args.viewport else self.scene_data['config']['viewport_size']
        self.viewport_size = fix_viewport_size(self.viewport_size)
        print('new viewport size ', self.viewport_size)

        # crop/resize viewport
        if self.scene_data['intrinsic_matrix'] is not None:
            K_src = self.scene_data['intrinsic_matrix']
            old_size = self.scene_data['config']['viewport_size']
            sx = self.viewport_size[0] / old_size[0]
            sy = self.viewport_size[1] / old_size[1]
            K_crop = rescale_K(K_src, sx, sy, keep_fov=args.keep_fov)
            self.scene_data['proj_matrix'] = get_proj_matrix(K_crop, self.viewport_size)
        elif self.scene_data['proj_matrix'] is not None:
            new_proj_matrix = crop_proj_matrix(self.scene_data['proj_matrix'], *self.scene_data['config']['viewport_size'], *self.viewport_size)
            self.scene_data['proj_matrix'] = new_proj_matrix
        else:
            raise Exception('no intrinsics are provided')

        if args.init_view:
            if args.init_view in self.scene_data['view_matrix']:
                idx = self.scene_data['camera_labels'].index(args.init_view)
                init_view = self.scene_data['view_matrix'][idx]
            elif os.path.exists(args.init_view):
                init_view = np.loadtxt(args.init_view)
        else:
            init_view = self.scene_data['view_matrix'][0]

        if args.origin_view:
            top_view = np.eye(4)
            top_view[2, 3] = 20.
            init_view = top_view

            if np.allclose(self.scene_data['model3d_origin'], np.eye(4)):
                print('Setting origin as mass center')
                origin = np.eye(4)
                origin[:3, 3] = -np.percentile(self.scene_data['pointcloud']['xyz'], 90, 0)
                self.scene_data['model3d_origin'] = origin
        else:
            # force identity origin
            self.scene_data['model3d_origin'] = np.eye(4)

        self.trackball = Trackball(init_view, self.viewport_size, 1, rotation_mode=args.rmode)

        args.use_mesh = args.use_mesh or _config.get('use_mesh') or args.use_texture

        # this also creates GL context necessary for setting up shaders
        self.window = app.Window(width=self.viewport_size[0], height=self.viewport_size[1], visible=True, fullscreen=False)
        self.window.set_size(*self.viewport_size)

        if args.checkpoint:
            assert 'Texture' in args.checkpoint, 'Set path to descriptors checkpoint'
            ep = re.search('epoch_[0-9]+', args.checkpoint).group().split('_')[-1]
            net_name = f'UNet_stage_0_epoch_{ep}_net.pth'
            net_ckpt = os.path.join(*args.checkpoint.split('/')[:-1], net_name)
            texture_ckpt = args.checkpoint

        need_neural_render = net_ckpt is not None
        self.out_buffer_location = 'torch' if need_neural_render else 'opengl'

        # setup screen image plane
        self.off_render = OffscreenRender(viewport_size=self.viewport_size, out_buffer_location=self.out_buffer_location,
                                            clear_color=args.clear_color)
        if self.out_buffer_location == 'torch':
            screen_tex, self.screen_tex_cuda = create_shared_texture(
                np.zeros((self.viewport_size[1], self.viewport_size[0], 4), np.float32)
            )
        else:
            screen_tex, self.screen_tex_cuda = self.off_render.color_buf, None
        self.screen_program = get_screen_program(screen_tex)

        self.scene = NNScene()

        if need_neural_render:
            print(f'Net checkpoint: {net_ckpt}')
            print(f'Texture checkpoint: {texture_ckpt}')
            self.model = OGL(self.scene, self.scene_data, self.viewport_size, net_ckpt, 
                texture_ckpt, out_buffer_location=self.out_buffer_location, supersampling=args.supersampling, temporal_average=args.temp_avg)
        else:
            self.model = None

        if args.pca:
            assert texture_ckpt
            tex = torch.load(texture_ckpt, map_location='cpu')['state_dict']['texture_']
            print('PCA...')
            pca = pca_color(tex)
            pca = (pca - np.percentile(pca, 10)) / (np.percentile(pca, 90) - np.percentile(pca, 10))
            pca = np.clip(pca, 0, 1)
            self.scene_data['pointcloud']['rgb'] = np.clip(pca, 0, 1)

        setup_scene(self.scene, self.scene_data, args.use_mesh, args.use_texture)
        if args.light_position is not None:
            self.scene.set_light_position(args.light_position)

        if args.replay_camera:
            self.camera_trajectory = load_camera_trajectory(args.replay_camera)
        else:
            self.camera_trajectory = None

        self.window.attach(self.screen_program['transform'])
        self.window.push_handlers(on_init=self.on_init)
        self.window.push_handlers(on_close=self.on_close)
        self.window.push_handlers(on_draw=self.on_draw)
        self.window.push_handlers(on_resize=self.on_resize)
        self.window.push_handlers(on_key_press=self.on_key_press)
        self.window.push_handlers(on_mouse_press=self.on_mouse_press)
        self.window.push_handlers(on_mouse_drag=self.on_mouse_drag)
        self.window.push_handlers(on_mouse_release=self.on_mouse_release)
        self.window.push_handlers(on_mouse_scroll=self.on_mouse_scroll)

        self.mode0 = NNScene.MODE_COLOR
        self.mode1 = 0
        self.point_size = 1
        self.point_mode = False
        self.draw_points = not args.use_mesh
        self.flat_color = True
        self.neural_render = need_neural_render
        self.show_pca = False

        self.n_frame = 0
        self.t_elapsed = 0
        self.last_frame = None
        self.last_view_matrix = None
        self.last_gt_image = None

        self.mouse_pressed = False

        self.args = args
        self.cameras = []
        self.src = init_view[0][:12]
        self.last4 = None
        self.count = 0
        self.current = 0



    def run(self):
        if self.args.fps:
            start_fps_job()

        app.run()

    def render_frame(self, view_matrix):
        self.scene.set_camera_view(view_matrix)

        if self.neural_render:
            frame = self.model.infer()['output'].flip([0])
        else:
            self.scene.set_mode(self.mode0, self.mode1)
            if self.point_mode == 0:
                self.scene.set_splat_mode(False)
                self.scene.program['splat_mode'] = int(0)
            elif self.point_mode == 1:
                self.scene.set_splat_mode(True)
                self.scene.program['splat_mode'] = int(0)
            elif self.point_mode == 2:
                self.scene.set_splat_mode(False)
                self.scene.program['splat_mode'] = int(1)
            if not self.scene.use_point_sizes:
                self.scene.set_point_size(self.point_size)
            self.scene.set_draw_points(self.draw_points)
            self.scene.set_flat_color(self.flat_color)
            frame = self.off_render.render(self.scene)

        return frame

    def print_info(self):
        print('-- start info')

        mode = [m[0] for m in NNScene.__dict__.items() if m[0].startswith('MODE_') and self.mode0 == m[1]][0]
        print(mode)

        n_mode = [m[0] for m in NNScene.__dict__.items() if m[0].startswith('NORMALS_MODE_') and self.mode1 == m[1]][0]
        print(n_mode)

        print(f'point size {self.point_size}')
        print(f'splat mode: {self.point_mode}')

        print('-- end info')

    def save_screen(self, out_dir='./data/screenshots'):
        os.makedirs(out_dir, exist_ok=True)

        get_name = lambda s: time.strftime(f"%m-%d_%H-%M-%S___{s}")
        
        img = self.last_frame.cpu().numpy()[..., :3][::-1, :, ::-1] * 255
        cv2.imwrite(os.path.join(out_dir, get_name('screenshot') + '.png'), img)
        
        np.savetxt(os.path.join(out_dir, get_name('pose') + '.txt'), self.last_view_matrix)

    def get_next_view_matrix(self, frame_num, elapsed_time):
        if self.camera_trajectory is None:
            return self.trackball.pose

        n = int(elapsed_time * args.replay_fps) % len(self.camera_trajectory)
        return self.camera_trajectory[n]

    # ===== Window events =====

    def on_init(self):
        pass

    def on_key_press(self, symbol, modifiers):
        KEY_PLUS = 61
        if symbol == glfw.GLFW_KEY_X:
            self.mode0 = NNScene.MODE_XYZ
            self.neural_render = False
        elif symbol == glfw.GLFW_KEY_N:
            self.mode0 = NNScene.MODE_NORMALS
            self.neural_render = False
        elif symbol == glfw.GLFW_KEY_C:
            self.mode0 = NNScene.MODE_COLOR
            self.neural_render = False
        elif symbol == glfw.GLFW_KEY_U:
            self.mode0 = NNScene.MODE_UV
            self.neural_render = False
        # elif symbol == glfw.GLFW_KEY_D:
        #     self.mode0 = NNScene.MODE_DEPTH
            self.neural_render = False
        elif symbol == glfw.GLFW_KEY_L:
            self.mode0 = NNScene.MODE_LABEL
            self.neural_render = False
        elif symbol == glfw.GLFW_KEY_Y:
            self.neural_render = True
            self.show_pca = False
        elif symbol == glfw.GLFW_KEY_T:
            self.neural_render = True
            self.show_pca = True
        elif symbol == glfw.GLFW_KEY_Z:
            self.mode1 = (self.mode1 + 1) % 5
        elif symbol == KEY_PLUS:
            self.point_size = self.point_size + 1
        elif symbol == glfw.GLFW_KEY_MINUS:
            self.point_size = max(0, self.point_size - 1)
        elif symbol == glfw.GLFW_KEY_P:
            self.point_mode = (self.point_mode + 1) % 3
        # elif symbol == glfw.GLFW_KEY_Q:
        #     self.draw_points = not self.draw_points
        elif symbol == glfw.GLFW_KEY_F:
            self.flat_color = not self.flat_color
        elif symbol == glfw.GLFW_KEY_I:
            self.print_info()
        # elif symbol == glfw.GLFW_KEY_S:
        #     self.save_screen()
        # control camera pose
        elif symbol == glfw.GLFW_KEY_W:
            self.trackball.translate(True, axis=2)
        elif symbol == glfw.GLFW_KEY_S:
            self.trackball.translate(direct=False, axis=2)
        elif symbol == glfw.GLFW_KEY_A:
            self.trackball.translate(True, axis=0)
        elif symbol == glfw.GLFW_KEY_D:
            self.trackball.translate(direct=False, axis=0)
        elif symbol == glfw.GLFW_KEY_Q:
            self.trackball.translate(scale=0.5, direct=False, axis=1)
        elif symbol == glfw.GLFW_KEY_E:
            self.trackball.translate(scale=0.5, direct=True, axis=1)
        elif symbol == glfw.GLFW_KEY_1:
            # left rotate
            self.trackball.rotate(True, axis=1)
        elif symbol == glfw.GLFW_KEY_G:
            n = int(self.t_elapsed * args.replay_fps) % len(self.camera_trajectory)
            if self.count != n and (self.count - self.current) <= 0:
                pose = self.camera_trajectory[n]
                pose_ = np.array(pose).reshape(1, -1)
                extrinsics = pose_[0][:12]
                last = pose_[0][-4:]
                #extrinsics_new = extrinsics_h_rotate(extrinsics, -0.75, [], sampling=8)

                extrinsics_new =extrinsics
                extrinsics_new = np.append(extrinsics_new, last)
                extrinsics_new = np.array(extrinsics_new).reshape(4, 4)
                

                _pose=extrinsics_new
                _n_pose=extrinsics_new

                axis=1
                azimuth = math.pi/90*30
                if not True:
                    azimuth *= -1
                axis = _pose[:3,axis].flatten()
                eye = _pose[:3,3].flatten()
                rot_mat_ = transformations.rotation_matrix(  azimuth, axis, point= eye        )
        
                _pose = rot_mat_.dot(_pose)
                _n_pose = rot_mat_.dot(_n_pose)

                print('_n_pose',_n_pose)
                self.camera_trajectory[n] = _n_pose
                self.count = n

        elif symbol == glfw.GLFW_KEY_G:

            n = int(self.t_elapsed * args.replay_fps) % len(self.camera_trajectory)
            if self.count != n and (self.count - self.current) <= 0:
                pose = self.camera_trajectory[n]
                pose_ = np.array(pose).reshape(1, -1)
                extrinsics = pose_[0][:12]
                last = pose_[0][-4:]
                #extrinsics_new = extrinsics_h_rotate(extrinsics, -0.75, [], sampling=8)

                extrinsics_new =extrinsics
                extrinsics_new = np.append(extrinsics_new, last)
                extrinsics_new = np.array(extrinsics_new).reshape(4, 4)
                

                x_axis = extrinsics_new[:3,0].flatten()
                y_axis = extrinsics_new[:3,1].flatten()
                z_axis = extrinsics_new[:3,2].flatten()
                mindim = 0.3 * np.min(self.trackball._size)

                dx = 100
                dy = 0.1 


                
                direction2 = extrinsics_new[:3,0]
                eye = extrinsics_new[:3,3].flatten()

                intersection_point = linePlaneCollision(planeNormal=np.array([100,0,1.0]), planePoint=np.array([0,0,0.0]), rayDirection=direction2, rayPoint=eye)


                y_angle = dy / mindim
                y_rot_mat = transformations.rotation_matrix(
                    y_angle, project_onto_plane(-x_axis, [0.,0.,1.0]), intersection_point
                )

                x_angle = -dx / mindim
                x_rot_mat_ = transformations.rotation_matrix(
                    x_angle, y_axis, point=intersection_point
                )

                # self._n_pose = x_rot_mat_.dot(self._pose)
                _n_pose = x_rot_mat_.dot(y_rot_mat.dot(extrinsics_new))

                self.camera_trajectory[n] = _n_pose
                self.count = n

        elif symbol == glfw.GLFW_KEY_2:
            # right rotate
            self.trackball.rotate(False, axis=1)
        elif symbol == glfw.GLFW_KEY_3:
            # up rotate
            self.trackball.rotate(True, axis=0)
        elif symbol == glfw.GLFW_KEY_4:
            # down rotate
            self.trackball.rotate(False, axis=0)
        else:
            print(symbol, modifiers)

    def on_draw(self, dt):
        self.last_view_matrix = self.get_next_view_matrix(self.n_frame, self.t_elapsed)

        self.last_frame = self.render_frame(self.last_view_matrix)

        if self.out_buffer_location == 'torch':
            cpy_tensor_to_texture(self.last_frame, self.screen_tex_cuda)

        self.window.clear()

        gl.glDisable(gl.GL_CULL_FACE)

        # ensure viewport size is correct (offline renderer could change it)
        gl.glViewport(0, 0, self.viewport_size[0], self.viewport_size[1])

        self.screen_program.draw(gl.GL_TRIANGLE_STRIP)

        self.n_frame += 1
        self.t_elapsed += dt

        if self.args.nearest_train:
            ni = nearest_train(self.scene_data['view_matrix'], np.linalg.inv(self.scene_data['model3d_origin']) @ self.last_view_matrix)
            label = self.scene_data['camera_labels'][ni]
            assert self.args.gt, 'you must define path to gt images'
            path = self.args.gt.replace('*', str(label))
            if not os.path.exists(path):
                print(f'{path} NOT FOUND!')
            elif self.last_gt_image != path:
                self.last_gt_image = path
                img = cv2.imread(path)
                max_side = max(img.shape[:2])
                s = 1024 / max_side
                img = cv2.resize(img, None, None, s, s)
                cv2.imshow('nearest train', img)
            cv2.waitKey(1)

    def on_resize(self, w, h):
        print(f'on_resize {w}x{h}')
        self.trackball.resize((w, h))
        self.screen_program['position'] = [(0, 0), (0, h), (w, 0), (w, h)]

    def on_close(self):
        pass

    def on_mouse_press(self, x, y, buttons, modifiers):
        print(buttons, modifiers)
        self.trackball.set_state(Trackball.STATE_ROTATE)
        if (buttons == app.window.mouse.LEFT):
            ctrl = (modifiers & app.window.key.MOD_CTRL)
            shift = (modifiers & app.window.key.MOD_SHIFT)
            if (ctrl and shift):
                self.trackball.set_state(Trackball.STATE_ZOOM)
            elif ctrl:
                self.trackball.set_state(Trackball.STATE_ROLL)
            elif shift:
                self.trackball.set_state(Trackball.STATE_PAN)
        elif (buttons == app.window.mouse.MIDDLE):
            self.trackball.set_state(Trackball.STATE_PAN)
        elif (buttons == app.window.mouse.RIGHT):
            self.trackball.set_state(Trackball.STATE_LOCAL)

        self.trackball.down(np.array([x, y]))

        # Stop animating while using the mouse
        self.mouse_pressed = True

    def on_mouse_drag(self, x, y, dx, dy, buttons):
        self.trackball.drag(np.array([x, y]))

    def on_mouse_release(self, x, y, button, modifiers):
        self.mouse_pressed = False

    def on_mouse_scroll(self, x, y, dx, dy):
        self.trackball.scroll(dy)


if __name__ == '__main__':
    args = get_args()

    my_app = MyApp(args)
    my_app.run()

