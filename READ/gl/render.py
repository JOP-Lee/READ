from glumpy import app, gloo, gl

from contextlib import contextmanager
import numpy as np

try:
    import pycuda.driver
    from pycuda.gl import graphics_map_flags, BufferObject
    _PYCUDA = True
except ImportError as err:
    print('pycuda import error:', err)
    _PYCUDA = False

import torch


class OffscreenRender:
    def __init__(self, viewport_size, out_buffer_location='opengl', clear_color=None):
        self._init_buffers(viewport_size, out_buffer_location)

        self.clear_color = clear_color if clear_color is not None else (0., 0., 0., 1.)

    def _init_buffers(self, viewport_size, out_buffer_location):
        assert out_buffer_location in ['torch', 'opengl', 'numpy']

        if out_buffer_location == 'torch':
            assert _PYCUDA, 'pycuda is not available'
            try:
                import pycuda.gl.autoinit  # this may fails in headless mode
            except:
                raise RuntimeError('PyCUDA init failed, cannot use torch buffer')

            _ = torch.cuda.FloatTensor(1, 3, 512,512) # needs init here, otherwise does not work

            color_np = np.zeros((viewport_size[1], viewport_size[0], 4), np.float32)
            self.color_buf, self.color_buf_cuda = create_shared_texture(color_np)
            self.out_buf = torch.zeros((viewport_size[1], viewport_size[0], 4), dtype=torch.float32).cuda()
        elif out_buffer_location == 'opengl':
            self.color_buf = np.zeros((viewport_size[1], viewport_size[0], 4), dtype=np.float32).view(gloo.TextureFloat2D)
            self.out_buf = self.color_buf
        elif out_buffer_location == 'numpy':
            self.color_buf = np.zeros((viewport_size[1], viewport_size[0], 4), dtype=np.float32).view(gloo.TextureFloat2D)
            self.out_buf = np.zeros((viewport_size[1], viewport_size[0], 3), dtype=np.float32)

        self.viewport_size = viewport_size
        self.out_buffer_location = out_buffer_location

        self.depth_buf = gloo.DepthBuffer(viewport_size[0], viewport_size[1], gl.GL_DEPTH_COMPONENT32)

        self.fbo = gloo.FrameBuffer(color=self.color_buf, depth=self.depth_buf)

    def render(self, scene, cull_face=True):
        self.fbo.activate()

        gl.glEnable(gl.GL_PROGRAM_POINT_SIZE)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glShadeModel(gl.GL_FLAT)

        if cull_face:
            gl.glEnable(gl.GL_CULL_FACE)
            gl.glCullFace(gl.GL_BACK)
        else:
            gl.glDisable(gl.GL_CULL_FACE)
        
        gl.glClearColor(*self.clear_color)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glViewport(0, 0, self.viewport_size[0], self.viewport_size[1])

        if scene.draw_points:
            scene.program.draw(gl.GL_POINTS)
        else:
            assert scene.index_buffer is not None
            scene.program.draw(gl.GL_TRIANGLES, scene.index_buffer)

        if self.out_buffer_location == 'torch':
            frame = cpy_texture_to_tensor(self.color_buf_cuda, self.out_buf).clone()
        elif self.out_buffer_location == 'opengl':
            frame = self.out_buf
        else:
            gl.glReadPixels(0, 0, self.viewport_size[0], self.viewport_size[1], gl.GL_RGB, gl.GL_FLOAT, self.out_buf)
            frame = self.out_buf.copy()

        self.fbo.deactivate()

        return frame


@contextmanager
def cuda_activate_array(img):
    """Context manager simplifying use of pycuda.gl.RegisteredImage"""
    mapping = img.map()
    yield mapping.array(0,0)
    mapping.unmap()


@contextmanager
def cuda_activate_buffer(buf):
    mapping = buf.map()
    yield mapping.device_ptr()
    mapping.unmap()


def create_shared_texture(arr, map_flags=None):
    """Create and return a Texture2D with gloo and pycuda views."""

    if map_flags is None:
        map_flags = graphics_map_flags.WRITE_DISCARD
    
    gl_view = arr.view(gloo.TextureFloat2D)
    gl_view.activate() # force gloo to create on GPU
    gl_view.deactivate()

    cuda_view = pycuda.gl.RegisteredImage(
        int(gl_view.handle), gl_view.target, map_flags)

    return gl_view, cuda_view


def create_shared_buffer(arr):
    """Create and return a BufferObject with gloo and pycuda views."""
    gl_view = arr.view(gloo.VertexBuffer)
    gl_view.activate() # force gloo to create on GPU
    gl_view.deactivate()
    cuda_view = BufferObject(np.long(gl_view.handle))
    return gl_view, cuda_view


def cpy_texture_to_tensor(texture, tensor):
    """Copy GL texture (cuda view) to pytorch tensor"""
    with cuda_activate_array(texture) as src:
        cpy = pycuda.driver.Memcpy2D()

        cpy.set_src_array(src)
        cpy.set_dst_device(tensor.data_ptr())
        cpy.width_in_bytes = cpy.src_pitch = cpy.dst_pitch = tensor.shape[1] * 4 * 4  
        cpy.height = tensor.shape[0] 
        cpy(aligned=False)

        torch.cuda.synchronize()

    return tensor


def cpy_tensor_to_texture(tensor, texture):
    """Copy pytorch tensor to GL texture (cuda view)"""
    with cuda_activate_array(texture) as ary:
        cpy = pycuda.driver.Memcpy2D()

        cpy.set_src_device(tensor.data_ptr())
        cpy.set_dst_array(ary)
        cpy.width_in_bytes = cpy.src_pitch = cpy.dst_pitch = tensor.shape[1] * 4 * 4  
        cpy.height = tensor.shape[0] 
        cpy(aligned=False)

        torch.cuda.synchronize()

    return tensor


def cpy_buffer_to_tensor(buffer, tensor):
    """Copy GL buffer (cuda view) to pytorch tensor"""
    n = tensor.numel()*tensor.element_size()    
    with cuda_activate_buffer(buffer) as buf_ptr:
        pycuda.driver.memcpy_dtod(tensor.data_ptr(), buf_ptr, n)


def cpy_tensor_to_buffer(tensor, buffer):
    """Copy pytorch tensor to GL buffer (cuda view)"""
    n = tensor.numel()*tensor.element_size()    
    with cuda_activate_buffer(buffer) as buf_ptr:
        pycuda.driver.memcpy_dtod(buf_ptr, tensor.data_ptr(), n)  

