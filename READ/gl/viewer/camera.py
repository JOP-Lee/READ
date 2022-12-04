from glumpy import app, gl, gloo
import numpy as np

from READ.gl.viewer.trackball import Trackball
from READ.gl.render import window

assert window is not None, 'call render.get_window before import'


_trackball = None 
viewer_flags = {
            'mouse_pressed': False,
            'use_perspective_cam': True,
}


@window.event
def on_resize(width, height):
    _trackball.resize((width, height))


@window.event
def on_mouse_press(x, y, buttons, modifiers):
    """Record an initial mouse press.
    """

    print(buttons, modifiers)
    _trackball.set_state(Trackball.STATE_ROTATE)
    if (buttons == app.window.mouse.LEFT):
        print('left')
        ctrl = (modifiers & app.window.key.MOD_CTRL)
        shift = (modifiers & app.window.key.MOD_SHIFT)
        if (ctrl and shift):
            _trackball.set_state(Trackball.STATE_ZOOM)
        elif ctrl:
            _trackball.set_state(Trackball.STATE_ROLL)
        elif shift:
            _trackball.set_state(Trackball.STATE_PAN)
    elif (buttons == app.window.mouse.MIDDLE):
        _trackball.set_state(Trackball.STATE_PAN)
    elif (buttons == app.window.mouse.RIGHT):
        _trackball.set_state(Trackball.STATE_ZOOM)

    _trackball.down(np.array([x, y]))

    # Stop animating while using the mouse
    viewer_flags['mouse_pressed'] = True


@window.event
def on_mouse_drag(x, y, dx, dy, buttons):
    """Record a mouse drag.
    """
    _trackball.drag(np.array([x, y]))


@window.event
def on_mouse_release(x, y, button, modifiers):
    """Record a mouse release.
    """
    viewer_flags['mouse_pressed'] = False


@window.event
def on_mouse_scroll(x, y, dx, dy):
    """Record a mouse scroll.
    """
    if viewer_flags['use_perspective_cam']:
        _trackball.scroll(dy)
    else:
        spfc = 0.95
        spbc = 1.0 / 0.95
        sf = 1.0
        if dy > 0:
            sf = spfc * dy
        elif dy < 0:
            sf = - spbc * dy

        c = self._camera_node.camera
        xmag = max(c.xmag * sf, 1e-8)
        ymag = max(c.ymag * sf, 1e-8 * c.ymag / c.xmag)
        c.xmag = xmag
        c.ymag = ymag


def get_trackball(init_view, viewport_size, rotation_mode=1):
    global _trackball
    _trackball = Trackball(init_view, viewport_size, 1, rotation_mode=rotation_mode)

    return _trackball