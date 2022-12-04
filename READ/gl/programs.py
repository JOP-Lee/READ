import numpy as np

from glumpy import gloo
from glumpy import gl


# __all__ = ['Program', 'ColorProgram']


inv = np.linalg.inv


class Scene:
    def __init__(self):
        self.program = None
        self.index_buffer = None
        self.draw_points = True

    def set_draw_points(self, b):
        self.draw_points = b

    def set_indices(self, indices):
        self.index_buffer = indices.view(gloo.IndexBuffer)

    def set_vertices_auto(self, **kwargs):
        assert self.program is not None

        desc = []
        shape0 = []
        for k, v in kwargs.items():
            desc.append((k, np.float32, v.shape[1]))
            shape0.append(v.shape[0])

        assert len(set(shape0)), 'arrays must have the same shape[0]'
        shape0 = shape0[0]

        buf = np.zeros(shape0, desc)
        for k, v in kwargs.items():
            buf[k] = v        

        vb = buf.view(gloo.VertexBuffer)
        self.program.bind(vb)

    def set_vertices(positions, colors, normals):
        raise NotImplemented

    def set_camera_view(self, m):
        raise NotImplemented

    def set_model_view(self, m):
        raise NotImplemented

    def set_proj_matrix(self, m):
        raise NotImplemented


# #############################################################


class NNScene(Scene):
    MODE_COLOR = 0
    MODE_NORMALS = 1
    MODE_DEPTH = 2
    MODE_UV = 3
    MODE_XYZ = 4
    MODE_LABEL = 5

    NORMALS_MODE_MODEL = 0
    NORMALS_MODE_REFLECTION = 1
    NORMALS_MODE_LOCAL = 2
    NORMALS_MODE_DIRECTION = 3
    NORMALS_MODE_RAW = 4

    UV_TYPE_1D = 0
    UV_TYPE_2D = 1

    def __init__(self, flat_color=True):
        super().__init__()

        vertex = """
            uniform mat4   m_model;     // Model matrix
            uniform mat4   m_view;      // View matrix
            uniform mat4   m_proj;      // Projection matrix
            uniform mat4   m_normal;
            uniform vec3   cam_pos_world;  // Camera_position

            uniform vec3   xyz_min;
            uniform vec3   xyz_max;

            uniform int mode0;
            uniform int mode1;
            uniform float global_point_size;
            uniform float min_point_size;
            uniform int relative_point_size;
            
            in vec3 a_color;         // Vertex color
            in vec3 a_position;      // Vertex position
            in vec3 a_normal;        // Vertex normal
            in float a_uv1d;         // Vertex 1d uv coord
            in vec2 a_uv2d;          // Vertex 2d uv coord
            in float a_point_size;   // vertex point size
            in float a_discard;
            in vec2 a_perturb;
            //in float a_perturb;

            out vec3 v_position;
            out vec3 v_normal;
            out vec2 v_uv2d;
            
            out vec4 v_color_i;            // Interpolated vertex color (out)
            flat out vec4 v_color_f;       // Flat vertex color (out)

            flat out float v_discard;

            float rand(vec2 co){
                return fract(sin(dot(co.xy ,vec2(12.9898,78.233))) * 43758.5453);
            }
            
            void main()
            {   
                vec4 P = m_view * m_model * vec4(a_position, 1.0);
                v_position = P.xyz / P.w;
                v_normal = normalize(vec3(m_normal * vec4(a_normal, 0.0)));

                gl_Position = m_proj * m_view * m_model * vec4(a_position, 1.0);

                gl_Position.x += a_perturb.x;
                gl_Position.y += a_perturb.y;
                //gl_Position.z += a_perturb.z;
                //gl_Position.x += a_perturb;
                //gl_Position.y += a_perturb;
                //gl_Position.z += a_perturb;

                vec4 v_color = vec4(0., 0., 0., 1.);

                if (mode0 == 1) {
                    if (mode1 == 0) {
                        v_color = vec4(a_normal * 0.5 + 0.5, 1.0);
                    }
                    else if (mode1 == 1) {  // reflection
                        vec3 vertex_direction = normalize(cam_pos_world - a_position);
                        vec3 reflectionDirection = reflect(vertex_direction, a_normal); 
                        v_color = vec4( (normalize(reflectionDirection) * 0.5) + 0.5, 1.0);
                    } 
                    else if (mode1 == 2) {
                        // in camera frame
                        vec4 local_normal =  m_view * vec4(cam_pos_world + a_normal, 1.0); 
                        v_color = vec4( (normalize(local_normal.xyz) * 0.5) + 0.5, 1.0);
                    } 
                    else if (mode1 == 3) {
                        vec3 vertex_direction = normalize(cam_pos_world - a_position);
                        v_color = vec4( (normalize(vertex_direction) * 0.5) + 0.5, 1.0);
                    }
                    else {
                        // in case we want to put something special in this attribute, like point label
                        v_color = vec4(a_normal, 1.0);
                    }
                }
                else if (mode0 == 2) {
                    //float d = clamp(gl_Position.z / 20.f, 0, 1);
                    float d = gl_Position.z;
                    v_color = vec4(d, d, d, 1);
                }
                else if (mode0 == 3) {
                    if (mode1 == 0) {
                        v_color = vec4(a_uv1d, 0., 0., 1.);
                    }
                    else {
                        v_color = vec4(a_uv2d.r, a_uv2d.g, 0., 1.);
                    }
                }
                else if (mode0 == 4) {
                    vec3 xyz = (a_position - xyz_min) / (xyz_max - xyz_min + 1e-9);
                    v_color = vec4(xyz, 1.);
                }
                else if (mode0 == 5) {
                    v_color = vec4(a_normal.x / 255., 0.0, 0.0, 1.0);
                }
                else {
                    v_color = vec4(a_color, 1.0);
                }

                float point_size = global_point_size;
                if (point_size < 1) {
                    point_size = a_point_size;
                }

                if (relative_point_size == 1) {
                    gl_PointSize = max(min_point_size, point_size / gl_Position.z);
                } else {
                    gl_PointSize = point_size;
                }

                v_color_i = v_color;
                v_color_f = v_color;
                v_uv2d = a_uv2d;

                v_discard = a_discard;
            }
        """

        fragment = """
            uniform sampler2D texture;

            uniform int splat_mode;

            uniform int use_light;
            uniform vec3 light_position;
            const vec3 ambient_color = vec3(0.1, 0.1, 0.1);
            const vec3 diffuse_color = vec3(0.75, 0.75, 0.75);
            const vec3 specular_color = vec3(1.0, 1.0, 1.0);
            const float shininess = 128.0;
            const float gamma = 2.2;

            in vec3 v_position;
            in vec3 v_normal;
            in vec2 v_uv2d;
            
            in vec4 v_color_i;            // Interpolated vertex color (out)
            flat in vec4 v_color_f;       // Flat vertex color (out)
            out vec4 out_color;

            uniform int flat_color;       // use flat color or interpolated
            uniform int use_texture;

            flat in float v_discard;

            vec3 lightning() 
            {
                vec3 normal= normalize(v_normal);
                vec3 light_direction = normalize(light_position - v_position);
                float lambertian = max(dot(light_direction,normal), 0.0);
                float specular = 0.0;
                if (lambertian > 0.0)
                {
                    vec3 view_direction = normalize(-v_position);
                    vec3 half_direction = normalize(light_direction + view_direction);
                    float specular_angle = max(dot(half_direction, normal), 0.0);
                    specular = pow(specular_angle, shininess);
                }
                vec3 color_linear = ambient_color +
                                    lambertian * diffuse_color +
                                    specular * specular_color;
                vec3 color_gamma = pow(color_linear, vec3(1.0/gamma));
                return color_gamma;
            }

            void main()
            {
                if (v_discard == 1.f)
                    discard;

                if (use_texture == 1) {
                    out_color = texture2D(texture, vec2(v_uv2d.x, 1-v_uv2d.y));
                }
                else if (flat_color == 1) {
                    out_color = v_color_f;
                } else {
                    out_color = v_color_i;
                }

                if (use_light == 1) {
                    vec4 light = vec4(lightning(), 1);
                    //out_color = mix(light, out_color, 0.85);
                    out_color = mix(light, vec4(0.5, 0.5, 0.5, 1.), 0.65);
                }
            }
        """

        self.program = gloo.Program(vertex, fragment, version='140')

        self.program['m_view'] = np.eye(4)
        self.program['m_model'] = np.eye(4)
        self.program['light_position'] = 4.07625, 1.00545, 5.90386
        self.program['use_light'] = int(0)
        self.program['global_point_size'] = float(1)
        self.program['min_point_size'] = float(1)
        self.program['splat_mode'] = int(1)

        self.vb = None
        self.psb = None # point sizes buffer (need separate buffer for optimization)
        self.use_point_sizes = False
        self.point_discard_buffer = None
        self.point_perturb_buffer = None

    def __del__(self):
        self.delete()

    def delete(self):
        print('deleting buffers...')
        if self.vb is not None:
            gl.glDeleteBuffers(1, np.array([self.vb.handle], "I"))
            self.vb = None

        if self.psb is not None:
            gl.glDeleteBuffers(1, np.array([self.psb.handle], "I"))
            self.psb = None

        if self.program is not None:
            self.program.delete()
            self.program = None

    def set_vertices(self, positions, colors=None, normals=None, uv1d=None, uv2d=None, texture=None):
        n_pts = positions.shape[0]

        assert colors is None or n_pts == colors.shape[0], 'arrays must have the same shape[0]'
        assert normals is None or n_pts == normals.shape[0], 'arrays must have the same shape[0]'
        assert uv1d is None or n_pts == uv1d.shape[0], 'arrays must have the same shape[0]'
        assert uv2d is None or n_pts == uv2d.shape[0], 'arrays must have the same shape[0]'

        if self.vb is None:
            self.vb = np.zeros(n_pts, [
            ("a_position", np.float32, 3),
            ("a_color",    np.float32, 3),
            ("a_normal",   np.float32, 3),
            ("a_uv1d",   np.float32, 1),
            ("a_uv2d",   np.float32, 2),
            ("a_point_size", np.float32, 1),
            ("a_discard", np.float32, 1),
            ("a_perturb", np.float32, 2),
            ]).view(gloo.VertexBuffer)

        self.vb['a_position'] = positions
        # vb['a_color'] = colors / 255. if colors.max() > 1 else colors
        self.vb['a_color'] = np.zeros((n_pts, 3), dtype=np.float32) if colors is None else colors
        self.vb['a_normal'] = np.zeros((n_pts, 3), dtype=np.float32) if normals is None else normals
        self.vb['a_uv1d'] = np.zeros((n_pts,), dtype=np.float32) if uv1d is None else uv1d
        self.vb['a_uv2d'] = np.zeros((n_pts, 2), dtype=np.float32) if uv2d is None else uv2d
        self.vb['a_discard'] = np.zeros(positions.shape[0], dtype=np.float32)
        # self.vb['a_discard'] = np.random.rand(positions.shape[0]) > 0.5

        self.program['texture'] = texture if texture is not None else np.zeros((1, 1, 3), np.uint8)

        self.program['xyz_min'] = positions.min(axis=0)
        self.program['xyz_max'] = positions.max(axis=0)

        self.program.bind(self.vb)

    def set_point_sizes(self, point_sizes):
        if self.psb is None:
            self.psb = np.zeros(point_sizes.shape[0], [("a_point_size", np.float32, 1)]).view(gloo.VertexBuffer)
        self.psb['a_point_size'] = point_sizes.astype(np.float32)
        self.program.bind(self.psb)
        self.program['global_point_size'] = float(0)
        self.use_point_sizes = True

    def set_point_discard(self, arr):
        if self.point_discard_buffer is None:
            self.point_discard_buffer = np.zeros(arr.shape[0], [("a_discard", np.float32, 1)]).view(gloo.VertexBuffer)
        self.point_discard_buffer['a_discard'] = arr.astype(np.float32)
        self.program.bind(self.point_discard_buffer)

    def set_point_perturb(self, arr):
        if self.point_perturb_buffer is None:
            self.point_perturb_buffer = np.zeros(arr.shape[0], [("a_perturb", np.float32, 2)]).view(gloo.VertexBuffer)
        self.point_perturb_buffer['a_perturb'] = arr.astype(np.float32)
        self.program.bind(self.point_perturb_buffer)

    def set_use_light(self, b):
        self.program['use_light'] = int(b)

    def set_light_position(self, pos):
        self.program['light_position'] = pos

    def set_camera_view(self, m):
        self.program['m_view'] = inv(m).T
        self.program['cam_pos_world'] = m[:-1, -1]
        self._update_normal_matrix()

    def set_model_view(self, m):
        self.program['m_model'] = m.T
        self._update_normal_matrix()

    def _update_normal_matrix(self):
        view = self.program['m_view'].reshape(4, 4).T
        model = self.program['m_model'].reshape(4, 4).T
        self.program['m_normal'] = np.array(np.matrix(np.dot(view, model)).I.T)

    def set_proj_matrix(self, m):
        self.program['m_proj'] = m.T

    def set_flat_color(self, b):
        self.program['flat_color'] = int(b)

    def set_use_texture(self, b):
        self.program['use_texture'] = int(b)

    # def _set_const_point_size(self, point_size):
    #     n = self.program['a_position'].shape[0]
    #     self.program['a_point_size'] = np.ones((n,), dtype=np.float32) * point_size

    def set_point_size(self, point_size):
        # this would make shader use global point size
        self.program['global_point_size'] = float(point_size)

    def set_splat_mode(self, b):
        self.program['relative_point_size'] = int(b)

    def set_mode(self, mode0, mode1):
        mode1 = 0 if mode1 is None else mode1
        self.program['mode0'] = int(mode0)
        self.program['mode1'] = int(mode1)

    def set_params(self, skip=[], **params):
        if self.use_point_sizes:
            skip += ['point_size'] # don't set global point size
        for k, v in params.items():
            if k in skip:
                continue
            mn = f'set_{k}'
            if hasattr(self, mn):
                m = getattr(self, mn)
                if isinstance(v, (list, tuple)):
                    m(*v)
                else:
                    m(v)