"""Blender helpers for narrated 3D explainers. Import inside Blender only:

    from vidkit.blender import *
    V = setup(HERE)                     # reset the scene, read audio/timing.json and the command line
    sky(); floor(); sun()
    box = cube("Box", (1, 1, 1), (0, 0, 0.5), mat("Red", RED))
    move(box, "location", V.cue(1, 0), V.cue(1, 0) + 1.5, (0, 0, 0.5), (3, 0, 0.5))
    cam = Camera()
    cam.shot(V.start(0), V.start(1), cam_a, cam_b, target_a, target_b)
    render(V)

All times are in seconds on the video clock. V.start(i) / V.end(i) / V.cue(i, k) come from
the narration, so when you edit the script and re-run narration.py everything re-times itself.

Command line (after the "--"):  --res 1080  --samples 32  --engine EEVEE|CYCLES  --device GPU|CPU  --draft
                                --still SECONDS  --out frames  --save scene.blend  --frames A B
--draft writes to frames_draft/ (unless --out is given), so it never mixes with the final frames/.
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector  # noqa: F401  (re-exported for scene files)

from . import Timeline

FPS = 24
RED = (0.85, 0.06, 0.05)
BLUE = (0.04, 0.18, 0.85)
GREEN = (0.1, 0.75, 0.25)
YELLOW = (1.0, 0.75, 0.05)
WHITE = (0.95, 0.95, 0.95)
DARK = (0.08, 0.08, 0.09)
STEEL = (0.55, 0.56, 0.58)


def F(sec):
    """Seconds -> Blender frame number (frames start at 1)."""
    return int(round(sec * FPS)) + 1


class Video(Timeline):
    def __init__(self, project_dir, args):
        super().__init__(project_dir, use_recorded=False)
        self.dir, self.args = Path(project_dir), args
        self.end_frame = F(self.total) - 1


def _args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="EEVEE")
    ap.add_argument("--res", type=int, default=1080)        # width of a vertical frame (height = 16:9)
    ap.add_argument("--landscape", action="store_true")     # 16:9 wide instead of 9:16 tall
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--device", default="GPU")              # Cycles only: GPU (Metal) or CPU
    ap.add_argument("--draft", action="store_true")         # 540 wide, 8 samples: fast preview
    ap.add_argument("--frames", type=int, nargs=2, default=None)
    ap.add_argument("--out", default="frames")
    ap.add_argument("--save", default=None)
    ap.add_argument("--still", type=float, default=None)    # render one frame at this second
    args, _ = ap.parse_known_args(argv)
    if args.draft:
        args.res, args.samples = 540, 8
        if args.out == "frames":             # keep drafts apart: the final render resumes over frames/
            args.out = "frames_draft"
    return args


def setup(project_dir):
    """Empty the scene and return the Video timeline (V.start / V.end / V.cue / V.total)."""
    V = Video(project_dir, _args())
    if bpy.app.background:
        bpy.ops.wm.read_factory_settings(use_empty=True)
    else:  # inside the app: wipe the current file's contents instead of resetting the UI
        if bpy.context.object and bpy.context.object.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        for coll in (bpy.data.objects, bpy.data.meshes, bpy.data.curves, bpy.data.materials,
                     bpy.data.cameras, bpy.data.lights, bpy.data.worlds, bpy.data.actions):
            for item in list(coll):
                coll.remove(item)
    sc = bpy.context.scene
    sc.render.fps = FPS
    sc.frame_start, sc.frame_end = 1, V.end_frame
    return V


def link(obj):
    bpy.context.scene.collection.objects.link(obj)
    return obj


def empty(name, loc=(0, 0, 0), parent=None):
    o = link(bpy.data.objects.new(name, None))
    o.location = loc
    if parent:
        o.parent = parent
    return o


# ---------------------------------------------------------------- materials
def mat(name, color, metal=0.0, rough=0.5, emit=None, emit_strength=1.0):
    """emit: glow colour (lamps, LEDs, hot wires); emit_strength 0 turns the glow off, e.g. to key it on later."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*color, 1)
    b.inputs["Metallic"].default_value = metal
    b.inputs["Roughness"].default_value = rough
    if emit:
        b.inputs["Emission Color"].default_value = (*emit, 1)
        b.inputs["Emission Strength"].default_value = emit_strength
    return m


def bsdf(material, name):
    """A Principled BSDF input socket, e.g. bsdf(M, "Emission Strength"), for key()/move()."""
    return material.node_tree.nodes["Principled BSDF"].inputs[name]


# ---------------------------------------------------------------- shapes
def _finish(o, name, material, parent):
    o.name = name
    if material:
        o.data.materials.append(material)
    if parent:
        o.parent = parent
    bpy.ops.object.shade_smooth()
    return o


def cube(name, size, loc, material, rot=(0, 0, 0), bevel=0.04, parent=None):
    bpy.ops.mesh.primitive_cube_add(size=1, location=loc, rotation=rot)
    o = bpy.context.object
    o.scale = size
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    if bevel:
        mod = o.modifiers.new("bev", "BEVEL")
        mod.width, mod.segments = bevel, 3
    return _finish(o, name, material, parent)


def cyl(name, r, depth, loc, material, rot=(0, 0, 0), verts=48, parent=None):
    bpy.ops.mesh.primitive_cylinder_add(radius=r, depth=depth, location=loc, rotation=rot, vertices=verts)
    o = _finish(bpy.context.object, name, material, parent)
    try:
        o.data.set_sharp_from_angle(angle=math.radians(40))
    except Exception:
        pass
    return o


def sphere(name, r, loc, material, parent=None):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=r, location=loc, segments=48, ring_count=24)
    return _finish(bpy.context.object, name, material, parent)


def torus(name, major, minor, loc, material, rot=(0, 0, 0), parent=None):
    bpy.ops.mesh.primitive_torus_add(major_radius=major, minor_radius=minor, location=loc, rotation=rot,
                                     major_segments=48, minor_segments=12)
    return _finish(bpy.context.object, name, material, parent)


def text(body, loc, size, material, rot=(math.pi / 2, 0, 0), parent=None):
    """3D label. The default rotation stands it up facing the -Y direction (toward a front camera)."""
    bpy.ops.object.text_add(location=loc, rotation=rot)
    t = bpy.context.object
    t.data.body, t.data.size = body, size
    t.data.align_x, t.data.align_y, t.data.extrude = "CENTER", "CENTER", 0.01
    t.data.materials.append(material)
    if parent:
        t.parent = parent
    return t


def wire(name, pts, material, r=0.045):
    """A smooth tube through the points (wires, pipes, paths for follow_path)."""
    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions = "3D"
    sp = cu.splines.new("BEZIER")
    sp.bezier_points.add(len(pts) - 1)
    for p, c in zip(sp.bezier_points, pts):
        p.co = c
        p.handle_left_type = p.handle_right_type = "AUTO"
    cu.bevel_depth, cu.bevel_resolution = r, 4
    o = link(bpy.data.objects.new(name, cu))
    o.data.materials.append(material)
    return o


# ---------------------------------------------------------------- world: sky, floor, sun
def sky():
    """Blue gradient sky with soft procedural clouds."""
    world = bpy.data.worlds.new("Sky")
    bpy.context.scene.world = world
    world.use_nodes = True
    wn, wl = world.node_tree.nodes, world.node_tree.links
    bg = wn["Background"]
    tc = wn.new("ShaderNodeTexCoord")
    sep = wn.new("ShaderNodeSeparateXYZ"); wl.new(tc.outputs["Generated"], sep.inputs[0])
    grad = wn.new("ShaderNodeValToRGB")                       # horizon -> zenith
    grad.color_ramp.elements[0].color = (0.62, 0.78, 0.95, 1)
    grad.color_ramp.elements[1].position = 0.45
    grad.color_ramp.elements[1].color = (0.12, 0.35, 0.85, 1)
    wl.new(sep.outputs["Z"], grad.inputs[0])
    # project the view direction onto a cloud plane: (x/z, y/z)
    zc = wn.new("ShaderNodeMath"); zc.operation = "MAXIMUM"; zc.inputs[1].default_value = 0.04
    wl.new(sep.outputs["Z"], zc.inputs[0])
    dx = wn.new("ShaderNodeMath"); dx.operation = "DIVIDE"
    wl.new(sep.outputs["X"], dx.inputs[0]); wl.new(zc.outputs[0], dx.inputs[1])
    dy = wn.new("ShaderNodeMath"); dy.operation = "DIVIDE"
    wl.new(sep.outputs["Y"], dy.inputs[0]); wl.new(zc.outputs[0], dy.inputs[1])
    comb = wn.new("ShaderNodeCombineXYZ")
    wl.new(dx.outputs[0], comb.inputs[0]); wl.new(dy.outputs[0], comb.inputs[1])
    noise = wn.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 0.9
    noise.inputs["Detail"].default_value = 8.0
    noise.inputs["Roughness"].default_value = 0.6
    wl.new(comb.outputs[0], noise.inputs["Vector"])
    cramp = wn.new("ShaderNodeValToRGB")
    cramp.color_ramp.elements[0].position = 0.52
    cramp.color_ramp.elements[1].position = 0.68
    wl.new(noise.outputs["Fac"], cramp.inputs[0])
    fade = wn.new("ShaderNodeMapRange")                       # no clouds below the horizon
    fade.inputs["From Max"].default_value = 0.12
    wl.new(sep.outputs["Z"], fade.inputs["Value"])
    cm = wn.new("ShaderNodeMath"); cm.operation = "MULTIPLY"
    wl.new(cramp.outputs["Color"], cm.inputs[0]); wl.new(fade.outputs["Result"], cm.inputs[1])
    mixc = wn.new("ShaderNodeMix"); mixc.data_type = "RGBA"
    mixc.inputs["B"].default_value = (1, 1, 1, 1)
    wl.new(cm.outputs[0], mixc.inputs["Factor"]); wl.new(grad.outputs["Color"], mixc.inputs["A"])
    wl.new(mixc.outputs["Result"], bg.inputs["Color"])
    return world


def floor(size=400, tile=1.0):
    """Big grey tiled floor at z = 0."""
    m = bpy.data.materials.new("Tiles")
    m.use_nodes = True
    fn, fl = m.node_tree.nodes, m.node_tree.links
    brick = fn.new("ShaderNodeTexBrick")
    brick.inputs["Color1"].default_value = (0.62, 0.63, 0.64, 1)
    brick.inputs["Color2"].default_value = (0.55, 0.56, 0.57, 1)
    brick.inputs["Mortar"].default_value = (0.38, 0.38, 0.39, 1)
    brick.inputs["Scale"].default_value = tile
    brick.inputs["Mortar Size"].default_value = 0.012
    brick.inputs["Brick Width"].default_value = 1.0
    brick.inputs["Row Height"].default_value = 0.5
    brick.offset = 0.0
    ftc = fn.new("ShaderNodeTexCoord"); fl.new(ftc.outputs["Object"], brick.inputs["Vector"])
    b = fn["Principled BSDF"]; b.inputs["Roughness"].default_value = 0.8
    fl.new(brick.outputs["Color"], b.inputs["Base Color"])
    bpy.ops.mesh.primitive_plane_add(size=size, location=(0, 0, 0))
    o = bpy.context.object
    o.name = "Floor"
    o.data.materials.append(m)
    return o


def sun(energy=3.5, rot=(50, 10, 35)):
    bpy.ops.object.light_add(type="SUN", rotation=tuple(math.radians(a) for a in rot))
    s = bpy.context.object
    s.data.energy, s.data.angle = energy, math.radians(3)
    return s


# ---------------------------------------------------------------- animation
def _fcurves(idblock):
    ad = idblock.animation_data
    if not ad or not ad.action:
        return []
    act = ad.action
    if hasattr(act, "fcurves") and len(getattr(act, "fcurves", [])):
        return list(act.fcurves)
    out = []                                     # Blender 4.4+ layered actions
    for layer in getattr(act, "layers", []):
        for strip in layer.strips:
            for bag in strip.channelbags:
                out += list(bag.fcurves)
    return out


def key(thing, path, sec, value=None, interp=None):
    """Keyframe a property at a time in seconds. Works for objects and node sockets
    (pass the socket with path "default_value"). interp: None (smooth), "LINEAR" or "CONSTANT"."""
    if value is not None:
        setattr(thing, path, value)
    thing.keyframe_insert(path, frame=F(sec))
    if interp:
        full = thing.path_from_id(path)          # e.g. 'nodes["Principled BSDF"].inputs[2].default_value'
        for fc in _fcurves(thing.id_data):
            if fc.data_path == full:
                for kp in fc.keyframe_points:
                    if abs(kp.co[0] - F(sec)) < 0.5:
                        kp.interpolation = interp


def move(thing, path, t0, t1, a, b, interp=None):
    """Animate `path` from a (at t0) to b (at t1). Smooth ease in/out unless interp="LINEAR"."""
    key(thing, path, t0, a, interp)
    key(thing, path, t1, b, interp)


def show(obj, sec, visible=True):
    """Pop an object (and its children) in or out of the render at a time."""
    for o in [obj, *obj.children_recursive]:
        if o.animation_data is None or not any(fc.data_path == "hide_render" for fc in _fcurves(o)):
            key(o, "hide_render", 0, visible, "CONSTANT")      # the opposite state until `sec`
            key(o, "hide_viewport", 0, visible, "CONSTANT")
        key(o, "hide_render", sec, not visible, "CONSTANT")
        key(o, "hide_viewport", sec, not visible, "CONSTANT")


def spin(obj, t0, t1, turns, axis=2, start_deg=None):
    """Constant-speed rotation about an axis (0=X, 1=Y, 2=Z)."""
    r = list(obj.rotation_euler)
    a0 = math.radians(start_deg) if start_deg is not None else r[axis]
    r[axis] = a0
    key(obj, "rotation_euler", t0, tuple(r), "LINEAR")
    r[axis] = a0 + turns * 2 * math.pi
    key(obj, "rotation_euler", t1, tuple(r), "LINEAR")


# ---------------------------------------------------------------- camera
class Camera:
    """A camera that always looks at a target point. Each shot() is a cut: the camera jumps to
    cam_a / tgt_a at t0 and glides to cam_b / tgt_b by t1."""

    def __init__(self, lens=26):
        self.target = empty("CamTarget")
        self.data = bpy.data.cameras.new("Cam")
        self.data.lens = lens
        self.obj = link(bpy.data.objects.new("Cam", self.data))
        bpy.context.scene.camera = self.obj
        tt = self.obj.constraints.new("TRACK_TO")
        tt.target, tt.track_axis, tt.up_axis = self.target, "TRACK_NEGATIVE_Z", "UP_Y"

    def shot(self, t0, t1, cam_a, cam_b, tgt_a, tgt_b=None, lens=26, lens_b=None, ease=True):
        tgt_b = tgt_a if tgt_b is None else tgt_b
        interp = None if ease else "LINEAR"
        f1 = (F(t1) - 1 - 1) / FPS                   # last frame of the shot, so the next one cuts
        for t, c, g, ln in ((t0, cam_a, tgt_a, lens), (f1, cam_b, tgt_b, lens_b or lens)):
            key(self.obj, "location", t, Vector(c), interp)
            key(self.target, "location", t, Vector(g), interp)
            key(self.data, "lens", t, ln, interp)


# ---------------------------------------------------------------- render
def _cycles_device(want):
    """'GPU' if asked for and a Metal GPU is available (Apple Silicon / recent AMD Macs), else 'CPU'."""
    if want.upper() != "GPU":
        return "CPU"
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = "METAL"
        prefs.get_devices()
        gpus = [d for d in prefs.devices if d.type == "METAL"]
        for d in gpus:
            d.use = True
        if gpus:
            return "GPU"
    except (KeyError, TypeError, AttributeError):
        pass
    print("No Metal GPU found; Cycles renders on the CPU")
    return "CPU"


def render(V):
    """Apply render settings from the command line and render (or just build, inside the app)."""
    a, sc = V.args, bpy.context.scene
    r = sc.render
    w, h = a.res, int(a.res * 16 / 9)
    r.resolution_x, r.resolution_y = (h, w) if a.landscape else (w, h)
    r.resolution_percentage = 100
    r.image_settings.file_format = "PNG"
    r.use_overwrite = False                      # resumable: skips frames already on disk
    out = (V.dir / a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "info.json").write_text(json.dumps({"fps": FPS}))
    r.filepath = str(out) + "/f_"
    try:
        sc.view_settings.view_transform = "AgX"
        sc.view_settings.look = "AgX - Medium High Contrast"
    except Exception:
        pass
    if a.engine.upper().startswith("EEVEE"):
        for eng in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"):
            try:
                r.engine = eng
                break
            except TypeError:
                continue
        sc.eevee.taa_render_samples = a.samples
    else:
        r.engine = "CYCLES"
        c = sc.cycles
        c.device, c.samples, c.use_denoising = _cycles_device(a.device), a.samples, True
        c.max_bounces, c.diffuse_bounces, c.glossy_bounces = 4, 2, 2
        c.use_adaptive_sampling, c.adaptive_threshold = True, 0.05
        r.use_persistent_data = True
    sc.frame_set(1)
    if a.save:
        bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(a.save))
    if not bpy.app.background:
        print(f"Built {V.end_frame} frames ({V.total:.1f}s). Space to play, Numpad 0 for the camera.")
        return
    if a.still is not None:
        sc.frame_set(F(a.still))
        r.filepath = str(out / f"still_{a.still:05.1f}s.png")
        bpy.ops.render.render(write_still=True)
        print("wrote", r.filepath)
        return
    if a.frames:
        sc.frame_start, sc.frame_end = a.frames
    bpy.ops.render.render(animation=True)
    print("DONE", V.end_frame, "frames ->", out)
