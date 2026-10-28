"""
Electric motor explainer -- built entirely from code with Blender's Python API (bpy).
Tested on Blender 5.2 (needs Blender 4.4 or newer).

TWO WAYS TO RUN IT

1) Inside the Blender app (to look around / tweak):
   Scripting tab -> Open -> build_scene.py -> Run Script (Alt+P).
   It builds the whole scene in the current file. Press Space to play,
   Numpad 0 for the camera view, Render > Render Animation to render.

2) From a terminal, rendering straight to PNG frames (no UI):
   blender -b -P build_scene.py -- --frames 1 816
   Options after the "--":
     --engine EEVEE|CYCLES   (default EEVEE, fast on a GPU)
     --res 1080              (frame width; height is 16:9 vertical)
     --samples 32            (Eevee/Cycles samples)
     --out ~/electric_motor/frames
     --still 420             (render just one test frame)
     --save motor.blend      (save the built scene)
"""
import bpy, math, sys, argparse, os
from mathutils import Vector, Matrix

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--engine", default="EEVEE")
ap.add_argument("--res", type=int, default=1080)         # width; height = 16:9 vertical
ap.add_argument("--samples", type=int, default=32)
ap.add_argument("--frames", type=int, nargs=2, default=None)
ap.add_argument("--out", default="~/electric_motor/frames")
ap.add_argument("--save", default=None)
ap.add_argument("--still", type=int, default=None)      # render a single test frame
ap.add_argument("--fps", type=int, default=24)
ap.add_argument("--look", default="normal")             # normal | flat (Workbench, ~6x faster)
args, _ = ap.parse_known_args(argv)
OUT = os.path.abspath(os.path.expanduser(args.out))
os.makedirs(OUT, exist_ok=True)

FPS = args.fps
def F(sec): return int(round(sec * FPS)) + 1

# ---------------------------------------------------------------- timeline (seconds)
SHOTS = [  # name, start, end
    ("intro",     0.0,  3.5),
    ("magnets",   3.5,  7.5),
    ("attract",   7.5, 11.5),
    ("current",  11.5, 15.5),
    ("electro",  15.5, 19.0),
    ("pull",     19.0, 23.0),
    ("commut",   23.0, 27.0),
    ("spin",     27.0, 31.0),
    ("outro",    31.0, 35.0),
]
T = {n: (s, e) for n, s, e in SHOTS}
END = F(SHOTS[-1][2]) - 1

# ---------------------------------------------------------------- reset
if bpy.app.background:
    bpy.ops.wm.read_factory_settings(use_empty=True)
else:  # inside the app: wipe the current file's contents instead of resetting the UI
    if bpy.context.object and bpy.context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    for coll in (bpy.data.objects, bpy.data.meshes, bpy.data.curves, bpy.data.materials,
                 bpy.data.cameras, bpy.data.lights, bpy.data.worlds, bpy.data.actions):
        for item in list(coll):
            coll.remove(item)
scene = bpy.context.scene
scene.render.fps = FPS
PREF = bpy.context.preferences.edit
PREF_ORIG = PREF.keyframe_new_interpolation_type
PREF.keyframe_new_interpolation_type = "LINEAR"
scene.frame_start, scene.frame_end = 1, END

def link(obj):
    scene.collection.objects.link(obj)
    return obj

# ---------------------------------------------------------------- materials
def mat(name, color, metal=0.0, rough=0.5, emit=None, emit_strength=0.0):
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

RED  = (0.85, 0.06, 0.05)
BLUE = (0.04, 0.18, 0.85)
M_red    = mat("Red", RED, rough=0.35)
M_blue   = mat("Blue", BLUE, rough=0.35)
M_steel  = mat("Steel", (0.55, 0.56, 0.58), metal=1.0, rough=0.3)
M_dark   = mat("Dark", (0.08, 0.08, 0.09), rough=0.6)
M_copper = mat("Copper", (0.85, 0.3, 0.09), metal=1.0, rough=0.3,
               emit=(1.0, 0.45, 0.1), emit_strength=0.0)
M_white  = mat("White", (0.95, 0.95, 0.95), rough=0.4)
M_wire_r = mat("WireRed", (0.7, 0.03, 0.03), rough=0.4)
M_wire_k = mat("WireBlack", (0.03, 0.03, 0.03), rough=0.4)
M_spark  = mat("Spark", (1, 0.9, 0.4), emit=(1.0, 0.85, 0.3), emit_strength=25.0)
M_batt   = mat("Battery", (0.05, 0.05, 0.05), rough=0.3)
M_band   = mat("BatteryBand", (0.1, 0.75, 0.25), rough=0.3)

def pole_mat(name):
    """Rotor-end material: grey when off, red (N) / blue (S) when energised.
    'mix' input: 0 = red, 1 = blue.  'glow' input: 0 = off, 1 = on."""
    m = bpy.data.materials.new(name); m.use_nodes = True
    nt = m.node_tree; n = nt.nodes
    b = n["Principled BSDF"]
    b.inputs["Roughness"].default_value = 0.35
    mix = n.new("ShaderNodeMix"); mix.data_type = "RGBA"
    mix.inputs["A"].default_value = (*RED, 1); mix.inputs["B"].default_value = (*BLUE, 1)
    on = n.new("ShaderNodeMix"); on.data_type = "RGBA"
    on.inputs["A"].default_value = (0.5, 0.5, 0.52, 1)
    nt.links.new(mix.outputs["Result"], on.inputs["B"])
    nt.links.new(on.outputs["Result"], b.inputs["Base Color"])
    nt.links.new(mix.outputs["Result"], b.inputs["Emission Color"])
    em = n.new("ShaderNodeMath"); em.operation = "MULTIPLY"; em.inputs[1].default_value = 0.35
    nt.links.new(em.outputs[0], b.inputs["Emission Strength"])
    return m, mix.inputs["Factor"], on.inputs["Factor"], em.inputs[0]

def key(socket_or_obj, path, frame, value=None, constant=False):
    """Keyframe a property; works for node sockets and object props."""
    if value is not None:
        if path == "default_value":
            socket_or_obj.default_value = value
        else:
            setattr(socket_or_obj, path, value)
    socket_or_obj.keyframe_insert(path, frame=frame)

# ---------------------------------------------------------------- primitives
def cube(name, size, loc, material, rot=(0, 0, 0), bevel=0.04):
    bpy.ops.mesh.primitive_cube_add(size=1, location=loc, rotation=rot)
    o = bpy.context.object; o.name = name; o.scale = size
    bpy.ops.object.transform_apply(scale=True)
    if bevel:
        mod = o.modifiers.new("bev", "BEVEL"); mod.width = bevel; mod.segments = 3
    o.data.materials.append(material)
    bpy.ops.object.shade_smooth()
    return o

def cyl(name, r, depth, loc, material, rot=(0, 0, 0), verts=48):
    bpy.ops.mesh.primitive_cylinder_add(radius=r, depth=depth, location=loc,
                                        rotation=rot, vertices=verts)
    o = bpy.context.object; o.name = name
    o.data.materials.append(material)
    bpy.ops.object.shade_smooth()
    try:
        o.data.set_sharp_from_angle(angle=math.radians(40))
    except Exception:
        pass
    return o

def text(body, loc, rot, size, material, parent=None):
    bpy.ops.object.text_add(location=loc, rotation=rot)
    t = bpy.context.object; t.data.body = body; t.data.size = size
    t.data.align_x = "CENTER"; t.data.align_y = "CENTER"; t.data.extrude = 0.01
    t.data.materials.append(material)
    if parent: t.parent = parent
    return t

# ---------------------------------------------------------------- world: sky + clouds
world = bpy.data.worlds.new("Sky"); scene.world = world; world.use_nodes = True
wn = world.node_tree.nodes; wl = world.node_tree.links
bg = wn["Background"]; bg.inputs["Strength"].default_value = 1.0
tc = wn.new("ShaderNodeTexCoord")
sep = wn.new("ShaderNodeSeparateXYZ"); wl.new(tc.outputs["Generated"], sep.inputs[0])
grad = wn.new("ShaderNodeValToRGB")                       # horizon -> zenith
grad.color_ramp.elements[0].position = 0.0
grad.color_ramp.elements[0].color = (0.62, 0.78, 0.95, 1)
grad.color_ramp.elements[1].position = 0.45
grad.color_ramp.elements[1].color = (0.12, 0.35, 0.85, 1)
wl.new(sep.outputs["Z"], grad.inputs[0])
# project view direction onto a cloud plane: (x/z, y/z)
zc = wn.new("ShaderNodeMath"); zc.operation = "MAXIMUM"; zc.inputs[1].default_value = 0.04
wl.new(sep.outputs["Z"], zc.inputs[0])
dx = wn.new("ShaderNodeMath"); dx.operation = "DIVIDE"; wl.new(sep.outputs["X"], dx.inputs[0]); wl.new(zc.outputs[0], dx.inputs[1])
dy = wn.new("ShaderNodeMath"); dy.operation = "DIVIDE"; wl.new(sep.outputs["Y"], dy.inputs[0]); wl.new(zc.outputs[0], dy.inputs[1])
comb = wn.new("ShaderNodeCombineXYZ"); wl.new(dx.outputs[0], comb.inputs[0]); wl.new(dy.outputs[0], comb.inputs[1])
noise = wn.new("ShaderNodeTexNoise")
noise.inputs["Scale"].default_value = 0.9
noise.inputs["Detail"].default_value = 8.0
noise.inputs["Roughness"].default_value = 0.6
wl.new(comb.outputs[0], noise.inputs["Vector"])
cramp = wn.new("ShaderNodeValToRGB")
cramp.color_ramp.elements[0].position = 0.52
cramp.color_ramp.elements[1].position = 0.68
wl.new(noise.outputs["Fac"], cramp.inputs[0])
fade = wn.new("ShaderNodeMapRange")                       # no clouds below horizon
fade.inputs["From Min"].default_value = 0.0; fade.inputs["From Max"].default_value = 0.12
wl.new(sep.outputs["Z"], fade.inputs["Value"])
cm = wn.new("ShaderNodeMath"); cm.operation = "MULTIPLY"
wl.new(cramp.outputs["Color"], cm.inputs[0]); wl.new(fade.outputs["Result"], cm.inputs[1])
mixc = wn.new("ShaderNodeMix"); mixc.data_type = "RGBA"
mixc.inputs["B"].default_value = (1, 1, 1, 1)
wl.new(cm.outputs[0], mixc.inputs["Factor"]); wl.new(grad.outputs["Color"], mixc.inputs["A"])
wl.new(mixc.outputs["Result"], bg.inputs["Color"])

bpy.ops.object.light_add(type="SUN", rotation=(math.radians(50), math.radians(10), math.radians(35)))
sun = bpy.context.object; sun.data.energy = 3.5; sun.data.angle = math.radians(3)

# ---------------------------------------------------------------- tiled floor
M_floor = bpy.data.materials.new("Tiles"); M_floor.use_nodes = True
fn = M_floor.node_tree.nodes; fl = M_floor.node_tree.links
brick = fn.new("ShaderNodeTexBrick")
brick.inputs["Color1"].default_value = (0.62, 0.63, 0.64, 1)
brick.inputs["Color2"].default_value = (0.55, 0.56, 0.57, 1)
brick.inputs["Mortar"].default_value = (0.38, 0.38, 0.39, 1)
brick.inputs["Scale"].default_value = 1.0
brick.inputs["Mortar Size"].default_value = 0.012
brick.inputs["Brick Width"].default_value = 1.0
brick.inputs["Row Height"].default_value = 0.5
brick.offset = 0.0
ftc = fn.new("ShaderNodeTexCoord"); fl.new(ftc.outputs["Object"], brick.inputs["Vector"])
fb = fn["Principled BSDF"]; fb.inputs["Roughness"].default_value = 0.8
fl.new(brick.outputs["Color"], fb.inputs["Base Color"])
bpy.ops.mesh.primitive_plane_add(size=400, location=(0, 0, 0))
floor = bpy.context.object; floor.data.materials.append(M_floor)

# ---------------------------------------------------------------- the motor
H = 1.6           # height of motor axis
base = cube("Base", (3.6, 1.4, 0.5), (0, 0.25, 0.25), M_dark)

# stator: two big permanent magnets
magN = cube("StatorN", (0.55, 1.0, 1.5), (-1.45, 0, H), M_red, bevel=0.06)
magS = cube("StatorS", (0.55, 1.0, 1.5), ( 1.45, 0, H), M_blue, bevel=0.06)
# stands run from the base top (0.5) to the magnet bottom; slightly narrower so no faces coincide
STAND_H = (H - 0.75) - 0.5 + 0.02
cube("StandN", (0.5, 0.94, STAND_H), (-1.45, 0, 0.5 + STAND_H / 2 - 0.01), M_steel, bevel=0.02)
cube("StandS", (0.5, 0.94, STAND_H), ( 1.45, 0, 0.5 + STAND_H / 2 - 0.01), M_steel, bevel=0.02)
text("N", (-1.45, -0.505, H), (math.radians(90), 0, 0), 0.7, M_white)
text("S", ( 1.45, -0.505, H), (math.radians(90), 0, 0), 0.7, M_white)

# rotor (armature): an iron bar wrapped in copper coil, spinning about the Y axis
rotor = bpy.data.objects.new("Rotor", None); link(rotor)
rotor.location = (0, 0, H)
core = cube("Core", (1.9, 0.34, 0.34), (0, 0, 0), M_steel, bevel=0.03); core.parent = rotor
coil = []
for i in range(12):
    x = -0.62 + i * (1.24 / 11)
    bpy.ops.mesh.primitive_torus_add(major_radius=0.27, minor_radius=0.055,
                                     location=(x, 0, 0), rotation=(0, math.radians(90), 0),
                                     major_segments=40, minor_segments=12)
    t = bpy.context.object; t.data.materials.append(M_copper); t.parent = rotor
    bpy.ops.object.shade_smooth(); coil.append(t)
M_poleA, A_mix, A_on, A_glow = pole_mat("PoleA")
M_poleB, B_mix, B_on, B_glow = pole_mat("PoleB")
pa = cube("PoleA", (0.16, 0.42, 0.42), ( 0.92, 0, 0), M_poleA, bevel=0.03); pa.parent = rotor
pb = cube("PoleB", (0.16, 0.42, 0.42), (-0.92, 0, 0), M_poleB, bevel=0.03); pb.parent = rotor

# shaft: front (toward camera, -Y) carries a fan; back carries the commutator
shaft = cyl("Shaft", 0.07, 3.8, (0, 0.6, 0), M_steel, rot=(math.radians(90), 0, 0)); shaft.parent = rotor
# commutator: split ring at y = +1.1, two copper halves with gaps
for k, sgn in enumerate((1, -1)):
    bpy.ops.mesh.primitive_cylinder_add(radius=0.2, depth=0.35, vertices=48,
                                        location=(0, 1.15, 0), rotation=(math.radians(90), 0, 0))
    half = bpy.context.object; half.name = f"Commutator{k}"
    bpy.ops.object.mode_set(mode="EDIT"); bpy.ops.mesh.select_all(action="DESELECT")
    bpy.ops.object.mode_set(mode="OBJECT")
    for v in half.data.vertices:          # keep one half (x side) with a small gap
        v.select = (v.co.x * sgn) < 0.07
    bpy.ops.object.mode_set(mode="EDIT"); bpy.ops.mesh.delete(type="VERT")
    bpy.ops.object.mode_set(mode="OBJECT")
    half.data.materials.append(M_copper); half.parent = rotor
    bpy.ops.object.shade_smooth()
# fan on the front
fan = bpy.data.objects.new("Fan", None); link(fan); fan.parent = rotor; fan.location = (0, 2.3, 0)
cyl("Hub", 0.22, 0.25, (0, 0, 0), M_dark, rot=(math.radians(90), 0, 0)).parent = fan
for i in range(4):
    a = i * math.pi / 2
    b = cube(f"Blade{i}", (0.34, 0.05, 1.0), (0, 0, 0), M_white, bevel=0.02)
    b.data.transform(Matrix.Rotation(math.radians(25), 4, "Z"))   # blade pitch
    b.data.transform(Matrix.Translation((0, 0, 0.62)))
    b.rotation_euler = (0, a, 0)
    b.parent = fan
# bearing posts holding the shaft
cube("PostF", (0.3, 0.3, H - 0.5), (0, -1.15, 0.5 + (H - 0.5) / 2), M_steel, bevel=0.02)
cube("PostB", (0.3, 0.3, H - 0.5), (0, 1.55, 0.5 + (H - 0.5) / 2), M_steel, bevel=0.02)

# brushes pressing on the commutator from top and bottom
brushT = cube("BrushT", (0.14, 0.2, 0.3), (0, 1.15, H + 0.36), M_dark, bevel=0.02)
brushB = cube("BrushB", (0.14, 0.2, 0.3), (0, 1.15, H - 0.36), M_dark, bevel=0.02)

# battery
BAT = Vector((2.9, 1.2, 0))
cyl("BatteryBody", 0.45, 1.6, BAT + Vector((0, 0, 0.8)), M_batt)
cyl("BatteryBand", 0.455, 0.45, BAT + Vector((0, 0, 1.38)), M_band)
cyl("BatteryTip", 0.15, 0.12, BAT + Vector((0, 0, 1.66)), M_steel)
text("+", BAT + Vector((0, -0.46, 1.38)), (math.radians(90), 0, 0), 0.4, M_white)

def wire(name, pts, material, r=0.045):
    cu = bpy.data.curves.new(name, "CURVE"); cu.dimensions = "3D"
    sp = cu.splines.new("BEZIER"); sp.bezier_points.add(len(pts) - 1)
    for p, c in zip(sp.bezier_points, pts):
        p.co = c; p.handle_left_type = p.handle_right_type = "AUTO"
    cu.bevel_depth = r; cu.bevel_resolution = 4
    o = bpy.data.objects.new(name, cu); link(o); o.data.materials.append(material)
    return o
w_pos = wire("WirePos", [BAT + Vector((0, 0, 1.7)), Vector((2.6, 1.3, 3.0)),
                         Vector((0.8, 1.2, 2.6)), Vector((0, 1.15, H + 0.52))], M_wire_r)
w_neg = wire("WireNeg", [Vector((0, 1.15, H - 0.52)), Vector((0.4, 1.4, 0.6)),
                         Vector((1.8, 1.5, 0.1)), BAT + Vector((-0.2, 0.2, 0.05))], M_wire_k)

# ---------------------------------------------------------------- side prop: two bar magnets
PROP = Vector((-12, 0, 0))
def bar_magnet(name, loc, flip=False):
    root = bpy.data.objects.new(name, None); link(root); root.location = loc
    n = cube(name + "N", (0.9, 0.6, 0.6), (-0.45 if not flip else 0.45, 0, 0), M_red, bevel=0.04)
    s = cube(name + "S", (0.9, 0.6, 0.6), ( 0.45 if not flip else -0.45, 0, 0), M_blue, bevel=0.04)
    n.parent = s.parent = root
    text("N", (-0.45 if not flip else 0.45, -0.305, 0), (math.radians(90), 0, 0), 0.35, M_white).parent = root
    text("S", ( 0.45 if not flip else -0.45, -0.305, 0), (math.radians(90), 0, 0), 0.35, M_white).parent = root
    return root
magA = bar_magnet("BarA", PROP + Vector((-1.2, 0, 0.3)))            # N left, S right
magB = bar_magnet("BarB", PROP + Vector((3.5, 0, 0.3)))             # N left faces A's S

# ================================================================ ANIMATION
def lin(obj):
    if obj.animation_data and obj.animation_data.action:
        act = obj.animation_data.action
        curves = []
        try:
            curves = list(act.fcurves)
        except Exception:
            for layer in getattr(act, "layers", []):
                for strip in layer.strips:
                    for bag in strip.channelbags:
                        curves += list(bag.fcurves)
        for fc in curves:
            for kp in fc.keyframe_points:
                kp.interpolation = "LINEAR"

# ---- bar magnets attract (shot "attract")
s, e = T["attract"]
magB.location = PROP + Vector((3.5, 0, 0.3)); magB.keyframe_insert("location", frame=F(s))
magB.location = PROP + Vector((2.9, 0, 0.3)); magB.keyframe_insert("location", frame=F(s + 1.6))
magB.location = PROP + Vector((0.6, 0, 0.3)); magB.keyframe_insert("location", frame=F(s + 2.1))  # snap!

# ---- rotor rotation (degrees about Y). Pole A starts at the top.
ROT_START = -90.0
def rot_key(frame, deg):
    rotor.rotation_euler = (0, math.radians(deg), 0)
    rotor.keyframe_insert("rotation_euler", frame=frame)

rot_key(1, ROT_START)
rot_key(F(T["pull"][0] + 0.3), ROT_START)
rot_key(F(T["pull"][1] - 0.4), -4.0)          # pulled almost level with the magnets (eased)
rot_key(F(T["commut"][0] + 1.0), -4.0)        # hold so the viewer sees the alignment
rot_key(F(T["commut"][0] + 1.6), 0.0)         # passes level: the commutator flips
rot_key(F(T["commut"][1]), 160.0)             # carried on round
# spin up smoothly from the commut shot's speed (no jump, slow enough to follow the poles)
spin_s, out_e = T["spin"][0], T["outro"][1]
W0 = 160.0 / (T["commut"][1] - T["commut"][0] - 1.6)   # deg/s at the end of the commut shot
TURNS = 2.5                                            # turns during spin + outro
DUR = out_e - spin_s
ACC = 2 * (360 * TURNS - W0 * DUR) / DUR ** 2
def spin_angle(sec): return 160.0 + W0 * sec + 0.5 * ACC * sec ** 2
for f in range(F(spin_s), END + 1):
    rot_key(f, spin_angle((f - F(spin_s)) / FPS))
lin(rotor)
# ease the 'pull' segment for a natural snap toward the magnets
for fc in (rotor.animation_data.action.fcurves if hasattr(rotor.animation_data.action, "fcurves") else []):
    for kp in fc.keyframe_points:
        if F(T["pull"][0]) <= kp.co[0] <= F(T["pull"][1]):
            kp.interpolation = "BEZIER"

# ---- current: coil glow + pole colours
def coil_glow(frame, v):
    key(M_copper.node_tree.nodes["Principled BSDF"].inputs["Emission Strength"],
        "default_value", frame, v)
coil_glow(1, 0.0)
coil_glow(F(T["current"][0] + 0.6), 0.0)
coil_glow(F(T["current"][0] + 1.4), 0.7)
for i, t in enumerate(range(F(T["current"][0] + 1.4), END, round(FPS / 2.4))):   # gentle flicker
    coil_glow(t, 0.45 + (0.25 if i % 2 else 0))

def poles(frame, a_is_red, on=1.0):
    key(A_mix, "default_value", frame, 0.0 if a_is_red else 1.0)
    key(B_mix, "default_value", frame, 1.0 if a_is_red else 0.0)
    key(A_on, "default_value", frame, on); key(B_on, "default_value", frame, on)
    key(A_glow, "default_value", frame, on); key(B_glow, "default_value", frame, on)

poles(1, True, 0.0)
poles(F(T["electro"][0] + 0.5), True, 0.0)
poles(F(T["electro"][0] + 1.0), True, 1.0)
# flip every time the bar passes horizontal (rotation = 0, 180, 360 ...)
flip_times = [F(T["commut"][0] + 1.6)]
k = 180
while k < spin_angle(DUR):                    # solve spin_angle(sec) = k
    sec = (-W0 + math.sqrt(W0 ** 2 + 2 * ACC * (k - 160))) / ACC
    flip_times.append(F(spin_s) + sec * FPS); k += 180
a_red = True
BLEND = max(1, FPS // 24)                     # half the blend, in frames (1/24 s)
for ft in flip_times:                         # hold, then blend over 1/12 s (no hard pop)
    f = int(round(ft))
    poles(f - BLEND, a_red, 1.0)
    a_red = not a_red
    poles(f + BLEND, a_red, 1.0)
def all_fcurves(action):
    if hasattr(action, "fcurves") and len(getattr(action, "fcurves", [])):
        return list(action.fcurves)
    out = []
    for layer in action.layers:
        for strip in layer.strips:
            for bag in strip.channelbags:
                out += list(bag.fcurves)
    return out
for m in (M_poleA, M_poleB):
    for fc in all_fcurves(m.node_tree.animation_data.action):
        for kp in fc.keyframe_points:
            kp.interpolation = "LINEAR"
# ---- electrons flowing along the wires (glowing beads on a path)
def beads(curve, n, start, end, speed=0.6):
    for i in range(n):
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.07, segments=16, ring_count=8)
        b = bpy.context.object; b.data.materials.append(M_spark)
        c = b.constraints.new("FOLLOW_PATH"); c.target = curve; c.use_fixed_location = True
        b.hide_render = True; b.keyframe_insert("hide_render", frame=1)
        b.hide_render = False; b.keyframe_insert("hide_render", frame=start)
        off0 = i / n
        for f in range(start, end + 1):        # one key per frame, wrapping 0..1
            c.offset_factor = (off0 + (f - start) / FPS * speed / 3.0) % 1.0
            c.keyframe_insert("offset_factor", frame=f)
beads(w_pos, 7, F(T["current"][0] + 0.6), END)
beads(w_neg, 7, F(T["current"][0] + 0.6), END)

# ================================================================ CAMERA (one shot per caption)
target = bpy.data.objects.new("CamTarget", None); link(target)
cam_data = bpy.data.cameras.new("Cam"); cam_data.lens = 26
cam = bpy.data.objects.new("Cam", cam_data); link(cam); scene.camera = cam
tt = cam.constraints.new("TRACK_TO"); tt.target = target
tt.track_axis = "TRACK_NEGATIVE_Z"; tt.up_axis = "UP_Y"

def shot(name, cam_a, cam_b, tgt_a, tgt_b, lens=26):
    s, e = T[name]
    for f, c, t in ((F(s), cam_a, tgt_a), (F(e) - 1, cam_b, tgt_b)):
        cam.location = c; cam.keyframe_insert("location", frame=f)
        target.location = t; target.keyframe_insert("location", frame=f)
        cam_data.lens = lens; cam_data.keyframe_insert("lens", frame=f)

M = Vector((0, 0, H))
shot("intro",   (-7.5, -9.0, 4.2), (-4.0, -10.5, 3.2), M + Vector((0.5, 0, -0.3)), M + Vector((0.3, 0, -0.3)))
shot("magnets", (0, -6.4, H + 0.2), (0, -5.6, H + 0.2), M, M, lens=24)
shot("attract", PROP + Vector((1.0, -8.5, 1.6)), PROP + Vector((-0.3, -7.2, 1.4)),
                PROP + Vector((1.0, 0, 0.3)), PROP + Vector((-0.3, 0, 0.3)), lens=38)
shot("current", (4.6, -3.0, 3.8), (3.6, -2.2, 3.2), Vector((1.0, 0.8, 1.5)), Vector((0.6, 0.6, 1.6)))
shot("electro", (1.6, -3.4, H + 1.3), (0.8, -3.0, H + 0.9), M, M)
shot("pull",    (0, -6.6, H + 0.4), (0, -6.0, H + 0.3), M, M, lens=24)
shot("commut",  (-2.4, 2.6, H + 1.9), (-1.9, 2.2, H + 1.6), Vector((0, 1.0, H)), Vector((0, 1.0, H)), lens=28)
shot("spin",    (-2.0, -4.4, H + 0.8), (-6.0, -8.5, 2.8), Vector((0, -1.2, H)), M)
shot("outro",   (-5.5, -8.0, 2.4), (-7.0, -10.0, 3.0), M + Vector((0.6, 0, -0.2)), M + Vector((0.6, 0, -0.1)))
lin(cam); lin(target)

# ================================================================ RENDER SETTINGS
r = scene.render
r.resolution_x = args.res; r.resolution_y = int(args.res * 16 / 9); r.resolution_percentage = 100
r.image_settings.file_format = "PNG"
r.use_overwrite = False    # resumable: skips frames already on disk
r.filepath = OUT + "/f_"
try:
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.look = "AgX - Medium High Contrast"
except Exception:
    pass
if args.engine.upper().startswith("EEVEE"):
    for eng in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"):
        try:
            r.engine = eng; break
        except TypeError:
            continue
    scene.eevee.taa_render_samples = args.samples
else:
    r.engine = "CYCLES"
    c = scene.cycles; c.device = "CPU"; c.samples = args.samples
    c.use_denoising = True
    try: c.denoiser = "OPENIMAGEDENOISE"
    except Exception: pass
    c.max_bounces = 4; c.diffuse_bounces = 2; c.glossy_bounces = 2
    c.transmission_bounces = 0; c.volume_bounces = 0; c.transparent_max_bounces = 2
    c.use_adaptive_sampling = True; c.adaptive_threshold = 0.05
    r.use_persistent_data = True

# ================================================================ FLAT LOOK (Workbench)
# Workbench ignores shader nodes and lights, so colours go on each material's viewport
# colour, the tile texture becomes an image, and node animation is copied to keyframes.
if args.look == "flat":
    import numpy as np
    TW, TH, MORT = 256, 128, 3                         # one 1 x 0.5 tile
    tile = np.full((TH, TW, 4), 0.80, np.float32); tile[..., 3] = 1
    tile[:MORT, :, :3] = tile[-MORT:, :, :3] = 0.66
    tile[:, :MORT, :3] = tile[:, -MORT:, :3] = 0.66
    img = bpy.data.images.new("TileTex", TW, TH); img.pixels.foreach_set(tile.ravel()); img.pack()
    tex = M_floor.node_tree.nodes.new("ShaderNodeTexImage"); tex.image = img
    M_floor.node_tree.nodes.active = tex
    floor.scale = (5, 5, 1); cam_data.clip_end = 1500 # floor reaches the horizon, no cut-off edge
    for l in floor.data.uv_layers.active.data:       # repeat the tile across the 2000 m plane
        l.uv = (l.uv[0] * 2000, l.uv[1] * 4000)

    for m in bpy.data.materials:
        b = m.node_tree.nodes.get("Principled BSDF") if m.node_tree else None
        if b and not b.inputs["Base Color"].is_linked:
            m.diffuse_color = b.inputs["Base Color"].default_value
            m.metallic = min(b.inputs["Metallic"].default_value, 0.4)
            m.roughness = max(b.inputs["Roughness"].default_value, 0.35)
    M_copper.diffuse_color = (0.95, 0.52, 0.22, 1)
    M_spark.diffuse_color = (1.0, 0.92, 0.35, 1)
    M_steel.diffuse_color = (0.72, 0.74, 0.77, 1)
    M_dark.diffuse_color = (0.18, 0.2, 0.24, 1)
    M_batt.diffuse_color = (0.15, 0.16, 0.18, 1)

    GREY, R_, B_ = Vector((0.6, 0.6, 0.63)), Vector(RED), Vector(BLUE)
    COP, HOT = Vector(M_copper.diffuse_color[:3]), Vector((1.0, 0.75, 0.35))
    glow = M_copper.node_tree.nodes["Principled BSDF"].inputs["Emission Strength"]
    for f in range(1, END + 1):                      # pole colours + coil glow, frame by frame
        scene.frame_set(f)
        for m, mix, on in ((M_poleA, A_mix, A_on), (M_poleB, B_mix, B_on)):
            col = GREY.lerp(R_.lerp(B_, mix.default_value), on.default_value)
            m.diffuse_color = (*col, 1); m.keyframe_insert("diffuse_color", frame=f)
        M_copper.diffuse_color = (*COP.lerp(HOT, min(glow.default_value / 0.7, 1) * 0.6), 1)
        M_copper.keyframe_insert("diffuse_color", frame=f)

    r.engine = "BLENDER_WORKBENCH"
    sh = scene.display.shading
    sh.light = "STUDIO"; sh.studio_light = "outdoor.sl"; sh.use_world_space_lighting = True
    sh.color_type = "TEXTURE"
    sh.show_shadows = True; sh.shadow_intensity = 0.35
    sh.show_cavity = True; sh.cavity_type = "WORLD"
    sh.cavity_ridge_factor = 0.6; sh.cavity_valley_factor = 1.0
    sh.show_specular_highlight = True
    sh.show_object_outline = True; sh.object_outline_color = (0.05, 0.05, 0.08)
    floor.display.show_shadows = False
    scene.display.light_direction = (0.45, -0.35, 0.82)
    scene.display.render_aa = "8"
    world.color = (0.5, 0.56, 0.64)                  # plain soft background, no sky
    scene.view_settings.view_transform = "Standard"; scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.4
    r.image_settings.file_format = "JPEG"; r.image_settings.quality = 95   # faster than PNG
EXT = "jpg" if args.look == "flat" else "png"

PREF.keyframe_new_interpolation_type = PREF_ORIG   # leave the user's preference as it was
scene.frame_set(1)

if args.save:
    bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(args.save))

if args.still is not None:
    scene.frame_set(args.still)
    r.filepath = OUT + f"/still_{args.still:04d}.{EXT}"
    bpy.ops.render.render(write_still=True)
elif args.frames:
    scene.frame_start, scene.frame_end = args.frames
    bpy.ops.render.render(animation=True)
print("DONE", END, "frames")
