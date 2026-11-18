"""Build a 2008 slider phone (Nokia 6210 Navigator style, no logo) and export it as a .glb.

Run from the repo root (Blender 5.x; the Flatpak needs absolute paths):
    flatpak run org.blender.Blender -b --factory-startup --python "$PWD/web/blender/build_phone.py"
or, with a normal install:
    blender -b --factory-startup --python web/blender/build_phone.py

Output: web/site/assets/models/phone.glb   (+ phone_preview.png, a quick render to eyeball changes)

Units are centimetres (real size about 10.3 x 4.9 x 1.5 cm). Blender's Z is up; the glTF exporter
turns that into three.js's Y-up, so the screen faces +Z in the browser.

The website looks objects up BY NAME, so keep these names if you edit the shape:
    slide          the sliding front half (screen, keys)  - moves up to reveal the keypad
    base           the back half (camera, keypad)
    screen         flat plane with 0..1 UVs; the site draws its UI on it
    camera_lens    the back camera glass; the "fly into the camera" transition aims at it
Materials are named too (body, body_back, trim, screen, key, key_dark, key_label, accent_red, accent_green,
glass, lens, flash, metal). The site replaces them with comic-style materials of the same name.
"""
import math
import os

import bpy
import bmesh

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(os.path.dirname(HERE), "site", "assets", "models")
OUT = os.path.join(OUT_DIR, "phone.glb")

W, H = 4.9, 10.3            # width, height
D_BASE, D_SLIDE = 0.78, 0.74  # thickness of the two halves
GAP = 0.04                  # seam between halves
SCREEN_W, SCREEN_H = 3.72, 4.96   # 2.4" 240x320 (3:4)
SCREEN_Z = 2.15             # screen centre height on the slide
LENS_Z = 3.55               # back camera height

# ---------------------------------------------------------------- scene reset
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene


def material(name, rgb, rough=0.5, metal=0.0, emit=None):
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = m.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*rgb, 1)
    bsdf.inputs["Roughness"].default_value = rough
    bsdf.inputs["Metallic"].default_value = metal
    if emit:
        bsdf.inputs["Emission Color"].default_value = (*emit, 1)
        bsdf.inputs["Emission Strength"].default_value = 1.0
    return m


MAT = {
    "body": material("body", (0.035, 0.035, 0.04), 0.25),
    "body_back": material("body_back", (0.06, 0.06, 0.065), 0.55),
    "trim": material("trim", (0.75, 0.76, 0.78), 0.2, 1.0),
    "screen": material("screen", (0.02, 0.05, 0.06), 0.1, emit=(0.02, 0.05, 0.06)),
    "glass": material("glass", (0.01, 0.01, 0.012), 0.05),
    "key": material("key", (0.12, 0.12, 0.13), 0.4),
    "key_dark": material("key_dark", (0.02, 0.02, 0.025), 0.35),
    "key_label": material("key_label", (0.9, 0.9, 0.9), 0.5),
    "icon": material("icon", (0.85, 0.86, 0.9), 0.5),
    "led_blue": material("led_blue", (0.2, 0.55, 1.0), 0.3, emit=(0.2, 0.55, 1.0)),
    "accent_red": material("accent_red", (0.8, 0.05, 0.04), 0.35),
    "accent_green": material("accent_green", (0.05, 0.6, 0.15), 0.35),
    "lens": material("lens", (0.01, 0.02, 0.05), 0.02),
    "flash": material("flash", (0.95, 0.92, 0.8), 0.3),
    "metal": material("metal", (0.55, 0.56, 0.58), 0.25, 1.0),
}


def finish(obj, mat, bevel=0.0, segments=4, parent=None, smooth=True):
    obj.data.materials.clear()
    obj.data.materials.append(MAT[mat])
    if bevel:
        mod = obj.modifiers.new("bevel", "BEVEL")
        mod.width = bevel
        mod.segments = segments
        mod.limit_method = "ANGLE"
    if smooth:
        for p in obj.data.polygons:
            p.use_smooth = True
        obj.modifiers.new("wn", "WEIGHTED_NORMAL").keep_sharp = True
    if parent:
        set_parent(obj, parent)
    return obj


def set_parent(obj, parent):
    """Parent without moving: keep the child where it was built (world coordinates)."""
    bpy.context.view_layer.update()
    obj.parent = parent
    obj.matrix_parent_inverse = parent.matrix_world.inverted()


def box(name, size, loc, mat, bevel=0.0, segments=4, parent=None):
    bpy.ops.mesh.primitive_cube_add(size=1, location=loc)
    o = bpy.context.active_object
    o.name = name
    o.scale = size
    bpy.ops.object.transform_apply(scale=True)
    return finish(o, mat, bevel, segments, parent)


def rounded_slab(name, w, d, h, r, loc, mat, edge=0.12, parent=None):
    """Phone-shaped slab: rounded corners seen from the front, softened front/back edges."""
    bm = bmesh.new()
    seg = 8
    pts = []
    for cx, cz, a0 in ((w / 2 - r, h / 2 - r, 0), (-w / 2 + r, h / 2 - r, 90),
                       (-w / 2 + r, -h / 2 + r, 180), (w / 2 - r, -h / 2 + r, 270)):
        for i in range(seg + 1):
            a = math.radians(a0 + 90 * i / seg)
            pts.append((cx + r * math.cos(a), cz + r * math.sin(a)))
    front = [bm.verts.new((x, -d / 2, z)) for x, z in pts]
    face = bm.faces.new(front)
    bmesh.ops.extrude_face_region(bm, geom=[face])
    bm.verts.ensure_lookup_table()
    for v in bm.verts[len(pts):]:
        v.co.y = d / 2
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    o = bpy.data.objects.new(name, me)
    scene.collection.objects.link(o)
    o.location = loc
    return finish(o, mat, bevel=edge, segments=4, parent=parent)


def cylinder(name, radius, depth, loc, mat, rot=(math.pi / 2, 0, 0), verts=48, parent=None, bevel=0.0):
    bpy.ops.mesh.primitive_cylinder_add(vertices=verts, radius=radius, depth=depth, location=loc, rotation=rot)
    o = bpy.context.active_object
    o.name = name
    bpy.ops.object.transform_apply(rotation=True)
    return finish(o, mat, bevel, 3, parent)


def ring(name, r_out, r_in, depth, loc, mat, parent=None):
    outer = cylinder(name, r_out, depth, loc, mat, parent=None)
    inner = cylinder(name + "_cut", r_in, depth * 3, loc, mat)
    b = outer.modifiers.new("cut", "BOOLEAN")
    b.object = inner
    b.operation = "DIFFERENCE"
    bpy.context.view_layer.objects.active = outer
    bpy.ops.object.modifier_apply(modifier="cut")
    bpy.data.objects.remove(inner)
    if parent:
        set_parent(outer, parent)
    return outer


def plane(name, w, h, loc, mat, parent=None, uv=True):
    bpy.ops.mesh.primitive_plane_add(size=1, location=loc, rotation=(math.pi / 2, 0, 0))
    o = bpy.context.active_object
    o.name = name
    o.scale = (w, h, 1)
    bpy.ops.object.transform_apply(scale=True, rotation=True)
    # +90 deg about X turns the plane's normal from +Z to -Y: it faces the viewer
    return finish(o, mat, smooth=False, parent=parent)


# ---------------------------------------------------------------- the two halves
yb = D_BASE / 2 + GAP / 2           # base centre (behind the seam)
ys = -(D_SLIDE / 2 + GAP / 2)       # slide centre (in front)
front = ys - D_SLIDE / 2            # y of the front face
back = yb + D_BASE / 2              # y of the back face

base = rounded_slab("base", W, D_BASE, H, 0.75, (0, yb, 0), "body_back", edge=0.18)
slide = rounded_slab("slide", W, D_SLIDE, H, 0.75, (0, ys, 0), "body", edge=0.16)
# silver band around the slide's edge (the 6210's metal frame)
trim = rounded_slab("trim", W + 0.06, 0.16, H + 0.06, 0.78, (0, ys + 0.12, 0), "trim", edge=0.05)
set_parent(trim, slide)

# ---------------------------------------------------------------- slide front: screen + keys
glass = rounded_slab("screen_window", SCREEN_W + 0.5, 0.02, SCREEN_H + 0.62, 0.12,
                     (0, front - 0.005, SCREEN_Z - 0.04), "glass", edge=0.005, parent=slide)
screen = plane("screen", SCREEN_W, SCREEN_H, (0, front - 0.02, SCREEN_Z), "screen", parent=slide)
box("earpiece", (1.1, 0.04, 0.12), (0, front - 0.01, 4.72), "metal", bevel=0.05, parent=slide)
cylinder("front_cam", 0.11, 0.04, (1.5, front - 0.01, 4.72), "lens", parent=slide)

ky = front - 0.04
NAV_Z = -2.45


def handset(name, loc, mat, angle=0.0):
    """Little telephone-handset icon (an arc with two ear pieces), lying on the key face."""
    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions = "3D"
    cu.bevel_depth = 0.045
    sp = cu.splines.new("POLY")
    pts = [(0.2 * math.cos(a), 0.0, 0.2 * math.sin(a)) for a in [math.radians(d) for d in range(200, 341, 20)]]
    sp.points.add(len(pts) - 1)
    for p_, (x, y, z) in zip(sp.points, pts):
        p_.co = (x, y, z + 0.08, 1)
    o = bpy.data.objects.new(name, cu)
    scene.collection.objects.link(o)
    bpy.context.view_layer.objects.active = o
    o.select_set(True)
    bpy.ops.object.convert(target="MESH")
    o = bpy.context.active_object
    for i, ex in enumerate((pts[0], pts[-1])):          # ear and mouth pieces
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.075, location=(ex[0], 0, ex[2] + 0.08), segments=12, ring_count=8)
        blob = bpy.context.active_object
        blob.scale = (1.2, 0.5, 0.8)
        o.select_set(True)
        bpy.context.view_layer.objects.active = o
        bpy.ops.object.join()
    o.name = name
    o.rotation_euler = (0, angle, 0)
    o.location = loc
    bpy.ops.object.transform_apply(rotation=True)
    return finish(o, mat, smooth=True, parent=slide)


def label(name, text, loc, size, mat):
    bpy.ops.object.text_add(location=loc, rotation=(math.pi / 2, 0, 0))
    t = bpy.context.active_object
    t.data.body = text
    t.data.size = size
    t.data.align_x = "CENTER"
    t.data.extrude = 0.01
    bpy.ops.object.convert(target="MESH")
    t.name = name
    return finish(t, mat, smooth=False, parent=slide)


# rounded-square navigation key (6210 Navigator style): chrome frame, dark ring, big centre key
rounded_slab("nav_frame", 2.25, 0.10, 2.05, 0.62, (0, ky + 0.0, NAV_Z), "trim", edge=0.03, parent=slide)
rounded_slab("nav_ring", 1.98, 0.14, 1.80, 0.52, (0, ky - 0.02, NAV_Z), "key", edge=0.05, parent=slide)
rounded_slab("nav_center", 1.05, 0.20, 0.95, 0.26, (0, ky - 0.05, NAV_Z), "key_dark", edge=0.06, parent=slide)
for i, (dx, dz) in enumerate(((0, 0.72), (0, -0.72), (0.8, 0), (-0.8, 0))):   # direction ticks
    box(f"nav_tick_{i}", (0.16 if dx == 0 else 0.05, 0.03, 0.05 if dx == 0 else 0.16),
        (dx, ky - 0.1, NAV_Z + dz), "icon", bevel=0.02, parent=slide)

# soft keys: thin bars with a blue light strip above each
for side in (-1, 1):
    tag = "l" if side < 0 else "r"
    box(f"soft_key_{tag}", (1.3, 0.12, 0.34), (side * 1.45, ky, -1.12), "key", bevel=0.15, parent=slide)
    box(f"soft_led_{tag}", (1.05, 0.03, 0.06), (side * 1.45, ky - 0.03, -0.86), "led_blue", bevel=0.02, parent=slide)

# menu key (left) and clear key (right) beside the navigation key
for side in (-1, 1):
    tag = "menu" if side < 0 else "clear"
    box(f"{tag}_key", (0.62, 0.12, 0.82), (side * 1.86, ky, NAV_Z + 0.05), "key", bevel=0.2, parent=slide)
label("clear_label", "C", (1.86, ky - 0.075, NAV_Z - 0.12), 0.42, "icon")
for i in range(2):                                    # menu icon: a tiny 2x2 grid of squares
    for j in range(2):
        box(f"menu_icon_{i}{j}", (0.1, 0.03, 0.1), (-1.86 - 0.07 + i * 0.14, ky - 0.08, NAV_Z + 0.12 - j * 0.14),
            "icon", bevel=0.02, parent=slide)

# call (green) and end (red) keys with handset icons
for side in (-1, 1):
    tag = "call" if side < 0 else "end"
    box(f"{tag}_key", (1.25, 0.12, 0.62), (side * 1.45, ky, -4.05), "key", bevel=0.2, parent=slide)
    handset(f"{tag}_icon", (side * 1.45, ky - 0.09, -3.98), "accent_green" if side < 0 else "accent_red",
            angle=math.radians(180) if side < 0 else math.radians(215))   # lifted / hung-up handset
cylinder("microphone", 0.05, 0.04, (0, ky + 0.02, -4.75), "key_dark", verts=12, parent=slide)

# ---------------------------------------------------------------- base front: keypad (hidden until the slide opens)
base_front = yb - D_BASE / 2
LABELS = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "*", "0", "#"]
for r in range(4):
    for c in range(3):
        x, z = (c - 1) * 1.38, -1.25 - r * 0.92
        # thick keys: they stand proud of the base (hidden inside the closed slide until it opens)
        box(f"key_{r}{c}", (1.2, 0.18, 0.72), (x, base_front - 0.09, z), "key", bevel=0.16, parent=base)
        bpy.ops.object.text_add(location=(x, base_front - 0.185, z - 0.17), rotation=(math.pi / 2, 0, 0))
        t = bpy.context.active_object
        t.data.body = LABELS[r * 3 + c]
        t.data.size = 0.5
        t.data.align_x = "CENTER"
        t.data.extrude = 0.01
        bpy.ops.object.convert(target="MESH")
        t.name = f"key_label_{r}{c}"
        finish(t, "key_label", smooth=False, parent=base)

# ---------------------------------------------------------------- back: camera, flash, speaker
bz = back + 0.02
cylinder("camera_bump", 0.78, 0.08, (0, bz, LENS_Z), "trim", parent=base, bevel=0.03)
ring("camera_ring", 0.62, 0.42, 0.1, (0, bz + 0.03, LENS_Z), "metal", parent=base)
lens = cylinder("camera_lens", 0.42, 0.06, (0, bz + 0.02, LENS_Z), "lens", parent=base)
cylinder("lens_iris", 0.18, 0.02, (0, bz + 0.055, LENS_Z), "glass", parent=base)
box("flash", (0.5, 0.06, 0.32), (1.45, bz, LENS_Z), "flash", bevel=0.08, parent=base)
for i in range(5):
    for j in range(2):
        cylinder(f"speaker_{i}{j}", 0.06, 0.04, (-0.5 + i * 0.25, bz, -4.1 + j * 0.25), "key_dark",
                 verts=12, parent=base)

# ---------------------------------------------------------------- export
os.makedirs(OUT_DIR, exist_ok=True)
for o in scene.objects:
    o.select_set(True)
bpy.ops.export_scene.gltf(filepath=OUT, export_format="GLB", use_selection=False, export_apply=True,
                          export_yup=True, export_materials="EXPORT", export_normals=True,
                          export_texcoords=True)
print("WROTE", OUT, os.path.getsize(OUT) // 1024, "KB")

# ---------------------------------------------------------------- quick preview render (for humans)
cam_data = bpy.data.cameras.new("cam")
cam = bpy.data.objects.new("cam", cam_data)
scene.collection.objects.link(cam)
cam.location = (9, -16, 6)
cam.rotation_euler = (math.radians(75), 0, math.radians(30))
cam_data.lens = 55
scene.camera = cam
sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
sun.data.energy = 4
sun.rotation_euler = (math.radians(50), math.radians(10), math.radians(30))
scene.collection.objects.link(sun)
world = bpy.data.worlds.new("w")
world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.95, 0.85, 0.1, 1)
scene.world = world
scene.render.engine = "BLENDER_EEVEE"
scene.render.resolution_x, scene.render.resolution_y = 600, 800
scene.render.filepath = os.path.join(HERE, "phone_preview.png")
bpy.ops.render.render(write_still=True)
print("PREVIEW", scene.render.filepath)
