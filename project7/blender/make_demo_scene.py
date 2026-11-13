"""Run with Blender: blender -b -P blender/make_demo_scene.py."""
from pathlib import Path
import bpy
from mathutils import Vector


def look_at(obj, target):
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat('-Z', 'Y').to_euler()


def main():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'CPU'
    scene.cycles.samples = 64
    scene.cycles.use_denoising = True
    scene.render.resolution_x = scene.render.resolution_y = 512
    scene.render.resolution_percentage = 100
    scene.world = bpy.data.worlds.new('World')
    scene.world.use_nodes = True
    scene.world.node_tree.nodes['Background'].inputs['Color'].default_value = (0.18, 0.18, 0.18, 1)
    scene.world.node_tree.nodes['Background'].inputs['Strength'].default_value = 0.4
    mat = bpy.data.materials.new('White clay')
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get('Principled BSDF')
    bsdf.inputs['Base Color'].default_value = (0.8, 0.8, 0.8, 1)
    bsdf.inputs['Roughness'].default_value = 0.7
    bpy.ops.mesh.primitive_plane_add(size=24)
    bpy.context.object.name = 'Floor'
    bpy.ops.mesh.primitive_uv_sphere_add(segments=64, ring_count=32, radius=0.62, location=(-0.95, 0, 0.62))
    bpy.context.object.name = 'Sphere'
    bpy.ops.object.shade_smooth()
    bpy.ops.mesh.primitive_cube_add(size=1.05, location=(0.5, -0.65, 0.525))
    cube = bpy.context.object
    cube.name = 'Beveled cube'
    bevel = cube.modifiers.new('Bevel', 'BEVEL')
    bevel.width, bevel.segments = 0.08, 4
    cube.modifiers.new('Weighted normal', 'WEIGHTED_NORMAL')
    bpy.ops.mesh.primitive_torus_add(major_radius=0.48, minor_radius=0.17, location=(-0.4, 1.15, 0.17))
    bpy.context.object.name = 'Torus'
    bpy.ops.object.shade_smooth()
    bpy.ops.mesh.primitive_cylinder_add(vertices=64, radius=0.4, depth=1.3, location=(1.1, 0.9, 0.65))
    bpy.context.object.name = 'Cylinder'
    bevel = bpy.context.object.modifiers.new('Bevel', 'BEVEL')
    bevel.width, bevel.segments = 0.035, 3
    bpy.context.object.modifiers.new('Weighted normal', 'WEIGHTED_NORMAL')
    for obj in scene.objects:
        if obj.type == 'MESH':
            obj.data.materials.append(mat)
    data = bpy.data.cameras.new('Camera')
    data.lens = 63
    camera = bpy.data.objects.new('Camera', data)
    scene.collection.objects.link(camera)
    camera.location = (4.8, -6, 3.5)
    look_at(camera, (0, 0.3, 0.45))
    scene.camera = camera
    for name, loc, energy, size in [
        ('Key', (3, -3.2, 4.2), 300, 3.5),
        ('Fill', (-3.6, -2.8, 2.2), 80, 5),
        ('Rim', (0.5, 3.5, 3.2), 220, 2.2),
    ]:
        data = bpy.data.lights.new(name, 'AREA')
        data.energy, data.size = energy, size
        obj = bpy.data.objects.new(name, data)
        scene.collection.objects.link(obj)
        obj.location = loc
        look_at(obj, (0, 0, 0.5))
    path = Path(__file__).resolve().parents[1] / 'assets/demo_white_model.blend'
    path.parent.mkdir(exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(path))


if __name__ == '__main__':
    main()
