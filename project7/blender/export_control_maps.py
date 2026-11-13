"""Export clay, inverted Mist depth and mapped normals in one CPU render."""
import argparse
import json
from pathlib import Path
import sys
import time
import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view


def scene_depth_range(scene):
    distances = []
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for obj in scene.objects:
        if obj.type != 'MESH' or obj.hide_render:
            continue
        evaluated = obj.evaluated_get(depsgraph)
        for v in evaluated.data.vertices:
            co = evaluated.matrix_world @ v.co
            ndc = world_to_camera_view(scene, scene.camera, co)
            if ndc.z > 0 and 0 <= ndc.x <= 1 and 0 <= ndc.y <= 1:
                distances.append((co - scene.camera.matrix_world.translation).length)
    if not distances:
        raise ValueError('No vertices in camera view')
    near, far = np.percentile(distances, [2, 98])
    return float(near), float(max(far, near + 0.01))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', default='output')
    parser.add_argument('--name', default='demo')
    parser.add_argument('--res', type=int, default=512)
    parser.add_argument('--samples', type=int, default=64)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    out = Path(args.outdir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'CPU'
    scene.cycles.samples = args.samples
    scene.render.threads_mode = 'FIXED'
    scene.render.threads = 16
    scene.render.resolution_x = scene.render.resolution_y = args.res
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGB'
    scene.render.filepath = str(out / f'{args.name}_clay.png')
    layer = bpy.context.view_layer
    layer.use_pass_mist = layer.use_pass_normal = True
    layer.material_override = bpy.data.materials['White clay']
    near, far = scene_depth_range(scene)
    scene.world.mist_settings.start = near
    scene.world.mist_settings.depth = far - near
    scene.world.mist_settings.falloff = 'LINEAR'
    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()
    render = tree.nodes.new('CompositorNodeRLayers')
    composite = tree.nodes.new('CompositorNodeComposite')
    tree.links.new(render.outputs['Image'], composite.inputs['Image'])
    invert = tree.nodes.new('CompositorNodeInvert')
    tree.links.new(render.outputs['Mist'], invert.inputs['Color'])
    mul = tree.nodes.new('CompositorNodeMixRGB')
    mul.blend_type = 'MULTIPLY'
    mul.inputs[2].default_value = (0.5, 0.5, 0.5, 1)
    add = tree.nodes.new('CompositorNodeMixRGB')
    add.blend_type = 'ADD'
    add.inputs[2].default_value = (0.5, 0.5, 0.5, 1)
    tree.links.new(render.outputs['Normal'], mul.inputs[1])
    tree.links.new(mul.outputs[0], add.inputs[1])
    for label, socket in [('depth', invert.outputs[0]), ('normal', add.outputs[0])]:
        node = tree.nodes.new('CompositorNodeOutputFile')
        node.base_path = str(out)
        node.file_slots[0].path = f'{args.name}_{label}_'
        node.format.file_format = 'PNG'
        node.format.color_mode = 'RGB'
        node.format.color_depth = '8'
        node.format.color_management = 'OVERRIDE'
        node.format.view_settings.view_transform = 'Raw'
        tree.links.new(socket, node.inputs[0])
    start = time.perf_counter()
    bpy.ops.render.render(write_still=True)
    elapsed = time.perf_counter() - start
    info = dict(render_seconds=elapsed, near=near, far=far, resolution=args.res,
                samples=args.samples, device='CPU', threads=16, blender=bpy.app.version_string,
                note='clay/depth/normal exported together; shared render time')
    (out / f'{args.name}_render.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    print(json.dumps(info), flush=True)


if __name__ == '__main__':
    main()
