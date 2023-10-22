# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from __future__ import print_function
import math, sys, random, argparse, json, os, tempfile
from datetime import datetime as dt
from collections import Counter

"""
Renders random scenes using Blender, each with with a random number of objects;
each object has a random size, position, color, and shape. Objects will be
nonintersecting but may partially occlude each other. Output images will be
written to disk as PNGs, and we will also write a JSON file for each image with
ground-truth scene information.

This file expects to be run from Blender like this:

blender --background --python render_images.py -- [arguments to this script]
"""
import math
import random
from random import sample
import numpy as np
INSIDE_BLENDER = True
try:
  import bpy, bpy_extras
  from mathutils import Vector
except ImportError as e:
  INSIDE_BLENDER = False
if INSIDE_BLENDER:
  try:
    import utils
  except ImportError as e:
    print("\nERROR")
    print("Running render_images.py from Blender and cannot import utils.py.") 
    print("You may need to add a .pth file to the site-packages of Blender's")
    print("bundled python with a command like this:\n")
    print("echo $PWD >> $BLENDER/$VERSION/python/lib/python3.5/site-packages/clevr.pth")
    print("\nWhere $BLENDER is the directory where Blender is installed, and")
    print("$VERSION is your Blender version (such as 2.78).")
    sys.exit(1)

parser = argparse.ArgumentParser()

# Input options
parser.add_argument('--base_scene_blendfile', default='data/base_scene.blend',
    help="Base blender file on which all scenes are based; includes " +
          "ground plane, lights, and camera.")
parser.add_argument('--properties_json', default='data/properties.json',
    help="JSON file defining objects, materials, sizes, and colors. " +
         "The \"colors\" field maps from CLEVR color names to RGB values; " +
         "The \"sizes\" field maps from CLEVR size names to scalars used to " +
         "rescale object models; the \"materials\" and \"shapes\" fields map " +
         "from CLEVR material and shape names to .blend files in the " +
         "--object_material_dir and --shape_dir directories respectively.")
parser.add_argument('--shape_dir', default='data/shapes',
    help="Directory where .blend files for object models are stored")
parser.add_argument('--material_dir', default='data/materials',
    help="Directory where .blend files for materials are stored")
parser.add_argument('--shape_color_combos_json', default=None,
    help="Optional path to a JSON file mapping shape names to a list of " +
         "allowed color names for that shape. This allows rendering images " +
         "for CLEVR-CoGenT.")

# Settings for objects
parser.add_argument('--min_objects', default=3, type=int,
    help="The minimum number of objects to place in each scene")
parser.add_argument('--max_objects', default=10, type=int,
    help="The maximum number of objects to place in each scene")
parser.add_argument('--min_dist', default=0.25, type=float,
    help="The minimum allowed distance between object centers")
parser.add_argument('--margin', default=0.4, type=float,
    help="Along all cardinal directions (left, right, front, back), all " +
         "objects will be at least this distance apart. This makes resolving " +
         "spatial relationships slightly less ambiguous.")
parser.add_argument('--min_pixels_per_object', default=200, type=int,
    help="All objects will have at least this many visible pixels in the " +
         "final rendered images; this ensures that no objects are fully " +
         "occluded by other objects.")
parser.add_argument('--max_retries', default=50, type=int,
    help="The number of times to try placing an object before giving up and " +
         "re-placing all objects in the scene.")

# Output settings
parser.add_argument('--start_idx', default=0, type=int,
    help="The index at which to start for numbering rendered images. Setting " +
         "this to non-zero values allows you to distribute rendering across " +
         "multiple machines and recombine the results later.")
parser.add_argument('--num_images', default=5, type=int,
    help="The number of images to render")
parser.add_argument('--filename_prefix', default='CLEVR',
    help="This prefix will be prepended to the rendered images and JSON scenes")
# parser.add_argument('--split', default='new',
#     help="Name of the split for which we are rendering. This will be added to " +
#          "the names of rendered images, and will also be stored in the JSON " +
#          "scene structure for each image.")
parser.add_argument('--output_image_dir', default='output/rmts_images/train_ood/',
    help="The directory where output images will be stored. It will be " +
         "created if it does not exist.")

parser.add_argument('--output_blend_dir', default='output/blendfiles',
    help="The directory where blender scene files will be stored, if the " +
         "user requested that these files be saved using the " +
         "--save_blendfiles flag; in this case it will be created if it does " +
         "not already exist.")
parser.add_argument('--save_blendfiles', type=int, default=0,
    help="Setting --save_blendfiles 1 will cause the blender scene file for " +
         "each generated image to be stored in the directory specified by " +
         "the --output_blend_dir flag. These files are not saved by default " +
         "because they take up ~5-10MB each.")
parser.add_argument('--version', default='1.0',
    help="String to store in the \"version\" field of the generated JSON file")
parser.add_argument('--license',
    default="Creative Commons Attribution (CC-BY 4.0)",
    help="String to store in the \"license\" field of the generated JSON file")
parser.add_argument('--date', default=dt.today().strftime("%m/%d/%Y"),
    help="String to store in the \"date\" field of the generated JSON file; " +
         "defaults to today's date")

# Rendering options
parser.add_argument('--use_gpu', default=0, type=int,
    help="Setting --use_gpu 1 enables GPU-accelerated rendering using CUDA. " +
         "You must have an NVIDIA GPU with the CUDA toolkit installed for " +
         "to work.")
parser.add_argument('--width', default=320, type=int,
    help="The width (in pixels) for the rendered images")
parser.add_argument('--height', default=240, type=int,
    help="The height (in pixels) for the rendered images")
parser.add_argument('--key_light_jitter', default=1.0, type=float,
    help="The magnitude of random jitter to add to the key light position.")
parser.add_argument('--fill_light_jitter', default=1.0, type=float,
    help="The magnitude of random jitter to add to the fill light position.")
parser.add_argument('--back_light_jitter', default=1.0, type=float,
    help="The magnitude of random jitter to add to the back light position.")
parser.add_argument('--camera_jitter', default=0, type=float,
    help="The magnitude of random jitter to add to the camera position")
parser.add_argument('--render_num_samples', default=64, type=int,
    help="The number of samples to use when rendering. Larger values will " +
         "result in nicer images but will cause rendering to take longer.")
parser.add_argument('--render_min_bounces', default=8, type=int,
    help="The minimum number of bounces to use for rendering.")
parser.add_argument('--render_max_bounces', default=8, type=int,
    help="The maximum number of bounces to use for rendering.")
parser.add_argument('--render_tile_size', default=64, type=int,
    help="The tile size to use for rendering. This should not affect the " +
         "quality of the rendered image but may affect the speed; CPU-based " +
         "rendering may achieve better performance using smaller tile sizes " +
         "while larger tile sizes may be optimal for GPU-based rendering.")



def rand(L):
  return 2.0 * L * (random.random() - 0.5)


def main(args):
  prob_answer_arr = np.load("RMTS_ood_train.npz")
  prob_arr = prob_answer_arr['x']
  
  # num_digits = 2
  # prefix = '%s_%s_' % (args.filename_prefix, args.split)
  # img_base_template = '%s%%0%dd.png' % (prefix, num_digits)
  # scene_base_template = '%s%%0%dd.json' % (prefix, num_digits)
  # blend_template = '%s%%0%dd.blend' % (prefix, num_digits)
  # problems = prob_answer_arr['all_problems']
  # answer_choices = prob_answer_arr['all_answer_choices']
  # blend_template = os.path.join(args.output_blend_dir, blend_template)

  for num_ex in range(10000):
    print("Generating for problem number >>",num_ex)
    img_base_template = args.output_image_dir+"prob_{}".format(num_ex)
    
    # scene_template = os.path.join(args.output_scene_dir+"prob_{}".format(num_ex), scene_base_template)
    # problems_ex = problems[num_ex].reshape(-1,3,3,3)
    # answer_choices_ex = answer_choices[num_ex]
    # context_answer_choices_ex = np.concatenate((problems_ex[:8],answer_choices_ex),axis=0)

    if not os.path.isdir(args.output_image_dir+"prob_{}".format(num_ex)):
      os.makedirs(args.output_image_dir+"prob_{}".format(num_ex))
    # if not os.path.isdir(args.output_scene_dir+"prob_{}".format(num_ex)):
    #   os.makedirs(args.output_scene_dir+"prob_{}".format(num_ex))
    # if args.save_blendfiles == 1 and not os.path.isdir(args.output_blend_dir):
    #   os.makedirs(args.output_blend_dir)
    angle = random.randrange(135,150)
    # theta = 360.0 * random.random()
    # angle = 45
    
    # material_idx = random.randrange(0,2)
    # all_scene_paths = []
    # i=0
    for i in range(2):
      
      img_path = os.path.join(img_base_template, 'CLEVR_{}.png'.format(i))
    
      all_feats = np.concatenate([prob_arr[num_ex][:2],prob_arr[num_ex][2*(i+1):2*(i+2)]],axis=0)

        # img_path = img_template % (i + args.start_idx)
        # print("image path and i>>",img_path,i)
        # scene_path = scene_template % (i + args.start_idx)
        # all_scene_paths.append(scene_path)
      blend_path = None
    # if args.save_blendfiles == 1:
    #   blend_path = blend_template % (i + args.start_idx)
    # num_objects = random.randint(args.min_objects, args.max_objects)
    # flattened_panel_feats = context_answer_choices_ex[i].reshape(-1,3)
    # object_idxs=[]
    # for j in range(9):
    #   if np.sum(flattened_panel_feats[j])!=-3:
    #     object_idxs.append(j)


    
    # num_objects = len(object_idxs) # write function to calculate number of objects
      num_objects = 4
   
      render_scene(args,angle,all_feats,
        num_objects=num_objects,
    
        output_image=img_path,
       
      )


      # After rendering all images, combine the JSON files for each scene into a
      # single JSON file.
        # all_scenes = []
        # for scene_path in all_scene_paths:
        #   with open(scene_path, 'r') as f:
        #     all_scenes.append(json.load(f))
        # output = {
        #   'info': {
        #     'date': args.date,
        #     'version': args.version,
        #     'split': args.split,
        #     'license': args.license,
        #   },
        #   'scenes': all_scenes
        # }
        # with open(args.output_scene_file, 'w') as f:
        #   json.dump(output, f)



def render_scene(args,angle,all_feats,
    num_objects=5,
    output_image='render.png',
    
  ):

  # Load the main blendfile
  bpy.ops.wm.open_mainfile(filepath=args.base_scene_blendfile)

  # Load materials
  utils.load_materials(args.material_dir)

  # Set render arguments so we can get pixel coordinates later.
  # We use functionality specific to the CYCLES renderer so BLENDER_RENDER
  # cannot be used.
  render_args = bpy.context.scene.render
  render_args.engine = "CYCLES"
  render_args.filepath = output_image
  render_args.resolution_x = args.width
  render_args.resolution_y = args.height
  render_args.resolution_percentage = 100
  render_args.tile_x = args.render_tile_size
  render_args.tile_y = args.render_tile_size
  if args.use_gpu == 1:
    # Blender changed the API for enabling CUDA at some point
    if bpy.app.version < (2, 78, 0):
      bpy.context.user_preferences.system.compute_device_type = 'CUDA'
      bpy.context.user_preferences.system.compute_device = 'CUDA_0'
    else:
      cycles_prefs = bpy.context.user_preferences.addons['cycles'].preferences
      cycles_prefs.compute_device_type = 'CUDA'

  # Some CYCLES-specific stuff
  bpy.data.worlds['World'].cycles.sample_as_light = True
  bpy.context.scene.cycles.blur_glossy = 2.0
  bpy.context.scene.cycles.samples = args.render_num_samples
  bpy.context.scene.cycles.transparent_min_bounces = args.render_min_bounces
  bpy.context.scene.cycles.transparent_max_bounces = args.render_max_bounces
  if args.use_gpu == 1:
    bpy.context.scene.cycles.device = 'GPU'

  # This will give ground-truth information about the scene and its objects
  scene_struct = {
   
      'image_filename': os.path.basename(output_image),
      'objects': [],
      'directions': {},
  }

  # Put a plane on the ground so we can compute cardinal directions
  bpy.ops.mesh.primitive_plane_add(radius=5)
  plane = bpy.context.object



  # Add random jitter to camera position
  # if args.camera_jitter > 0:
  #   for i in range(3):
  #     bpy.data.objects['Camera'].location[i] += rand(args.camera_jitter)

  # Figure out the left, up, and behind directions along the plane and record
  # them in the scene structure
  camera = bpy.data.objects['Camera']
  plane_normal = plane.data.vertices[0].normal
  cam_behind = camera.matrix_world.to_quaternion() * Vector((0, 0, -1))
  cam_left = camera.matrix_world.to_quaternion() * Vector((-1, 0, 0))
  cam_up = camera.matrix_world.to_quaternion() * Vector((0, 1, 0))
  plane_behind = (cam_behind - cam_behind.project(plane_normal)).normalized()
  plane_left = (cam_left - cam_left.project(plane_normal)).normalized()
  plane_up = cam_up.project(plane_normal).normalized()

  # Delete the plane; we only used it for normals anyway. The base scene file
  # contains the actual ground plane.
  utils.delete_object(plane)

  # Save all six axis-aligned directions in the scene struct
  scene_struct['directions']['behind'] = tuple(plane_behind)
  scene_struct['directions']['front'] = tuple(-plane_behind)
  scene_struct['directions']['left'] = tuple(plane_left)
  scene_struct['directions']['right'] = tuple(-plane_left)
  scene_struct['directions']['above'] = tuple(plane_up)
  scene_struct['directions']['below'] = tuple(-plane_up)

  # Add random jitter to lamp positions
  if args.key_light_jitter > 0:
    for i in range(3):
      bpy.data.objects['Lamp_Key'].location[i] += rand(args.key_light_jitter)
  if args.back_light_jitter > 0:
    for i in range(3):
      bpy.data.objects['Lamp_Back'].location[i] += rand(args.back_light_jitter)
  if args.fill_light_jitter > 0:
    for i in range(3):
      bpy.data.objects['Lamp_Fill'].location[i] += rand(args.fill_light_jitter)

  # Now make some random objects
  objects, blender_objects = add_random_objects(scene_struct, num_objects, args,angle,all_feats, camera)

  # Render the scene and dump the scene data structure
  scene_struct['objects'] = objects
  scene_struct['relationships'] = compute_all_relationships(scene_struct)
 
  while True:
    try:
      bpy.ops.render.render(write_still=True)
      break
    except Exception as e:
      print(e)

  # with open(output_scene, 'w') as f:
  #   json.dump(scene_struct, f, indent=2)

  # if output_blendfile is not None:
  #   bpy.ops.wm.save_as_mainfile(filepath=output_blendfile)


def add_random_objects(scene_struct, num_objects, args,angle,all_feats,camera):
  """
  Add random objects to the current blender scene
  """

  # Load the property file
  # with open(args.properties_json, 'r') as f:
  #   properties = json.load(f)
  #   color_name_to_rgba = {}
  #   for name, rgb in sorted(properties['colors'].items()):
  #     rgba = [float(c) / 255.0 for c in rgb] + [1.0]
  #     color_name_to_rgba[name] = rgba
  #   material_mapping = [(v, k) for k, v in sorted(properties['materials'].items())]
  #   object_mapping = [(v, k) for k, v in sorted(properties['shapes'].items())]
  #   size_mapping = list(sorted(properties['sizes'].items()))

  material_mapping = [('Rubber', 'rubber'), ('MyMetal', 'metal')] 
  object_mapping = [('SmoothCube_v2', 'cube'), ('SmoothCylinder', 'cylinder'), ('Sphere', 'sphere')] 

  size_mapping = [ ('small', 0.3), ('medium', 0.5), ('large', 0.7)]  

  color_mapping = [('cyan', [0.1607843137254902, 0.8156862745098039, 0.8156862745098039, 1.0]), ('brown', [0.5058823529411764, 0.2901960784313726, 0.09803921568627451, 1.0]), ('green', [0.11372549019607843, 0.4117647058823529, 0.0784313725490196, 1.0]), ('gray', [0.3411764705882353, 0.3411764705882353, 0.3411764705882353, 1.0]), ('yellow', [1.0, 0.9333333333333333, 0.2, 1.0]), ('purple', [0.5058823529411764, 0.14901960784313725, 0.7529411764705882, 1.0]), ('blue', [0.16470588235294117, 0.29411764705882354, 0.8431372549019608, 1.0]), ('red', [0.6784313725490196, 0.13725490196078433, 0.13725490196078433, 1.0])]
  # shape_color_combos = None
  # if args.shape_color_combos_json is not None:
  #   with open(args.shape_color_combos_json, 'r') as f:
  #     shape_color_combos = list(json.load(f).items())

  # print(material_mapping, object_mapping,size_mapping,list(color_name_to_rgba.items()))

  positions = []
  objects = []
  blender_objects = []
  # xys = [(-7,-1),(-4,2.5),(0.25,7),(-2.5,-3),(0,0),(2.5,3),(2,-4.5),(3.25,-2.5),(4.75,-1)]
  # xys = [(-2.8,-2.8),(-2.8,0),(-2.8,2.8),(2.8,-2.8),(2.8,0),(2.8,2.8)]
  xys = [(2.8,1.4), (2.8,-1.4), (-2.8,1.4), (-2.8,-1.4)]
  # xys= [(-4/math.sqrt(2),0),(-2/math.sqrt(2),2/math.sqrt(2)),(0,4/math.sqrt(2)),(-2/math.sqrt(2),-2/math.sqrt(2)),(0,0),(2/math.sqrt(2),2/math.sqrt(2)),(0,-4/math.sqrt(2)),(2/math.sqrt(2),-2/math.sqrt(2)),(4/math.sqrt(2),0)]
  for i in range(num_objects):
    # xys = (random.uniform(-3,3), random.uniform(-3,3))
    # size_name, r = size_mapping[flattened_panel_feats[object_idxs[i]][1]]
    size_name, r = size_mapping[all_feats[i,2]]
    # Choose a random size
    # size_name, r = random.choice(size_mapping)

    # Try to place the object, ensuring that we don't intersect any existing
    # objects and that we are more than the desired margin away from all existing
    # objects along all cardinal directions.
    num_tries = 0
    while True:
      # If we try and fail to place an object too many times, then delete all
      # the objects in the scene and start over.
      num_tries += 1
      if num_tries > args.max_retries:
        for obj in blender_objects:
          utils.delete_object(obj)
        return add_random_objects(scene_struct, num_objects, args,angle, camera)
      # x = random.uniform(-3, 3)
      # y = random.uniform(-3, 3)
      x = xys[i][0] * math.cos(math.radians(angle)) - xys[i][1] * math.sin(math.radians(angle)) + rand(0.1) 
      y = xys[i][0] * math.sin(math.radians(angle)) + xys[i][1] * math.cos(math.radians(angle))  + rand(0.1)
    

      # Check to make sure the new object is further than min_dist from all
      # other objects, and further than margin along the four cardinal directions
      dists_good = True
      margins_good = True
      # for (xx, yy, rr) in positions:

      #   dx, dy = x - xx, y - yy
      #   dist = math.sqrt(dx * dx + dy * dy)
      #   if dist - r - rr < args.min_dist:
      #     dists_good = False
      #     break
      #   for direction_name in ['left', 'right', 'front', 'behind']:
      #     direction_vec = scene_struct['directions'][direction_name]
      #     assert direction_vec[2] == 0
      #     margin = dx * direction_vec[0] + dy * direction_vec[1]
      #     if 0 < margin < args.margin:
      #       print(margin, args.margin, direction_name)
      #       print('BROKEN MARGIN!')
      #       margins_good = False
      #       break
      #   if not margins_good:
      #     break

      if dists_good and margins_good:
        break


    # Choose random color and shape
    # if shape_color_combos is None:
    # obj_name, obj_name_out = object_mapping[flattened_panel_feats[object_idxs[i]][0]]
    obj_name, obj_name_out = object_mapping[all_feats[i,1]]
    # obj_name, obj_name_out = random.choice(object_mapping)
    # color_name, rgba = list(color_name_to_rgba.items())[flattened_panel_feats[object_idxs[i]][2]]
    color_name, rgba = color_mapping[all_feats[i,0]]
    # color_name, rgba = random.choice(list(color_name_to_rgba.items()))
    # else:
    #   obj_name_out, color_choices = random.choice(shape_color_combos)
    #   color_name = random.choice(color_choices)
    #   obj_name = [k for k, v in object_mapping if v == obj_name_out][0]
    #   rgba = color_name_to_rgba[color_name]

    # For cube, adjust the size a bit
    if obj_name == 'Cube':
      r /= math.sqrt(2)

    # Choose random orientation for the object.
    theta = 0 #360.0 * random.random()

    # Actually add the object to the scene
    utils.add_object(args.shape_dir, obj_name, r, (x, y), theta=theta)
    obj = bpy.context.object
    blender_objects.append(obj)
    positions.append((x, y, r))

    # Attach a random material
    # mat_name, mat_name_out = material_mapping[material_idx]
    mat_name, mat_name_out = material_mapping[all_feats[i,3]]
    utils.add_material(mat_name, Color=rgba)

    # Record data about the object in the scene data structure

    pixel_coords = utils.get_camera_coords(camera, obj.location)
    objects.append({
      'shape': obj_name_out,
      'size': size_name,
      'material': mat_name_out,
      '3d_coords': tuple(obj.location),
      'rotation': theta,
      'pixel_coords': pixel_coords,
      'color': color_name,
    })

  #Check that all objects are at least partially visible in the rendered image
  # all_visible = check_visibility(blender_objects, args.min_pixels_per_object)
  # if not all_visible:
  #   # If any of the objects are fully occluded then start over; delete all
  #   # objects from the scene and place them all again.
  #   print('Some objects are occluded; replacing objects')
  #   for obj in blender_objects:
  #     utils.delete_object(obj)
  #   return add_random_objects(scene_struct, num_objects, args,angle, camera)

  return objects, blender_objects


def compute_all_relationships(scene_struct, eps=0.2):
  """
  Computes relationships between all pairs of objects in the scene.
  
  Returns a dictionary mapping string relationship names to lists of lists of
  integers, where output[rel][i] gives a list of object indices that have the
  relationship rel with object i. For example if j is in output['left'][i] then
  object j is left of object i.
  """
  all_relationships = {}
  for name, direction_vec in scene_struct['directions'].items():
    if name == 'above' or name == 'below': continue
    all_relationships[name] = []
    for i, obj1 in enumerate(scene_struct['objects']):
      coords1 = obj1['3d_coords']
      related = set()
      for j, obj2 in enumerate(scene_struct['objects']):
        if obj1 == obj2: continue
        coords2 = obj2['3d_coords']
        diff = [coords2[k] - coords1[k] for k in [0, 1, 2]]
        dot = sum(diff[k] * direction_vec[k] for k in [0, 1, 2])
        if dot > eps:
          related.add(j)
      all_relationships[name].append(sorted(list(related)))
  return all_relationships


def check_visibility(blender_objects, min_pixels_per_object):
  """
  Check whether all objects in the scene have some minimum number of visible
  pixels; to accomplish this we assign random (but distinct) colors to all
  objects, and render using no lighting or shading or antialiasing; this
  ensures that each object is just a solid uniform color. We can then count
  the number of pixels of each color in the output image to check the visibility
  of each object.

  Returns True if all objects are visible and False otherwise.
  """
  f, path = tempfile.mkstemp(suffix='.png')
  object_colors = render_shadeless(blender_objects, path=path)
  img = bpy.data.images.load(path)
  p = list(img.pixels)
  color_count = Counter((p[i], p[i+1], p[i+2], p[i+3])
                        for i in range(0, len(p), 4))
  os.close(f)
  os.remove(path)
  if len(color_count) != len(blender_objects) + 1:
    return False
  for _, count in color_count.most_common():
    if count < min_pixels_per_object:
      return False
  return True


def render_shadeless(blender_objects, path='flat.png'):
  """
  Render a version of the scene with shading disabled and unique materials
  assigned to all objects, and return a set of all colors that should be in the
  rendered image. The image itself is written to path. This is used to ensure
  that all objects will be visible in the final rendered scene.
  """
  render_args = bpy.context.scene.render

  # Cache the render args we are about to clobber
  old_filepath = render_args.filepath
  old_engine = render_args.engine
  old_use_antialiasing = render_args.use_antialiasing

  # Override some render settings to have flat shading
  render_args.filepath = path
  render_args.engine = 'BLENDER_RENDER'
  render_args.use_antialiasing = False

  # Move the lights and ground to layer 2 so they don't render
  utils.set_layer(bpy.data.objects['Lamp_Key'], 2)
  utils.set_layer(bpy.data.objects['Lamp_Fill'], 2)
  utils.set_layer(bpy.data.objects['Lamp_Back'], 2)
  utils.set_layer(bpy.data.objects['Ground'], 2)

  # Add random shadeless materials to all objects
  object_colors = set()
  old_materials = []
  for i, obj in enumerate(blender_objects):
    old_materials.append(obj.data.materials[0])
    bpy.ops.material.new()
    mat = bpy.data.materials['Material']
    mat.name = 'Material_%d' % i
    while True:
      r, g, b = [random.random() for _ in range(3)]
      if (r, g, b) not in object_colors: break
    object_colors.add((r, g, b))
    mat.diffuse_color = [r, g, b]
    mat.use_shadeless = True
    obj.data.materials[0] = mat

  # Render the scene

  bpy.ops.render.render(write_still=True)

  # Undo the above; first restore the materials to objects
  for mat, obj in zip(old_materials, blender_objects):
    obj.data.materials[0] = mat

  # Move the lights and ground back to layer 0
  utils.set_layer(bpy.data.objects['Lamp_Key'], 0)
  utils.set_layer(bpy.data.objects['Lamp_Fill'], 0)
  utils.set_layer(bpy.data.objects['Lamp_Back'], 0)
  utils.set_layer(bpy.data.objects['Ground'], 0)

  # Set the render settings back to what they were
  render_args.filepath = old_filepath
  render_args.engine = old_engine
  render_args.use_antialiasing = old_use_antialiasing

  return object_colors


if __name__ == '__main__':
  if INSIDE_BLENDER:
    # Run normally
    argv = utils.extract_args()
    args = parser.parse_args(argv)
    main(args)
  elif '--help' in sys.argv or '-h' in sys.argv:
    parser.print_help()
  else:
    print('This script is intended to be called from blender like this:')
    print()
    print('blender --background --python render_images.py -- [args]')
    print()
    print('You can also run as a standalone python script to view all')
    print('arguments like this:')
    print()
    print('python render_images.py --help')
