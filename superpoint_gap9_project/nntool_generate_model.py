from nntool.api import NNGraph
from tqdm import tqdm
import glob
from PIL import Image
import numpy as np
import sys
import os
import argparse
import argcomplete
import cv2
import torch

def nms_fast(in_corners, H, W, dist_thresh):
  """
  Run a faster approximate Non-Max-Suppression on numpy corners shaped:
    3xN [x_i,y_i,conf_i]^T

  Algo summary: Create a grid sized HxW. Assign each corner location a 1, rest
  are zeros. Iterate through all the 1's and convert them either to -1 or 0.
  Suppress points by setting nearby values to 0.

  Grid Value Legend:
  -1 : Kept.
    0 : Empty or suppressed.
    1 : To be processed (converted to either kept or supressed).

  NOTE: The NMS first rounds points to integers, so NMS distance might not
  be exactly dist_thresh. It also assumes points are within image boundaries.

  Inputs
    in_corners - 3xN numpy array with corners [x_i, y_i, confidence_i]^T.
    H - Image height.
    W - Image width.
    dist_thresh - Distance to suppress, measured as an infinty norm distance.
  Returns
    nmsed_corners - 3xN numpy matrix with surviving corners.
    nmsed_inds - N length numpy vector with surviving corner indices.
  """
  grid = np.zeros((H, W)).astype(int) # Track NMS data.
  inds = np.zeros((H, W)).astype(int) # Store indices of points.
  # Sort by confidence and round to nearest int.
  inds1 = np.argsort(-in_corners[2,:])
  corners = in_corners[:,inds1]
  rcorners = corners[:2,:].round().astype(int) # Rounded corners.
  # Check for edge case of 0 or 1 corners.
  if rcorners.shape[1] == 0:
    return np.zeros((3,0)).astype(int), np.zeros(0).astype(int)
  if rcorners.shape[1] == 1:
    out = np.vstack((rcorners, in_corners[2])).reshape(3,1)
    return out, np.zeros((1)).astype(int)
  # Initialize the grid.
  for i, rc in enumerate(rcorners.T):
    grid[rcorners[1,i], rcorners[0,i]] = 1
    inds[rcorners[1,i], rcorners[0,i]] = i
  # Pad the border of the grid, so that we can NMS points near the border.
  pad = dist_thresh
  grid = np.pad(grid, ((pad,pad), (pad,pad)), mode='constant')
  # Iterate through points, highest to lowest conf, suppress neighborhood.
  count = 0
  for i, rc in enumerate(rcorners.T):
    # Account for top and left padding.
    pt = (rc[0]+pad, rc[1]+pad)
    if grid[pt[1], pt[0]] == 1: # If not yet suppressed.
      grid[pt[1]-pad:pt[1]+pad+1, pt[0]-pad:pt[0]+pad+1] = 0
      grid[pt[1], pt[0]] = -1
      count += 1
  # Get all surviving -1's and return sorted array of remaining corners.
  keepy, keepx = np.where(grid==-1)
  keepy, keepx = keepy - pad, keepx - pad
  inds_keep = inds[keepy, keepx]
  out = corners[:, inds_keep]
  values = out[-1, :]
  inds2 = np.argsort(-values)
  out = out[:, inds2]
  out_inds = inds1[inds_keep[inds2]]
  return out, out_inds

def postprocessing(outs, H, W, conf_thresh = 0.015):
  cell = 8
  semi, coarse_desc = outs[0][0].transpose(0,3,1,2), outs[1][0].transpose(0,3,1,2)
# Convert pytorch -> numpy.
  semi = semi.squeeze()
  # --- Process points.
  dense = np.exp(semi) # Softmax.
  dense = dense / (np.sum(dense, axis=0)+.00001) # Should sum to 1.
  # Remove dustbin.
  nodust = dense[:-1, :, :]
  # Reshape to get full resolution heatmap.
  Hc = int(H / cell)
  Wc = int(W / cell)
  nodust = nodust.transpose(2, 0, 1)
  heatmap = np.reshape(nodust, [Hc, Wc, cell, cell])
  heatmap = np.transpose(heatmap, [0, 2, 1, 3])
  heatmap = np.reshape(heatmap, [Hc*cell, Wc*cell])
  xs, ys = np.where(heatmap >= conf_thresh) # Confidence threshold.
  if len(xs) == 0:
    return np.zeros((3, 0)), None, None
  pts = np.zeros((3, len(xs))) # Populate point data sized 3xN.
  pts[0, :] = ys
  pts[1, :] = xs
  pts[2, :] = heatmap[xs, ys]
  pts, _ = nms_fast(pts, H, W, dist_thresh=4) # Apply NMS.
  inds = np.argsort(pts[2,:])
  pts = pts[:,inds[::-1]] # Sort by confidence.
  # Remove points along border.
  bord = 4
  toremoveW = np.logical_or(pts[0, :] < bord, pts[0, :] >= (W-bord))
  toremoveH = np.logical_or(pts[1, :] < bord, pts[1, :] >= (H-bord))
  toremove = np.logical_or(toremoveW, toremoveH)
  pts = pts[:, ~toremove]
  # --- Process descriptor.
  D = coarse_desc.shape[1]
  if pts.shape[1] == 0:
    desc = np.zeros((D, 0))
  else:
    # Interpolate into descriptor map using 2D point locations.
    samp_pts = torch.from_numpy(pts[:2, :].copy())
    samp_pts[0, :] = (samp_pts[0, :] / (float(W)/2.)) - 1.
    samp_pts[1, :] = (samp_pts[1, :] / (float(H)/2.)) - 1.
    samp_pts = samp_pts.transpose(0, 1).contiguous()
    samp_pts = samp_pts.view(1, 1, -1, 2)
    samp_pts = samp_pts.float()
    desc = torch.nn.functional.grid_sample(torch.from_numpy(coarse_desc), samp_pts)
    desc = desc.data.cpu().numpy().reshape(D, -1)
    desc /= np.linalg.norm(desc, axis=0)[np.newaxis, :]
  return pts, desc, heatmap

def pt_loc_error(pts8, pts32, epsilon = 4):
  counter = 0
  error = 0.0
  for pt8 in pts8:
    min_dist = 4.00001
    for pt32 in pts32:
      diff = (pt8[:-1] - pt32[:-1])
      dist = np.sqrt(np.sum(diff**2))
      min_dist = min(min_dist,dist)
    if (min_dist < 4.0):
      counter += 1
      error += min_dist
  return error/counter

def desc_cosine_similarity(tensor8, tensor32):
  dim = tensor8.shape
  descs8 = tensor8.reshape((dim[1]*dim[2],dim[3]))
  descs32 = tensor32.reshape((dim[1]*dim[2],dim[3]))
  cosine_sim = 0.0
  for desc8, desc32 in zip(descs8,descs32):
    cosine_sim += np.dot(desc8,desc32)/(np.linalg.norm(desc8)*np.linalg.norm(desc32))
  return cosine_sim/descs32.shape[0]

      
class VideoStreamer(object):
  """ Class to help process image streams. Three types of possible inputs:"
    1.) USB Webcam.
    2.) A directory of images (files in directory matching 'img_glob').
    3.) A video file, such as an .mp4 or .avi file.
  """
  def __init__(self, basedir, camid, height, width, skip, img_glob):
    self.cap = []
    self.camera = False
    self.video_file = False
    self.listing = []
    self.sizer = [height, width]
    self.i = 0
    self.skip = skip
    self.maxlen = 1000000
    # If the "basedir" string is the word camera, then use a webcam.
    if basedir == "camera/" or basedir == "camera":
      print('==> Processing Webcam Input.')
      self.cap = cv2.VideoCapture(camid)
      self.listing = range(0, self.maxlen)
      self.camera = True
    else:
      # Try to open as a video.
      self.cap = cv2.VideoCapture(basedir)
      lastbit = basedir[-4:len(basedir)]
      if (type(self.cap) == list or not self.cap.isOpened()) and (lastbit == '.mp4'):
        raise IOError('Cannot open movie file')
      elif type(self.cap) != list and self.cap.isOpened() and (lastbit != '.txt'):
        print('==> Processing Video Input.')
        num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.listing = range(0, num_frames)
        self.listing = self.listing[::self.skip]
        self.camera = True
        self.video_file = True
        self.maxlen = len(self.listing)
      else:
        print('==> Processing Image Directory Input.')
        search = os.path.join(basedir, img_glob)
        self.listing = glob.glob(search)
        self.listing.sort()
        self.listing = self.listing[::self.skip]
        self.maxlen = len(self.listing)
        if self.maxlen == 0:
          raise IOError('No images were found (maybe bad \'--img_glob\' parameter?)')

  def read_image(self, impath, img_size):
    """ Read image as grayscale and resize to img_size.
    Inputs
      impath: Path to input image.
      img_size: (W, H) tuple specifying resize size.
    Returns
      grayim: float32 numpy array sized H x W with values in range [0, 1].
    """
    grayim = cv2.imread(impath, 0)
    if grayim is None:
      raise Exception('Error reading image %s' % impath)
    # Image is resized via opencv.
    interp = cv2.INTER_AREA
    grayim = cv2.resize(grayim, (img_size[1], img_size[0]), interpolation=interp)
    grayim = (grayim.astype('float32') / 255.)
    return grayim

  def next_frame(self):
    """ Return the next frame, and increment internal counter.
    Returns
       image: Next H x W image.
       status: True or False depending whether image was loaded.
    """
    if self.i == self.maxlen:
      return (None, False)
    if self.camera:
      ret, input_image = self.cap.read()
      if ret is False:
        print('VideoStreamer: Cannot get image from camera (maybe bad --camid?)')
        return (None, False)
      if self.video_file:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.listing[self.i])
      input_image = cv2.resize(input_image, (self.sizer[1], self.sizer[0]),
                               interpolation=cv2.INTER_AREA)
      input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
      input_image = input_image.astype('float')/255.0
    else:
      image_file = self.listing[self.i]
      input_image = self.read_image(image_file, self.sizer)
    # Increment internal counter.
    self.i = self.i + 1
    input_image = input_image.astype('float32')
    return (input_image, True)

def create_parser():
    # create the top-level parser
    parser = argparse.ArgumentParser(prog='at_model_gen')

    parser.add_argument('--trained_model', default="../superpoint.onnx",
                        help='Image directory or movie file or "camera" (for webcam).')
    parser.add_argument('--calibration', default="../assets/icl_snippet/",
                        help='Path to calibration samples')
    parser.add_argument('--img_glob', type=str, default='*.pgm',
                        help='Glob match if directory of images is specified (default: \'*.png\').')
    parser.add_argument('--skip', type=int, default=1,
                        help='Images to skip if input is movie or directory (default: 1).')
    parser.add_argument('--show_extra', action='store_true',
                        help='Show extra debug outputs (default: False).')
    parser.add_argument('--H', type=int, default=120,
                        help='Input image height (default: 120).')
    parser.add_argument('--W', type=int, default=160,
                        help='Input image width (default:160).')
    parser.add_argument('--camid', type=int, default=0,
                        help='OpenCV webcam video capture ID, usually 0 or 1 (default: 0).')
    parser.add_argument('--tensors_dir', default="tensors",
                        help="Where nntool stores the weights/bias tensors dir (only used in generate and performance mode)")
    parser.add_argument('--at_model_path', default=None,
                        help="Path to the C autotiler model file to generate (only used in generate mode)")
    parser.add_argument('--ram_type', default="AT_MEM_L3_DEFAULTRAM", choices=['AT_MEM_L3_HRAM', 'AT_MEM_L3_QSPIRAM', 'AT_MEM_L3_OSPIRAM', 'AT_MEM_L3_DEFAULTRAM'],
                        help="Ram type to use during inference on platform (only used in generate and performance mode)")
    parser.add_argument('--flash_type', default="AT_MEM_L3_DEFAULTFLASH", choices=['AT_MEM_L3_HFLASH', 'AT_MEM_L3_QSPIFLASH', 'AT_MEM_L3_OSPIFLASH', 'AT_MEM_L3_MRAMFLASH', 'AT_MEM_L3_DEFAULTFLASH'],
                        help="Flash type to use during inference (only used in generate and performance mode)")
    return parser

def main():
    parser = create_parser()
    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    def preprocessing(img_arr):
        return (img_arr - 127.5) / 127.5

    G = NNGraph.load_graph(args.trained_model)
    G.adjust_order()
    G.fusions("scaled_match_group")

    vs = VideoStreamer(args.calibration, args.camid, args.H, args.W, args.skip, args.img_glob)
    def representative_dataset():
        img, status = vs.next_frame()
        while(status):
            assert img.ndim == 2, 'Image must be grayscale.'
            assert img.dtype == np.float32, 'Image must be float32.'
            H, W = img.shape[0], img.shape[1]
            inp = img.copy()
            inp = (inp.reshape(1, H, W))
            inp = torch.from_numpy(inp)
            inp = torch.autograd.Variable(inp).view(1, 1, H, W)
            yield preprocessing(img)
            img, status = vs.next_frame()

    stats = G.collect_statistics(representative_dataset())

    G.quantize(
        stats,
        graph_options={
            "use_ne16": True,
            "hwc": True
        }
    )
    print(G.qshow([G[0]]))
    G[0].allocate = True

    print(G.qshow())
    
    vst = VideoStreamer(args.calibration, args.camid, args.H, args.W, args.skip, args.img_glob)
    def test_dataset():
        img, status = vst.next_frame()
        while(status):
            assert img.ndim == 2, 'Image must be grayscale.'
            assert img.dtype == np.float32, 'Image must be float32.'
            H, W = img.shape[0], img.shape[1]
            inp = img.copy()
            inp = (inp.reshape(1, H, W))
            inp = torch.from_numpy(inp)
            inp = torch.autograd.Variable(inp).view(1, 1, H, W)
            yield preprocessing(img)
            img, status = vst.next_frame()
    
    def test_model(model, dataset):
      counter = 0
      loc_error = 0.0
      cosine_sim = 0.0
      for img in tqdm(dataset):
          H, W = img.shape[0], img.shape[1]
          outsf32 = model.execute(img, quantize=False, dequantize=False)
          pts32, desc32, heatmap32 = postprocessing([outsf32[33],outsf32[45]], H, W, conf_thresh= 0.00002)
          outsb8 = model.execute(img, quantize=True, dequantize=True)
          pts8, desc8, heatmap8 = postprocessing([outsb8[33],outsb8[45]], H, W, conf_thresh= 0.0002)
          loc_error += pt_loc_error(pts8.T,pts32.T)
          cosine_sim += desc_cosine_similarity(outsb8[45][0],outsf32[45][0])
          counter += 1
      return (loc_error/counter, cosine_sim/counter)
    
    print("\n==> Testing Accuracy of Quantized Model")
    acc = test_model(G, test_dataset())
    print(f"Detector Loc Error: {acc[0]:.2f}")
    print(f"Descriptor Cosine Similarity: {acc[1]:.2f}")

    G.generate(
        write_constants=True,
        settings={
            "tensor_directory": args.tensors_dir,
            "model_directory": os.path.split(args.at_model_path)[0] if args.at_model_path else "",
            "model_file": os.path.split(args.at_model_path)[1] if args.at_model_path else "ATmodel.c",

            "l3_ram_device": args.ram_type,
            "l3_flash_device": args.flash_type, #"AT_MEM_L3_DEFAULTFLASH",

            "l1_size": 100000,
            "l2_size": 1000000,

            "graph_monitor_cycles": True,
            "graph_produce_node_names": True,
            "graph_produce_operinfos": True,
        }
    )

if __name__ == '__main__':
    main()
