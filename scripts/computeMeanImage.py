import numpy as np
import os
import PIL.Image as img
import argparse

def recursive_search(run_dir):
  #recursive search
  dir_list = os.listdir(run_dir)
  img_list = []
  depth_list = []
  for f in dir_list:
      if os.path.isdir(run_dir + '/' + f):
          f_list = [(f + '/' + temp) for temp in os.listdir(run_dir + '/' + f)]
          dir_list += f_list
      if f.endswith(".png"):
          img_list.append(f)
          continue
      if f.startswith("depth_image") or ("/depth_image" in f):
          depth_list.append(f)
          continue
  return (img_list, depth_list)

def compute_mean(run_dir,f_list):
  N = len(f_list)
  mean_image = None #np.zeros((480,640,3))
  for f in f_list:
      if f.endswith(".png"):
          temp_image = img.open(run_dir + '/' + f).resize((640,480))
          temp_image = np.array(temp_image.getdata()).reshape(temp_image.size[1],temp_image.size[0],3)
          temp_image = temp_image[:,:,0:3]/255.0 #Cut out alpha
          if mean_image is None:
            mean_image = np.zeros(temp_image.shape)
          mean_image += temp_image#np.multiply(np.array(temp_image)#, 1/N)
      if f.startswith("depth_image") or ("/depth_image" in f):
          temp_image = np.load(run_dir + '/' + f)/10000.0
          if mean_image is None:
              mean_image = np.zeros(temp_image.shape)
          mean_image += temp_image
  mean_image = np.multiply(mean_image,1/float(N))
  #im = img.fromarray(np.uint8(mean_image*255))
  #im.show()
  return mean_image

def create_mean_image(folder):
  (img_list,depth_list) = recursive_search(folder)
  color_mean = compute_mean(folder,img_list)
  mean_img_loc = folder + "/mean_color_image.npy"
  np.save(mean_img_loc,color_mean)
  depth_mean = compute_mean(folder,depth_list)
  mean_depth_loc = folder + "/mean_depth_image.npy"
  np.save(mean_depth_loc,depth_mean)
 
def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('data', help='data file') 
  args = parser.parse_args()
  create_mean_image(args.data)

if __name__ == '__main__':
    main()
