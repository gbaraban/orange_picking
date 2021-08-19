import numpy as np
import os
import PIL.Image as img
import argparse

def compute_mean_image(run_dir):
  #num_points = model.num_points
  #time_window = num_points*dt
  #Assume reduced N is constant for all trials
  trial_list = os.listdir(run_dir)
  N = 0
  for trial in trial_list:
    if os.path.isdir(run_dir + '/' + trial):
      image_list = os.listdir(run_dir + '/' + trial)
      N += len(image_list) - 1
  print(N)
  mean_image = None #np.zeros((480,640,3))
  for trial in trial_list:
    if os.path.isdir(run_dir + '/' + trial):
      image_list = os.listdir(run_dir + '/' + trial)
      for image in image_list:
        if 'png' in image:
          #print(image)
          temp_image = img.open(run_dir + '/' + trial + '/' + image).resize((640,380))
          temp_image = np.array(temp_image.getdata()).reshape(temp_image.size[1],temp_image.size[0],3)
          print(temp_image.shape)
          temp_image = temp_image[:,:,0:3]/255.0 #Cut out alpha
          if mean_image is None:
            mean_image = np.zeros(temp_image.shape)
          mean_image += temp_image#np.multiply(np.array(temp_image)#, 1/N)
  print(np.max(mean_image))
  mean_image = np.multiply(mean_image,1/N)
  print(np.max(mean_image))
  im = img.fromarray(np.uint8(mean_image*255))
  im.show()
  return mean_image

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('data', help='data file') 
  args = parser.parse_args()
  data_loc_name = args.data.strip("..").strip(".").strip("/").replace("/", "_")
  mean_img_loc = args.data + "../mean_img_" + data_loc_name + '.npy' 
  print(mean_img_loc)
  mean_image = compute_mean_image(args.data)
  np.save(mean_img_loc,mean_image)

if __name__ == '__main__':
    main()
