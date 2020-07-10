import os
import PIL.Image as Image

if __name__ == "__main__":
	location = "./simulation"
	os.makedirs("./gifs")
	for dir in os.listdir(location):
		if dir.startswith("Sim"):
			base_dir = location + "/"  + dir + "/"
			for tname in os.listdir(base_dir):
				if tname.startswith("trial"):
					img_dir = base_dir + tname + "/"
					im = []
					for fname in range(0,500):
						img = Image.open(img_dir + "sim_image" + str(fname) + ".png")
						img = img.resize((336,200))
						im.append(img)
					im[0].save('./gifs/'+ str(dir) + "_" + str(tname) +'.gif',save_all=True, append_images=im[1:], duration=10, loop=0,optimize=True, quality=50)
