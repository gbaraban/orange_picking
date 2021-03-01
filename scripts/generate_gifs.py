import os
import argparse
import PIL.Image as Image


def generate_gif(fname, loc="/home/gabe/ws/ros_ws/src/orange_picking/data/simulation/", gifs="/home/gabe/ws/ros_ws/src/orange_picking/gifs/"):
        if not os.path.isdir(gifs):
                os.makedirs(gifs)

        img_dir = loc + fname + "/"
        im = []
        for iname in range(0,500):
                #print(img_dir + "sim_image" + str(iname) + ".png")
                #print(os.path.isfile(img_dir + "sim_image" + str(iname) + ".png"))
                if os.path.isfile(img_dir + "sim_image" + str(iname) + ".png"):
                        img = Image.open(img_dir + "sim_image" + str(iname) + ".png")
                        img = img.resize((336,200))
                        im.append(img)
                else:
                        break

        if len(im) > 0:
                temp_name = fname.strip("data/simulation/").replace("/","_")
                im[0].save(gifs + temp_name + '_sim.gif',save_all=True, append_images=im[1:], duration=10, loop=0,optimize=True, quality=100)

        im = []
        for iname in range(0,500):
                if os.path.isfile(img_dir + "ext_image" + str(iname) + ".png"):
                        img = Image.open(img_dir + "ext_image" + str(iname) + ".png")
                        img = img.resize((336,200))
                        im.append(img)
                else:
                        break

        if len(im) > 0:
                temp_name = fname.strip("data/simulation/").replace("/","_")
                im[0].save(gifs + temp_name + '_ext.gif',save_all=True, append_images=im[1:], duration=10, loop=0,optimize=True, quality=100)


def generate_gif_data(fname, loc="/home/gabe/ws/ros_ws/src/orange_picking/data/simulation/", gifs="/home/gabe/ws/ros_ws/src/orange_picking/gifs/"):
        #messy way to distinguish, fix
        if not os.path.isdir(gifs):
                os.makedirs(gifs)

        img_dir = loc + fname + "/"
        im = []
        for iname in range(0,500):
                #print(img_dir + "sim_image" + str(iname) + ".png")
                #print(os.path.isfile(img_dir + "sim_image" + str(iname) + ".png"))
                if os.path.isfile(img_dir + "image" + str(iname) + ".png"):
                        img = Image.open(img_dir + "image" + str(iname) + ".png")
                        img = img.resize((336,200))
                        im.append(img)
                else:
                        break

        if len(im) > 0:
                run_num = loc.strip().strip("/").rsplit('/',1)[1]
                temp_name = run_num + "_" + fname.strip().strip("/").strip()
                print(temp_name)
                im[0].save(gifs + temp_name + '.gif',save_all=True, append_images=im[1:], duration=10, loop=0,optimize=True, quality=100)


def generate_gifs_dataset(loc, gifs):
    print("Generating Dataset gifs")
    if not os.path.isdir(gifs):
        os.makedirs(gifs)

    for dir in os.listdir(loc):
        print(dir)
        count = 0
        for file in os.listdir(loc + "/" + dir):
            if file.startswith("image") and file.endswith("png"):
                count += 1

        print(count)
        im = []
        for i in range(count):
            if os.path.isfile(loc + "/" + dir + "/image" + str(i) + ".png"):
                img = Image.open(loc + "/" + dir + "/image" + str(i) + ".png")
                img = img.resize((336,200))
                im.append(img)
            else:
                break

        if len(im) > 0:
                run_num = dir
                temp_name = run_num
                print(temp_name)
                im[0].save(gifs + "/" + temp_name + '.gif',save_all=True, append_images=im[1:], duration=10, loop=0,optimize=True, quality=100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('fname', help='folder name for imgs')
    parser.add_argument('--loc', type=str, default="/home/gabe/ws/ros_ws/src/orange_picking/data/simulation/", help='folder to find above fname')
    parser.add_argument('--gifs', type=str, default="/home/gabe/ws/ros_ws/src/orange_picking/gifs/", help='folder to save gifs in')
    parser.add_argument('--data_collection', type=int, default=0)
    parser.add_argument('--dataset', type=bool, default=False)
    args = parser.parse_args()

    if args.dataset:
        generate_gifs_dataset(args.fname, args.gifs)

    else:
        if args.data_collection == 0:
            generate_gif(args.fname, loc=args.loc, gifs=args.gifs)
        else:
            generate_gif_data(args.fname, loc=args.loc, gifs=args.gifs)
