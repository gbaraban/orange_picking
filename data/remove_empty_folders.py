import os
import shutil

empty_folders = []

def check_empty(dir):
	if os.path.isdir(dir):
		ld = os.listdir(dir)
		if ld == []:
			empty_folders.append(dir)
			return
		for l in ld:
			check_empty(dir + "/" + l)

	else:
		return

if __name__ == "__main__":
	check_empty("./")
	print(empty_folders)
	destination = "./empty_folders/"
	for source in empty_folders:
		if source == "empty_folder":
			continue
		shutil.move(source, destination + source)
