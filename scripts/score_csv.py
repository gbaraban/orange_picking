import os


def main(loc):
	dest_csv = open("simulation_data.csv","x")
	dest_csv.write("Model Name,Iterations,Hz,Trial,Score\n")
	lsdir = os.listdir(loc)
	for dir in lsdir:
		if dir.startswith("data_simulation"):
			f = open(loc + "/" + dir ,"r")
			info = dir.strip("data_simulation_").split("_")
			st = ""
			for inf in info:
				if inf.startswith("trial"):
					inf = inf.strip("trial")
				st += str(inf) + ","

			for line in f:
				st += line.strip("\n").strip() + "\n"

			dest_csv.write(st)


if __name__ == "__main__":
	loc = "./score/"
	main(loc)
