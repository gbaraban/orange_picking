import os

def main():
	all_data = open("./simulation_data.csv","r")
	trials = 10
	ftrial = []
	for i in range(trials):
		ftrial.append(open("./gifs/sim/trial"+str(i)+"/sim_data.csv","x"))

	first_line = ""
	for i, line in enumerate(all_data):
		if i == 0:
			first_line = line
			for ftr in ftrial:
				ftr.write(first_line)
			continue
		info = line.split(",")
		ftrial[int(info[3])].write(line)

	for i in range(len(ftrial)):
		ftrial[i].close()


if __name__ == "__main__":
	main()
