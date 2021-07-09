import os


loc = "./simulation/"
versions = {}
dist = {}

for dir in os.listdir(loc):
	if dir.startswith("data_simulation"):
		sp = dir.strip().split("_")
		sim = sp[2]
		ver = sp[3] + "_" + sp[4]
		trial = int(sp[5].lstrip("trial"))
		if ver not in versions:
			versions[ver] = {}
			dist[ver] = {}
			for i in range(1,5):
				versions[ver]["Sim"+str(i)] = 0
				dist[ver]["Sim"+str(i)] = [-9999]*10

		f = open(loc + dir, "r")
		success = False
		for line in f:
			l = line.strip("\n").strip().split(" ")
			print(l[1] + "asdf")
			if l[1] != "None":
				success = True
			dist[ver][sim][trial] = l[2]

		if success:
			versions[ver][sim] += 1


f = open("info.csv", "w")
for ver in versions:
	st = ver
	for i in range(1,5):
		st += "," + str(versions[ver]["Sim"+str(i)])
	st += "\n"
	f.write(st)
f.close()

f = open("data_info.csv", "w")
for ver in dist:
	st = ver
	for i in range(1,5):
		st += "," + str(dist[ver]["Sim"+str(i)])
	st += "\n"
	f.write(st)
f.close()

