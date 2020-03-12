import torch

#TODO: Add custom image transformations here

class pointToBins(object):
    def __init__(self,min_list,max_list, bins):
        self.min_list = min_list
        self.max_list = max_list
        self.bins = bins

    def __call__(self,points):
        local_pts = []
        for ctr, point in enumerate(points):
            #Convert into bins
            min_i = self.min_list[ctr]
            max_i = self.max_list[ctr]
            bin_nums = (point - min_i)/(max_i-min_i)
            bin_nums_scaled = (bin_nums*self.bins).astype(int)
            bin_nums = np.clip(bin_nums_scaled,a_min=0,a_max=model.bins-1)
            local_pts.append(bin_nums)
        return local_pts

class GaussLabels(object):
    def __init__(self,mean,stdev,bins):
        self.mean = mean
        self.stdev = stdev
        self.bins = bins

    def __call__(self,bin_list):
        label_list = []
        for bins in bin_list:#TODO: Double check this
            labels = np.zeros((3,self.bins))
            for j in range(len(bins)):#
                for i in range(labels.shape[1]):
                    labels[j][i] = mean * (np.exp((-np.power(bin_nums[j]-i, 2))/(2 * np.power(stdev, 2))))
            label_list.append(labels)
        label_list = np.array(label_list)
        label_list.resize((model.num_points,3,model.bins))
        return label_list

