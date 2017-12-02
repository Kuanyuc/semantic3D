import os
import numpy as np 
import matplotlib.pyplot as plt 


#calculate label stats for the downloaded semantic dataset 
input_dir = "/home/chen-gogia/Documents/steven/semantic3D/dataset/training/"
filenames = [
"bildstein_station1_xyz_intensity_rgb",
"domfountain_station1_xyz_intensity_rgb",
"sg27_station4_intensity_rgb",
 "domfountain_station2_xyz_intensity_rgb",
 "sg27_station5_intensity_rgb",
"bildstein_station3_xyz_intensity_rgb",
"domfountain_station3_xyz_intensity_rgb",
 "sg27_station9_intensity_rgb",
 "neugasse_station1_xyz_intensity_rgb",
 "sg28_station4_intensity_rgb",
 "bildstein_station5_xyz_intensity_rgb", 
 "sg27_station1_intensity_rgb", 
 "sg27_station2_intensity_rgb", 
 "untermaederbrunnen_station1_xyz_intensity_rgb", 
 "untermaederbrunnen_station3_xyz_intensity_rgb"
    ]

label_count_total = {0 : 0, 
1 : 0,
2 : 0,
3 : 0,
4 : 0,
5 : 0,
6 : 0,
7 : 0,
8 : 0}

# label_count_zero = {0 : 0, 
# 1 : 0,
# 2 : 0,
# 3 : 0,
# 4 : 0,
# 5 : 0,
# 6 : 0,
# 7 : 0,
# 8 : 0}


#read in names of the files 
for file in filenames:
	full_filename = os.path.join(input_dir, file+".labels")
	with open(full_filename, 'r') as f:
		for line in f:
			if int(line)!=0:
				label_count_total[int(line)]+=1;


plt.bar(label_count_total.keys(), label_count_total.values(), 1.0, color='g')
plt.show()

#average label count per file 


#overall label count 
