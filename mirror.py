
file = open("normal_oniFiles/ONI_EXPERTS/backhand/p1.txt","r")
vectors_for_all_frames = []
bp_x_vector = []
bp_y_vector = []
bp_z_vector = []
for line in file:
    if "FRAME: 0" in line:
        pass
    elif "FRAME:" in line or line.strip() == '':
        #we know have 1 frame
        if sum(bp_x_vector)!= 0:
            
            vectors_for_all_frames.append((bp_x_vector,bp_y_vector,bp_z_vector))
        bp_x_vector = []
        bp_y_vector = []
        bp_z_vector = []
    else:
        bp_x,bp_y,bp_z = line.split()
        bp_x_vector.append(float(bp_x))
        bp_y_vector.append(float(bp_y))
        bp_z_vector.append(float(bp_z))



file.close()

frame_to_test = vectors_for_all_frames[5]

import matplotlib.pyplot as plt
import numpy as np


xpoints = frame_to_test[0]



ypoints = frame_to_test[1]
zpoints = frame_to_test[2]
#plt.plot(xpoints,ypoints)
#fig = plt.figure()
#ax = plt.axes(projection='3d')


#ax.scatter3D(xpoints, ypoints, zpoints, cmap='Greens')
plt.scatter(xpoints,ypoints,c='orange')
plt.scatter(new_x_points,ypoints,c='blue')
plt.show()
