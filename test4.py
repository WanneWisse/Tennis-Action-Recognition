import os
path = "C:/Users/Wanne/OneDrive/Documenten/tennis-recog/project/normal_oniFiles - kopie/ONI_AMATEURS/serviceflat/servicekick/"

for filename in os.listdir(path):
    my_dest =filename.split(".")[0] + "s1.txt" 
    my_source =path + filename
    my_dest =path + my_dest
    os.rename(my_source, my_dest)

