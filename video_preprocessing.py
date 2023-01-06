import os
import shutil
# assign directory
raw_directory = 'VIDEO_RGB_SUBSET'
processed_directory = 'Splitted_data'
percentage_train = 0.8
percentage_test = 0.2
amount_classes = len(os.listdir(raw_directory))

for index, class_directory in enumerate(os.listdir(raw_directory)):
    if index + 1 > amount_classes:
        break
    class_directory_raw = os.path.join(raw_directory, class_directory)
    amount_train = int(len(os.listdir(class_directory_raw )) * percentage_train)
    amount_test = int(len(os.listdir(class_directory_raw )) * percentage_test)
    
    amount_index = 0
    for filename in os.listdir(class_directory_raw):
        amount_index +=1
        file = os.path.join(class_directory_raw, filename)
        
        if amount_index < amount_train:
            class_directory_processed = os.path.join(processed_directory, "train", class_directory)
        elif amount_index < amount_train + amount_test:
            class_directory_processed = os.path.join(processed_directory, "test", class_directory)    
        else:
            class_directory_processed = os.path.join(processed_directory, "val", class_directory)
        
        if not os.path.exists(class_directory_processed):
            os.makedirs(class_directory_processed)
        shutil.copy(file,class_directory_processed)


        

