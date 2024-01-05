import os
import numpy as np
import glob
import xml.etree.ElementTree as xET

data_dir = "/home/stu5/ranshui/FasterRCNN/VOC"
class_index_to_name = {
    "background": 0,
    "aeroplane": 1,
    "bicycle": 2,
    "bird": 3,
    "boat": 4,
    "bottle": 5,
    "bus": 6,
    "car": 7,
    "cat": 8,
    "chair": 9,
    "cow": 10,
    "diningtable": 11,
    "dog": 12,
    "horse": 13,
    "motorbike": 14,
    "person": 15,
    "pottedplant": 16,
    "sheep": 17,
    "sofa": 18,
    "train": 19,
    "tvmonitor": 20
}

## used to generatex test.txt and train.txt 
# # get all name of the data
# image_names = glob.glob(os.path.join(data_dir, "JPEGImages", "*.jpg"))
# image_names.sort()
# ## delete all the extension of the file name
# image_names = [os.path.splitext(x)[0] for x in image_names]
# ## split the name of files into year and index via "_"
# image_names = [os.path.basename(x).split("_") for x in image_names]

# ## check the length of the year == 4 and index == 6
# ## only used in debug mode
# # for name in image_names:
# #     assert len(name[0]) == 4
# #     assert len(name[1]) == 6

# ## spilt the trainset and testset
# ## Trainset: year < 2012 && (year == 2012 && index <= 001051)
# ## Testset: year == 2012 && index > 001051
# trainset = []
# testset = []
# for name in image_names:
#     if int(name[0]) < 2012:
#         trainset.append(name)
#     elif int(name[0]) == 2012 and int(name[1]) <= 1051:
#         trainset.append(name)
#     else:
#         testset.append(name)

# print("The number of trainset is %d" % len(trainset))
# print("The number of testset is %d" % len(testset))
# print("The number of total is %d" % len(image_names))
# assert len(trainset) + len(testset) == len(image_names)

# ## save the idx & year of trainset and testset into txt file in the format of str(year) + "_" + str(idx))
# os.makedirs(os.path.join(data_dir, "ImageSets", "Main"), exist_ok=True)

# with open(os.path.join(data_dir, "ImageSets", "Main", "train.txt"), "w") as f:
#     for name in trainset:
#         f.write(name[0] + "_" + name[1] + "\n")

# with open(os.path.join(data_dir, "ImageSets", "Main", "test.txt"), "w") as f:
#     for name in testset:
#         f.write(name[0] + "_" + name[1] + "\n")

# ## get all the annotions of the trainset and testset
# ## the annotation of the trainset and testset is in the format of xml
# annotations = []
# for filename in glob.glob(os.path.join(data_dir, 'Annotations', '*.xml')):
#     xml_file = os.path.join(data_dir, "Annotations", filename)
#     tree = xET.parse(xml_file)

# used to generate Main/*_train.txt and Main/*_test.txt
def parse_xml_and_generate_txt(xml_dir, output_dir, txt_file_path):
    # read the txt file for the name of xml files
    with open(txt_file_path, "r") as txt_file:
        xml_file_names = [line.strip() for line in txt_file.readlines()]

    # read all xml files and record the class info
    for xml_file_name in xml_file_names:
        xml_file = f"{xml_file_name}.xml"
        xml_path = os.path.join(xml_dir, xml_file)
        tree = xET.parse(xml_path)
        root = tree.getroot()

        # create a dict to record the class info
        class_info = {class_index: -1 for class_index in class_index_to_name.values()}

        # find all objects with according name and record the class info
        for obj in root.findall("object"):
            name = obj.find("name").text
            if name in class_index_to_name:
                class_index = class_index_to_name[name]
                class_info[class_index] = 1

        # write "train_.txt" or "test_.txt"
        base_name = os.path.splitext(xml_file_name)[0]
        for class_name, class_index in class_index_to_name.items():
            txt_file = os.path.join(output_dir, f"{class_name}_train.txt")
            with open(txt_file, "a") as f:
                f.write(f"{base_name} {class_info[class_index]}\n")


xml_dir = "/home/stu5/ranshui/FasterRCNN/VOC/Annotations"
output_dir = "/home/stu5/ranshui/FasterRCNN/VOC/ImageSets/Main"
txt_file_path = "/home/stu5/ranshui/FasterRCNN/VOC/ImageSets/Layout/train.txt"
parse_xml_and_generate_txt(xml_dir, output_dir,txt_file_path)