import json
from copy import deepcopy

file1 = r'../temp_jsons_to_merge/2018_2022_wet_scene_aoi_merge1.json'
file2 = r'../temp_jsons_to_merge/2018_2022_wet_scene_aoi (2).json'
outfile = r'../temp_jsons_to_merge/2018_2022_wet_scene_aoi_merge2.json'

json1 = json.load(open(file1))
json2 = json.load(open(file2))

# initiate an export dictionary using json1 data
out_dict = deepcopy(json1)

# for keys in json1 that are present in json2
for key in json1.keys():
    if key in json2.keys():
        out_dict[key] = list(set(json1[key] + json2[key]))

# for keys in json2 that are not present in json1
for key in json2.keys():
    if not key in json1.keys():
        out_dict[key] = json2[key]

with open(outfile, 'w') as f:
    f.write(
        json.dumps(
            out_dict, indent=4)
    )
    f.write('\n')