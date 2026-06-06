import glob
import json
results = []
final = [{} for i in range(20)]

import os
path = os.path.join("atc_asr_output", "predictions", "predictions_run*.json")

for x in glob.glob(str(path)):
    with open(x) as f:
        j = json.load(f)
    for i in range(20):
        if len(final[i]) == 0:
            final[i] = {
                "id":j[i].pop("id"),
                # "reference_raw":j[i].pop("reference_raw"),
                "reference": j[i].pop("reference"),
                "model_result": [j[i]]
            }
        else:
            j[i].pop("id")
            # j[i].pop("reference_raw")
            j[i].pop("reference")
            final[i]["model_result"].append([j[i]])


with open("top_20_predictions.json", mode="w") as f:
    f.write(json.dumps(final))