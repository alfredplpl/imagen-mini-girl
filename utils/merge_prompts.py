import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--prompts', required=True, nargs="*", type=str, help='a list of prompts files')
args = parser.parse_args()

merged_prompts=[]
for file in args.prompts:
    with open(file,"rb") as f:
        prompts=pickle.load(f)
    merged_prompts.extend(prompts)

with open(f"prompts/prompts_{len(merged_prompts)}.pickle","wb") as f:
    pickle.dump(merged_prompts,f,pickle.HIGHEST_PROTOCOL)
