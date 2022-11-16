import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('prompts_file', type=str, help='A prompts file')
parser.add_argument('number_of_prompts', type=int, help='')
args = parser.parse_args()


with open(args.prompts_file,"rb") as f:
    prompts=pickle.load(f)

prompts_former=prompts[:args.number_of_prompts]
prompts_later=prompts[args.number_of_prompts:]

with open(f"prompts/prompts_{len(prompts_former)}.pickle","wb") as f:
    pickle.dump(prompts_former,f,pickle.HIGHEST_PROTOCOL)

with open(f"prompts_{len(prompts_later)}.pickle","wb") as f:
    pickle.dump(prompts_later,f,pickle.HIGHEST_PROTOCOL)