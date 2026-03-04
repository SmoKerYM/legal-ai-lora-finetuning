import json

# Load the notebook
with open("sft_lora_A100.ipynb", "r") as f:
    nb = json.load(f)

# Remove the problematic widgets metadata
if "widgets" in nb.get("metadata", {}):
    del nb["metadata"]["widgets"]

# Save it back
with open("sft_lora_A100.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

print("Done — widgets metadata removed")