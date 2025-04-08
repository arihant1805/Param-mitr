import json

# Replace 'your_notebook.ipynb' with your notebook's filename
input_file = 'Param_Mitr.ipynb'
output_file = 'peft_model.ipynb'

with open(input_file, 'r') as f:
    nb = json.load(f)

# Check if widgets exist in metadata
if 'widgets' in nb.get('metadata', {}):
    # Option A: Add empty state
    if 'state' not in nb['metadata']['widgets']:
        nb['metadata']['widgets']['state'] = {}
    
    # Option B: Remove widgets entirely (uncomment to use instead)
    # del nb['metadata']['widgets']

# Save the fixed notebook
with open(output_file, 'w') as f:
    json.dump(nb, f, indent=2)
