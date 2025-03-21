import os

directory = "/volumn/Retree_exp/workloads"

code_to_append = '''

with open(f"/volumn/Retree_exp/queries/Retree/workloads/workload_models.csv", "a", encoding="utf-8") as f:
    f.write(f"{data_name},{model_name}")
'''

for root, dirs, files in os.walk(directory):
    for file in files:
        if file.startswith("train_") and file.endswith(".py"):
            file_path = os.path.join(root, file)
            with open(file_path, "a", encoding="utf-8") as f:
                f.write(code_to_append)
