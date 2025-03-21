import os

directory = "/volumn/Retree_exp/workloads"

code_to_append = '''\n
with open(f"/volumn/Retree_exp/queries/Retree/workloads/workload_models.csv", "a", encoding="utf-8") as f:
    f.write(f"{data_name},{model_name}\\n")
'''

def remove_last_n_lines(file_path, n):
    with open(file_path, 'r', encoding="utf-8") as file:
        lines = file.readlines()
    lines = lines[:-n]

    with open(file_path, 'w', encoding="utf-8") as file:
        file.writelines(lines)

for root, dirs, files in os.walk(directory):
    for file in files:
        if file.startswith("train_") and file.endswith(".py"):
            file_path = os.path.join(root, file)
            remove_last_n_lines(file_path, 5)
            with open(file_path, "a", encoding="utf-8") as f:
                f.write(code_to_append)
