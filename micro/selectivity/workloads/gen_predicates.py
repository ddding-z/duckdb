workloads = ["medical_charges", "nyc-taxi-green-dec-2016"]
indices = [1, 3, 5, 10, 30, 50, 70, 90, 95, 97, 99]
for w in workloads:
    predicates = None
    with open(f"{w}/predicates-all.txt", "r") as file:
        predicates = [str(line.strip()) for line in file if line.strip() != ""]
    selected_predicates = [predicates[i] for i in indices]
    with open(f"{w}/predicates.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(selected_predicates))