workload_models = {
    "medical_charges" : "medical_charges_d10_l945_n1889_20250321150635",
    "nyc-taxi-green-dec-2016" : "nyc-taxi-green-dec-2016_d10_l857_n1713_20250321151140"
} 
for w, m in workload_models.items():
    predicates = None
    lines = []
    with open(f"{w}/predicates.txt", "r") as file:
        predicates = [str(line.strip()) for line in file if line.strip() != ""]
    for p in predicates:
        lines.append(f"{w},{m},{p}")
    with open(f"workload_models.csv", "a") as file:
        file.write("\n".join(lines))
        file.write("\n")