import subprocess

from chrisbase.io import files

source_file = "predict.py"

for predction_file in files("output/ZSE-predict/ZSE-validation-pred-*.jsonl")[:2]:
    command = f"""
        python
            {source_file} evaluate
                --predction_file {predction_file}
    """
    command = command.strip().split()
    print("*" * 120)
    print("[COMMAND]", " ".join(command))
    print("*" * 120)

    subprocess.run(command)
    print("\n" * 3)
