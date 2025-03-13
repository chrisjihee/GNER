import subprocess

from chrisbase.io import files

source_file = "generate.py"


for output_file in (
        files("output/ZSE-rerank/roberta-large-*-num=30/**/ZSE-test-pred-*-rerank-*.jsonl") +
        files("output/ZSE-rerank/roberta-large-*-num=40/**/ZSE-test-pred-*-rerank-*.jsonl") +
        []
):
    command = f"""python {source_file} check_possibility {output_file}"""
    command = command.strip().split()
    print("*" * 120)
    print("[COMMAND]", " ".join(command))
    print("*" * 120)

    subprocess.run(command)
    print("\n" * 3)
