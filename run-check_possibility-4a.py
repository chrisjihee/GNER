import subprocess

from chrisbase.io import files

source_file = "generate.py"


for output_file in (
        files("output/ZSE-rerank/ModernBERT-base-num=10/**/ZSE-test-pred-*-rerank-*.jsonl") +
        files("output/ZSE-rerank/ModernBERT-base-num=20/**/ZSE-test-pred-*-rerank-*.jsonl") +
        []
):
    command = f"""python {source_file} check_possibility {output_file}"""
    command = command.strip().split()
    print("*" * 120)
    print("[COMMAND]", " ".join(command))
    print("*" * 120)

    subprocess.run(command)
    print("\n" * 3)
