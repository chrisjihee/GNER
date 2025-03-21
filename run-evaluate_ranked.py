import subprocess

from chrisbase.io import files

source_file = "generate.py"


for output_file in (
        files("output/ZSE-rerank/deberta-v3-large-ep1-lr1e-5/**/ZSE-test-sampled-N700-*-rerank-*.jsonl") +
        files("output/ZSE-rerank/roberta-large-ep1-lr1e-5/**/ZSE-test-sampled-N700-*-rerank-*.jsonl") +
        files("output/ZSE-rerank/roberta-large-ep1-lr2e-5/**/ZSE-test-sampled-N700-*-rerank-*.jsonl") +
        files("output/ZSE-rerank/roberta-large-ep1-lr3e-5/**/ZSE-test-sampled-N700-*-rerank-*.jsonl") +
        files("output/ZSE-rerank/roberta-base-max_sampled=3/**/ZSE-test-sampled-N700-*-rerank-*.jsonl") +
        files("output/ZSE-rerank/deberta-v3-base-max_sampled=3/**/ZSE-test-sampled-N700-*-rerank-*.jsonl") +
        []
):
    command = f"""python {source_file} check_possibility {output_file}"""
    command = command.strip().split()
    print("*" * 120)
    print("[COMMAND]", " ".join(command))
    print("*" * 120)

    subprocess.run(command)
    print("\n" * 3)
