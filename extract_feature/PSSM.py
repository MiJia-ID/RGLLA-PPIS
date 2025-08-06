import os
import subprocess
import numpy as np


def split_fasta(fasta_file, split_dir):
    with open(fasta_file, 'r') as f:
        lines = f.readlines()

    seq_count = 0
    seq_data = []
    seq_id = None
    for line in lines:
        if line.startswith(">"):
            if seq_data:
                single_fasta = os.path.join(split_dir, f"{seq_id}.fasta")
                with open(single_fasta, 'w') as out_f:
                    out_f.writelines(seq_data)
                seq_data = []
            seq_id = line[1:].strip().replace(" ", "_")  # 提取序列ID，去掉">"并替换空格
            seq_data.append(line)
        else:
            seq_data.append(line)

    # 保存最后一个序列
    if seq_data:
        single_fasta = os.path.join(split_dir, f"{seq_id}.fasta")
        with open(single_fasta, 'w') as out_f:
            out_f.writelines(seq_data)


def parse_pssm(pssm_file):
    with open(pssm_file, 'r') as f:
        lines = f.readlines()

    start = False
    matrix = []
    for line in lines:
        if "Last position-specific scoring matrix computed" in line:
            start = True
            continue
        if start:
            if line.strip() == "" or line.startswith("Lambda") or line.startswith("K"):
                continue
            elements = line.split()
            if len(elements) >= 22:  # 确保至少有20列的得分
                try:
                    row = list(map(int, elements[2:22]))
                    matrix.append(row)
                except ValueError:
                    print(f"Skipping line due to parsing error: {line}")
                    continue

    return np.array(matrix)


def run_psiblast(fasta_file, pssm_output_dir):
    base_name = os.path.splitext(os.path.basename(fasta_file))[0]  # 获取文件名的基础名称（无扩展名）

    pssm_file = os.path.join(pssm_output_dir, f"{base_name}.pssm")
    output_file = os.path.join(pssm_output_dir, f"{base_name}_psiblast_output.txt")

    if not os.path.exists(pssm_file):
        # 构建 psiblast 命令
        cmd = [
            "psiblast",
            "-query", fasta_file,
            "-db", "/home/mijia/app/psiblast/swissprot",
            "-num_iterations", "3",
            "-out_ascii_pssm", pssm_file,
            "-out", output_file
        ]

        # 运行命令
        subprocess.run(cmd, check=True)
        print(f"PSI-BLAST completed for {fasta_file}")

        # 解析并保存PSSM矩阵
        #pssm_matrix = parse_pssm(pssm_file)
        #npy_file = os.path.join(pssm_output_dir, f"{base_name}_pssm.npy")
        np.save(pssm_output_dir, pssm_file)
        print(f"Saved PSSM matrix to {pssm_file}")

        # 删除中间生成的文件
        os.remove(fasta_file)  # 删除单序列FASTA文件
        #os.remove(pssm_file)  # 删除PSSM文件
        os.remove(output_file)  # 删除PSI-BLAST输出文件


def main():
    fasta_file = "/home/mijia/egpdi_evolla/af3_data/SJT1.fasta"  # 多序列FASTA文件路径
    split_dir = "/home/mijia/egpdi_evolla/af3_data/PSSM/split_data"  # 存放拆分后单序列FASTA文件的目录
    pssm_output_dir = "/home/mijia/egpdi_evolla/af3_data/PSSM"  # 保存PSSM矩阵的输出目录

    # 创建必要的目录
    os.makedirs(split_dir, exist_ok=True)
    os.makedirs(pssm_output_dir, exist_ok=True)

    # 拆分多序列FASTA文件为单序列文件
    split_fasta(fasta_file, split_dir)

    # 处理拆分后的FASTA文件，生成PSSM矩阵并删除中间文件
    for single_fasta in os.listdir(split_dir):
        if single_fasta.endswith(".fasta"):
            full_path = os.path.join(split_dir, single_fasta)
            run_psiblast(full_path, pssm_output_dir)

    # 删除拆分目录
    os.rmdir(split_dir)
    print("Finished processing all sequences. Intermediate files have been deleted.")


if __name__ == "__main__":
    main()
#用终端执行效果更加

