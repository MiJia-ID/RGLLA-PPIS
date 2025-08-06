from openbabel import pybel


def cif_to_pdb(cif_file, pdb_file):
    # 读取CIF文件
    mol = pybel.readfile("cif", cif_file).__next__()

    # 写入PDB文件
    mol.write("pdb", pdb_file)
    print(f"Converted {cif_file} to {pdb_file}")


# 使用示例
cif_file = "/home/mijia/RGCNPPIS/Dataset/pdb_dir/1acb/1acb_model.cif"  # CIF文件路径
pdb_file = "/home/mijia/RGCNPPIS/Dataset/pdb_dir/1acb/1acb.pdb"  # 输出PDB文件路径

cif_to_pdb(cif_file, pdb_file)
