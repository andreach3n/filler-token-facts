from pathlib import Path
src = Path("/dev/shm/models/DeepSeek-V4-Flash/inference/convert.py")
dst = Path("/dev/shm/models/DeepSeek-V4-Flash/inference/convert_inplace.py")
s = src.read_text()

# 1. Copy each tensor into RAM. safe_open returns memory-mapped views, so without .clone()
#    deleting the source file does not free the RAM disk and the output write runs out of space.
old = "                param: torch.Tensor = f.get_tensor(name)"
new = "                param: torch.Tensor = f.get_tensor(name).clone()  # real copy, not an mmap view"
assert s.count(old) == 1
s = s.replace(old, new)

# 2. Delete each source once its tensors are copied.
old = """                state_dicts[i][name] = new_param

    os.makedirs(save_path, exist_ok=True)"""
new = """                state_dicts[i][name] = new_param
        os.remove(file_path)

    os.makedirs(save_path, exist_ok=True)"""
assert s.count(old) == 1
s = s.replace(old, new)

# 3. Free each rank dict right after saving, and report free space.
old = """        save_file(state_dicts[i], os.path.join(save_path, f"model{i}-mp{mp}.safetensors"))"""
new = """        free_gb = shutil.disk_usage(save_path).free / 1e9
        print(f"saving rank {i}: {free_gb:.0f} GB free on target", flush=True)
        save_file(state_dicts[i], os.path.join(save_path, f"model{i}-mp{mp}.safetensors"))
        state_dicts[i].clear()"""
assert s.count(old) == 1
s = s.replace(old, new)
dst.write_text(s)
print("patched", dst)
