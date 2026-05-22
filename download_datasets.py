import libero.libero.utils.download_utils as download_utils
from libero.libero import get_libero_path

download_dir = get_libero_path("datasets")
datasets = "all" # Can specify "all", "libero_goal", "libero_spatial", "libero_object", "libero_100"

datasets_default_path = get_libero_path("datasets")
benchmark_root_path = get_libero_path("benchmark_root")
init_states_default_path = get_libero_path("init_states")
bddl_files_default_path = get_libero_path("bddl_files")
print("Default benchmark root path: ", benchmark_root_path)
print("Default dataset root path: ", datasets_default_path)
print("Default bddl files root path: ", bddl_files_default_path)


libero_datasets_exist = download_utils.check_libero_dataset(download_dir=download_dir)

if not libero_datasets_exist:
    download_utils.libero_dataset_download(download_dir=download_dir, datasets=datasets)