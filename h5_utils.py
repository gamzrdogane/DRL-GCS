import h5py
import argparse


def print_h5_info(file_path):
    """
    Prints the structure and attributes of an HDF5 file.
    """
    with h5py.File(file_path, 'r') as f:
        print(f"HDF5 File: {file_path}")
        print("\nStructure:")
        
        def print_structure(name, obj):
            if isinstance(obj, h5py.Group):
                print(f"\n{name}: {type(obj)}")
            else:
                print(f" - {name}: \n    type: {type(obj)}")

            if isinstance(obj, h5py.Dataset):
                print(f"   shape: {obj.shape}, dtype: {obj.dtype}")
                for key, value in obj.attrs.items():
                    print(f"   attr[{key}]: {value}")
            elif isinstance(obj, h5py.Group):
                for key, value in obj.attrs.items():
                    print(f"   attr[{key}]: {value}")
        
        f.visititems(print_structure)


# External Call
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print HDF5 file structure and attributes.")
    parser.add_argument("file_path", type=str, help="Path to the HDF5 file.")
    args = parser.parse_args()

    print_h5_info(args.file_path)