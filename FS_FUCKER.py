from fsf import *
import argparse
from REV import *
from pprint import pprint
import msvcrt


def main():
    parser = argparse.ArgumentParser(description="Process input and output CSV files and enable plotting.")
    parser.add_argument("input_folder", type=str, help="Path to the input folder containing CSV files.")
    parser.add_argument("--plot", action="store_true", help="Enable plotting if this flag is set.", default=False)
    
    args = parser.parse_args()

    input_folder = args.input_folder
    enable_plot = args.plot

    print(f"Input folder: {input_folder}")
    print(f"Plotting enabled: {enable_plot}")
    ds_original = []

    for f in os.listdir(input_folder):
        if not os.path.isfile(f"{input_folder}/{f}"):
            print(f"Error: File '{f}' does not exist.")
            return

        (ds, units) = read_csv_gps(f"{input_folder}/{f}")
        ds_original.append((f,ds))
    pprint(ds_original)
    if enable_plot:
        plot_time_altitude_metrics(ds_original)


if __name__ == "__main__":
    print_banner("FS-FUCKER tool")
    main()  