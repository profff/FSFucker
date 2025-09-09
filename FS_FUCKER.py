from fsf import *
import argparse
from REV import *
from pprint import pprint


def main():
    parser = argparse.ArgumentParser(description="Process input and output CSV files and enable plotting.")
    parser.add_argument("input_csv", type=str, help="Path to the input CSV file.")
    parser.add_argument("output_csv", type=str, help="Path to the output CSV file.")
    parser.add_argument("--plot", action="store_true", help="Enable plotting if this flag is set.", default=False)
    
    args = parser.parse_args()

    input_csv = args.input_csv
    output_csv = args.output_csv
    enable_plot = args.plot

    print(f"Input CSV: {input_csv}")
    print(f"Output CSV: {output_csv}")
    print(f"Plotting enabled: {enable_plot}")

    (ds_original, units) = read_csv_gps(input_csv)
    pprint(ds_original)

    if enable_plot:
        plot_time_altitude_metrics([("original", ds_original)])


if __name__ == "__main__":
    print_banner("FS-FUCKER tool")
    main()  