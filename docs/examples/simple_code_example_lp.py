import sys

import argparse
import torch_geometric

sys.path.append("../fedgraph")
from src.federated_methods import run_LP

torch_geometric.seed.seed_everything(42)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parsing arguments")

    # Add arguments
    parser.add_argument(
        "--method", type=str, default="FedLink", help="Specify the method"
    )
    parser.add_argument(
        "--use_buffer", type=bool, default=False, help="Specify whether to use buffer"
    )
    parser.add_argument(
        "--buffer_size", type=int, default=300000, help="Specify buffer size"
    )
    parser.add_argument(
        "--online_learning", type=bool, default=False, help="Specify online learning"
    )
    parser.add_argument(
        "--global_rounds", type=int, default=20, help="Specify global rounds"
    )
    parser.add_argument(
        "--local_steps", type=int, default=3, help="Specify local steps"
    )
    parser.add_argument(
        "--repeat_time", type=int, default=10, help="Specify repeat time"
    )
    parser.add_argument(
        "--global_file_path", type=str, default="data_seperated_by_country/raw_Checkins_anonymized_five_countries.txt"
    )
    parser.add_argument(
        "traveled_file_path", type=str, default="traveled_users.txt", help="Specify traveled file path"
    )
    parser.add_argument(
        "--record_results",
        type=bool,
        default=False,
        help="Record model AUC and Running time",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Specify device")

    # Parse the arguments
    args = parser.parse_args()
    print(args)

    country_codes = [
        # "US",
        # "BR",
        # "ID",
        "TR",
        # "JP",
    ]  # top 5 biggest country , 'US', 'BR', 'ID', 'TR',

    run_LP(args=args, country_codes=country_codes)
