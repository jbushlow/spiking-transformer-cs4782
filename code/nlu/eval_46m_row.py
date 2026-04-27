import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

from nlu.test_cls import test_task


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sst2-checkpoint", required=True)
    parser.add_argument("--sst5-checkpoint", required=True)
    parser.add_argument("--mr-checkpoint", required=True)
    parser.add_argument("--subj-checkpoint", required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--ctx_len", type=int, default=1024)
    args = parser.parse_args()

    results = {
        "SST-2": test_task("sst2", args.sst2_checkpoint, args.batch_size, args.ctx_len) * 100,
        "SST-5": test_task("sst5", args.sst5_checkpoint, args.batch_size, args.ctx_len) * 100,
        "MR": test_task("mr", args.mr_checkpoint, args.batch_size, args.ctx_len) * 100,
        "Subj": test_task("subj", args.subj_checkpoint, args.batch_size, args.ctx_len) * 100,
    }

    print("\nTable-ready SpikeGPT 46M row")
    print("Method | Spiking | Recurrent | Complexity per layer | SST-2 | SST-5 | MR | Subj")
    print(
        "SpikeGPT 46M | yes | yes | O(T·d) | "
        f"{results['SST-2']:.2f} | {results['SST-5']:.2f} | {results['MR']:.2f} | {results['Subj']:.2f}"
    )


if __name__ == "__main__":
    main()
