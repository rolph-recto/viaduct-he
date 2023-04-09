#!/usr/bin/env python3

import argparse
import csv
import functools
import re
import subprocess
import sys
from pathlib import Path
import time
import io
import statistics
import math

compile_benchmarks = {
    "distance": [
        {
            "name": "e1-o0",
            "size": 4096,
            "epochs": 1,
            "opt_duration": 0,
        },
        {
            "name": "e2-o0",
            "size": 4096,
            "epochs": 2,
            "opt_duration": 0,
        },
    ]
}

# benchmarks = ["retrieval-256", "distance"]
exec_benchmarks = {
    "distance": ["baseline"]
}

def compile_dummy(args):
    pass

def bench_compile(args):
    timestamps = []
    benchmarks = sorted(exec_benchmarks.keys())
    for trial in range(args.trials):
        timestamp = str(int(time.time()))
        timestamps.append(timestamp)

        out_dir = Path(args.out_path, timestamp)
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"starting trial {trial+1}")
        print(f"storing trial data in {out_dir}")

        for bench in benchmarks:
            for cfg in compile_benchmarks[bench]:
                cfg_name = cfg["name"]
                benchfile = Path(args.in_path, f"{bench}.tlhe")
                size = cfg["size"]
                opt_duration = cfg["opt_duration"]
                epochs = cfg["epochs"]
                
                print(f"compiling {cfg_name}-{bench}")

                cmd = f"./he_vectorizer -b pyseal -s {size} -e {epochs} -d {opt_duration} {benchfile}"

                compile_proc = subprocess.run(
                    cmd.split(),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=True,
                    encoding="utf-8",
                    env={"RUST_LOG": "he_vectorizer=info"}
                )

                out_filename = Path(out_dir, f"{cfg_name}-{bench}.txt")
                out_filename.touch(exist_ok=False)
                with open(out_filename, "w") as out_file:
                    out_file.write(compile_proc.stdout)

        print(f"finished trial {trial+1}")
            

def bench_exec(args):
    print("benchmarking execution time")

    timestamps = []
    benchmarks = sorted(exec_benchmarks.keys())
    for trial in range(args.trials):
        timestamp = str(int(time.time()))
        timestamps.append(timestamp)

        print(f"starting trial {trial+1} at {out_dir}")

        for bench in benchmarks:
            bench_input = Path(in_dir, f"input-{bench}.json")

            if not bench_input.is_file():
                raise Exception(f"input file {bench_input} does not exist")

            with open(bench_input) as input_file:
                input_str = input_file.read()

                for cfg in exec_benchmarks[bench]:
                    cfg_bench = f"{cfg}-{bench}"
                    cfg_bench_file = Path(in_dir, f"{cfg_bench}.py")
                    if not bench_input.is_file():
                        raise Exception(f"benchmark file {bench_input} does not exist")

                    print(f"running {cfg_bench}...")

                    bench_proc = subprocess.run(
                        ["python3", cfg_bench_file],
                        input=input_str,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        check=True,
                        encoding="utf-8"
                    )

                    out_filename = Path(out_dir, f"{cfg_bench}.txt")
                    out_filename.touch(exist_ok=False)
                    with open(out_filename, "w") as out_file:
                        out_file.write(bench_proc.stdout)

                        print(f"dumped output to {out_filename}")
                        time.sleep(1)

        print(f"finished trial {trial+1}")

    print("created trial directories from {} to {}".format(timestamps[0], timestamps[-1]))


def collect_exec(args):
    out_dir = Path(args.out_path)
    if not out_dir.is_dir():
        raise Exception(f"output directory {out_dir} does not exist")

    start = None if args.from_dir is None else int(args.from_dir)
    end = None if args.to_dir is None else int(args.to_dir)
    trials = []
    for path in out_dir.glob("*"):
        if path.is_dir():
            path_time = int(path.stem)

            within_start = True if start is None else start <= path_time
            within_end = True if end is None else path_time <= end
            if within_start and within_end:
                trials.append(path)

    if len(trials) == 0:
        print("no trials found in that time interval")
        return

    n = len(trials)
    csv_out = io.StringIO()
    writer = csv.DictWriter(csv_out, fieldnames=["bench","cfg","trials","exec_time","sterror","error_pct"])
    benchmarks = sorted(exec_benchmarks.keys())
    for bench in benchmarks:
        for cfg in exec_benchmarks[bench]:
            exec_times = []
            for trial in trials:
                cfg_bench = Path(trial, f"{cfg}-{bench}.txt")
                if not cfg_bench.is_file():
                    raise Exception(f"expected output file {cfg_bench} does not exist")

                with open(cfg_bench) as cfg_bench_file:
                    cfg_bench_out = cfg_bench_file.read()
                    res = re.search("exec duration: (?P<time>[0-9]+)ms", cfg_bench_out)
                    exec_time = int(res.group("time"))
                    exec_times.append(exec_time)

            assert(len(exec_times) == n)
            mean = statistics.mean(exec_times)
            stdev = statistics.stdev(exec_times)
            sterror = stdev / math.sqrt(n)
            writer.writerow({
                "bench": bench,
                "cfg": cfg,
                "trials": n,
                "exec_time": round(mean, 2),
                "sterror": round(sterror, 2),
                "error_pct": round(sterror / mean, 2)
            })

    print(csv_out.getvalue())


def argument_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="COMMAND", required=True)

    # benchmark compile data
    bench_compile_parser = subparsers.add_parser("bench-compile", help="benchmark compilation time")
    bench_compile_parser.set_defaults(func=bench_compile)

    bench_compile_parser.add_argument(
        "-t", "--trials", dest="trials", type=int, default=1,
        help="number of times to compile each benchmark")

    bench_compile_parser.add_argument(
        "-i", "--inpath", dest="in_path", type=str, default="benchmarks",
        help="base path to retrieve benchmarks")

    bench_compile_parser.add_argument(
        "-o", "--outpath", dest="out_path", type=str, default="bench-compile",
        help="base path to store trial information")

    # benchmark execution data
    bench_exec_parser = subparsers.add_parser("bench-exec", help="benchmark execution time")
    bench_exec_parser.set_defaults(func=bench_exec)
    bench_exec_parser.add_argument(
        "-t", "--trials", dest="trials", type=int, default=1,
        help="number of times to run each benchmark")

    bench_exec_parser.add_argument(
        "-o", "--outpath", dest="out_path", type=str, default="bench-exec",
        help="base path to store trial information")

    bench_exec_parser.add_argument(
        "-i", "--inpath", dest="in_path", type=str, default="benchmarks",
        help="base path to retrieve benchmarks")

    # collect execution data
    collect_exec_parser = subparsers.add_parser("collect-exec", help="collect execution data")
    collect_exec_parser.set_defaults(func=collect_exec)

    collect_exec_parser.add_argument(
        "-f" "--from", dest="from_dir", type=str,
        help="timestamp of start folder")

    collect_exec_parser.add_argument(
        "-t" "--to", dest="to_dir", type=str,
        help="timestamp of end folder")

    collect_exec_parser.add_argument(
        "-o", "--outpath", dest="out_path", type=str, default="bench-exec",
        help="base path to retrieve trial information")

    return parser


if __name__ == "__main__":
    arguments = argument_parser().parse_args()
    arguments.func(arguments)

