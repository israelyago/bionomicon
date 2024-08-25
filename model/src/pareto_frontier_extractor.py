import pandas as pd
import requests
import csv
import os


def get_url(hash):
    return f"http://localhost:6007/data/plugin/scalars/scalars?tag=Validation%2FF1&run={hash}&format=csv"


def tb_data(log_dir):
    trials = os.listdir(log_dir)
    trials = [trial for trial in trials if "-" not in trial]
    fdf = {}
    with open("pareto_frontier.csv", "w", newline="") as csv_file:
        writer = csv.writer(csv_file, delimiter=",")
        writer.writerow(("Trial", "F1", "Wall time"))
        for trial in trials:
            r = requests.get(
                get_url(trial),
                allow_redirects=True,
            )
            data = r.text
            data_csv = csv.reader(data.splitlines())
            data_csv = list(data_csv)
            _headers = data_csv[0]
            data = data_csv[1:]
            wall_time_start = float(data[0][0])
            wall_time_end = float(data[-1][0])
            wall_time = wall_time_end - wall_time_start
            last_f1 = float(data[-1][2])
            writer.writerow((trial, last_f1, wall_time))


runs_dir = input("Runs directory:")
tb_data(runs_dir)
