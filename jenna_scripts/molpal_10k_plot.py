from scripts.experiment import Experiment


result_path = "molpal_10k"


def make_figure(path):
    a = Experiment(path)
    return a


if __name__ == "__main__":
    experiment = make_figure(result_path)
    print(len(experiment))
