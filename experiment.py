import json
import datetime as dt
import os

from utils import create_dir


class Experiment():
    def __init__(self, modelname, folder="experiments"):
        self.config = {}
        date = dt.datetime.today()
        string_date = "_".join([
            str(date.year).zfill(4),
            str(date.month).zfill(2),
            str(date.day).zfill(2),
            str(date.hour).zfill(2),
            str(date.minute).zfill(2)
        ])
        self.folder = os.path.join(folder, string_date, modelname)
        create_dir(self.folder)

        self.configname = self.allpath("config.json")
        self.modelname = self.allpath(modelname + "_weights.hdf5")
        self.pnggraph = self.allpath("graph.png")
        self.pngloss = self.allpath("loss.png")
        self.pngacc = self.allpath("acc.png")
        
    def allpath(self, path):
        return os.path.join(self.folder, path)

    def addconfig(self, configname, theconfig):
        self.config[configname] = theconfig

    def saveconfig(self, verbose=False):
        conf = json.dumps(self.config, sort_keys=True, indent=4)
        with open(self.configname, "w") as f:
            f.write(conf)
        if verbose:
            print("The experiment is written in " + self.folder)


if __name__ == "__main__":
    exp = Experiment(modelname="testmodel")
    exp.addconfig("lr", 0.1)
    exp.addconfig("stop_thinking", [5, 5, 33])

    exp.saveconfig()
