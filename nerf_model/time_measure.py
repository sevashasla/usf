import time
import numpy as np
import json

class TimeMeasure:
    def __init__(self):
        self.times = {}

    def save(self, path):
        with open(path, "w") as f:
            f.write(json.dumps(self.times, indent=4))

    def start(self, name):
        if not name in self.times:
            self.times[name] = []
        self.times[name].append(time.time())

    def end(self, name):
        delta = time.time() - self.times[name][-1]
        self.times[name][-1] = delta
    
    def mean(self, name=None):
        if name is None:
            result = {}
            for key, arr in self.times.items():
                result[key] = np.mean(arr)
            return result
        else:
            return np.mean(self.times.get(name, [-1.0]))
        
    
