from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy as np
import time
from typing import Callable, Any, TypeVar, Union
import os


class Llimcobe(ABC):
    def __init__(self):
        super().__init__()
        self.models = {}
        self.dataset = self.prepare_dataset()
        self.lens = list(map(np.size, self.dataset))
        temp = os.path.join(os.path.abspath(__file__), "..", ".temp")
        if not os.path.exists(temp):
            os.makedirs(temp)
        self.temp = os.path.join(temp, "img")



    @abstractmethod
    def prepare_dataset(self):
        """
        This abstract method will be used to prepare dataset.
        :return: A list of images in NxMx3 numpy.ndarray format.
        """
        pass

    T1 = TypeVar('T1')
    T2 = TypeVar('T2')

    def set_model(self, name: str, model: Union[None, Callable[[T1], T2]],
                  preprocess: Callable[[np.ndarray], T1], save: Union[Callable[[T1, str], Any], Callable[[T2, str], Any]]):
        """
        This function includes into a dictionary the model.
        If the name exists in models dictionary the model will be overwritten.

        :param name: string name of the model to be allocated.
        :param model: function that calls the model. If the model is auto-saved, leave it at None.
        :param preprocess: function to modify dataset with model conditions.
        :param save: function to save compressed image. If the model is auto-saved,
                    pass the calling function through this variable.
        :return: True if model is included, false if not.
        """
        if (name and model and save) or (name and save):
            self.models[name] = {"model": model, "preprocess": preprocess, "save": save}
            return True

        return False

    def get_model(self, name):
        """
        This function will return the model if exists or False if not exists.
        :param name: The name of the model
        :return: The model if exists, False if not.
        """
        if name in self.models.keys():
            if self.models[name]["model"]:
                return self.models[name]["model"]
            return self.models[name]["save"]
        return False

    def del_model(self, name):
        """
        This function will delete the model if exists.
        :param name: The name of the model
        :return: True if the model is deleted or false if not.
        """
        if name in self.models.keys():
            del self.models[name]
            return True
        return False

    def benchmark(self):
        # TODO:
        # - Decoding throughput
        # - Lossless fail detection

        plt.subplots(figsize=(5, 2.7), layout='constrained')
        all_bpsp = {}
        all_throughput = {}

        for name in self.models:
            dataset = list(map(self.models[name]["preprocess"], self.dataset))
            compression_throughput = []
            bpsp = []
            for (image, length) in zip(dataset, self.lens):
                start_time = time.perf_counter()
                if self.models[name]["model"]:
                    compressed = self.models[name]["model"](image)
                    self.models[name]["save"](compressed, self.temp)
                else:
                    self.models[name]["save"](image, self.temp)
                elapsed_time = time.perf_counter() - start_time
                size = os.path.getsize(self.temp) * 8
                bpsp.append(float(size/length))
                compression_throughput.append(float((size/(8*10**6))/elapsed_time))

            all_bpsp[name] = bpsp
            all_throughput[name] = compression_throughput

        plt.subplot(131)
        for name in self.models:
            all_bpsp[name].sort()
            bpsp = all_bpsp[name]
            plt.plot(bpsp, label=name)
        plt.legend()
        plt.ylabel("Compression Rate [bpsp]")
        plt.xlabel("Sorted Images Index")

        plt.subplot(132)
        for name in self.models:
            all_throughput[name].sort()
            compression_throughput = all_throughput[name]
            plt.plot(compression_throughput, label=name)
        plt.legend()
        plt.ylabel("Compression Throughput [MB/s]")
        plt.xlabel("Sorted Images Index")

        plt.subplot(133)
        markers = ['o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']
        for ix, name in enumerate(self.models):
            plt.scatter(sum(all_bpsp[name])/len(all_bpsp[name]), sum(all_throughput[name])/len(all_throughput[name]), label=name, marker=markers[ix % len(markers)])
        plt.ylim(0)
        plt.xlim(0)
        plt.legend()
        plt.ylabel("Compression Throughput [MB/s]")
        plt.xlabel("Compression Rate [bpsp]")
        plt.suptitle("Benchmark")
        plt.show()
