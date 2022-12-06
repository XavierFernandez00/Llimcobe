from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy as np
import time
from typing import Callable, Any, TypeVar, Union
import os
import warnings
from threading import Thread


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
                  preprocess: Callable[[np.ndarray], T1],
                  save: Union[Callable[[T1, str], Any], Callable[[T2, str], Any]],
                  load: Callable[[str], np.ndarray]):
        """
        This function includes into a dictionary the model.
        If the name exists in models dictionary the model will be overwritten.

        :param name: string name of the model to be allocated.
        :param model: function that calls the model. If the model is auto-saved, leave it at None.
        :param preprocess: function to modify dataset with model conditions.
        :param save: function to save compressed image. If the model is auto-saved,
                    pass the calling function through this variable.
        :param load: function to load the image and transform to np.ndarray image.
        :return: True if model is included, false if not.
        """
        if (name and model and save and load) or (name and save and load):
            self.models[name] = {"model": model, "preprocess": preprocess, "save": save, "load": load}
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

    def benchmark(self, num_images: int):

        fig, axs = plt.subplots(2, 3)
        all_bpsp = {}
        all_throughput = {}
        all_dthroughput = {}
        lossy_flag = False

        for name in self.models:
            dataset = list(map(self.models[name]["preprocess"], self.dataset))
            compression_throughput = []
            decompression_throughput = []
            bpsp = []
            for (image, length, _) in zip(dataset, self.lens, range(num_images)):
                start_time = time.perf_counter()
                if self.models[name]["model"]:
                    compressed = self.models[name]["model"](image)
                    self.models[name]["save"](compressed, self.temp)
                else:
                    self.models[name]["save"](image, self.temp)
                elapsed_time1 = time.perf_counter() - start_time

                start_time = time.perf_counter()
                loaded = self.models[name]["load"](self.temp)
                elapsed_time2 = time.perf_counter() - start_time
                pixel_size = loaded.dtype.itemsize * 8
                loaded = self.models[name]["preprocess"](loaded)
                if image != loaded and lossy_flag is False:
                    warnings.warn("Pre-compressed image and post-decompressed image don't match")

                    t1 = Thread(target=image.show)
                    t1.start()

                    t2 = Thread(target=loaded.show)
                    t2.start()

                    t1.join()
                    t2.join()
                    lossy_flag = True

                image_size = os.path.getsize(self.temp) * 8
                bpsp.append(float(image_size / length))
                compression_throughput.append(float((length * pixel_size / (8 * 10 ** 6)) / elapsed_time1))
                decompression_throughput.append(float((image_size / (8 * 10 ** 6)) / elapsed_time2))

            all_bpsp[name] = bpsp
            all_throughput[name] = compression_throughput
            all_dthroughput[name] = decompression_throughput

        for name in self.models:
            all_bpsp[name].sort()
            bpsp = all_bpsp[name]
            axs[0, 0].plot(bpsp, label=name)
        axs[0, 0].legend()
        axs[0, 0].set(xlabel="Sorted Images Index", ylabel="Compression Rate [bpsp]")

        for name in self.models:
            all_throughput[name].sort()
            compression_throughput = all_throughput[name]
            axs[0, 1].plot(compression_throughput, label=name)
        axs[0, 1].legend()
        axs[0, 1].set(xlabel="Sorted Images Index", ylabel="Compression Throughput [MB/s]")

        for name in self.models:
            all_dthroughput[name].sort()
            decompression_throughput = all_dthroughput[name]
            axs[0, 2].plot(decompression_throughput, label=name)
        axs[0, 2].legend()
        axs[0, 2].set(xlabel="Sorted Images Index", ylabel="Decompression Throughput [MB/s]")

        markers = ['o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']

        fig.delaxes(axs[1, 0])

        for ix, name in enumerate(self.models):
            axs[1, 1].scatter(sum(all_bpsp[name]) / len(all_bpsp[name]),
                              sum(all_throughput[name]) / len(all_throughput[name]), label=name,
                              marker=markers[ix % len(markers)])
        axs[1, 1].legend()
        axs[1, 1].set(xlabel="Compression Rate [bpsp]", ylabel="Compression Throughput [MB/s]", ylim=0, xlim=0)

        for ix, name in enumerate(self.models):
            axs[1, 2].scatter(sum(all_bpsp[name]) / len(all_bpsp[name]),
                              sum(all_dthroughput[name]) / len(all_dthroughput[name]), label=name,
                              marker=markers[ix % len(markers)])
        axs[1, 2].legend()
        axs[1, 2].set(xlabel="Compression Rate [bpsp]", ylabel="Decompression Throughput [MB/s]", ylim=0, xlim=0)

        # fig.suptitle("Benchmark")
        plt.show()
