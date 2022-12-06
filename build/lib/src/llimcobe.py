from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy


class Llimcobe(ABC):
    def __init__(self):
        super().__init__()
        self.models = {}
        self.dataset = self.prepare_dataset()

    @abstractmethod
    def prepare_dataset(self):
        """
        This abstract method will be used to prepare dataset.
        :return: A list of images in NxMx3 numpy.ndarray format.
        """
        pass

    def preprocess_dataset(self):
        pass

    def set_model(self, name, model, preprocess):
        """
        This function includes into a dictionary the model.
        If the name exists in models dictionary the model will be overwritten.

        :param name: string name of the model to be allocated.
        :param model: function that calls the model.
        :param preprocess: function to modify dataset with model conditions.
        :return: True if model is included, false if not.
        """
        if name and model:
            self.models[name] = {"model": model, "preprocess": preprocess}
            return True

        return False

    def get_model(self, name):
        """
        This function will return the model if exists or False if not exists.
        :param name: The name of the model
        :return: The model if exists, False if not.
        """
        if name in self.models.keys():
            return self.models[name]["model"]
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
        for image in self.dataset:
            pass
        print(self.models["png"])

