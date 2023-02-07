# Llimcobe
Lossless Image Compression Benchmark

The library has a single class called LCB that must be used as a super of your benchmark class.
The LCB class forces you to create a function called ``prepare_dataset()`` that will return a list of images in HxWxC numpy.ndarray format.

To include a model on which to perform the benchmark, you must call the ``set_model()`` function. 
To create the benchmark, you must call the ``benchmark()`` function.


### Example of use:
 ```python
from llimcobe import LCB
import numpy as np
from PIL import Image
import pickle


class Benchmark(LCB):
    def __init__(self):
        super().__init_()

    def prepare_dataset(self):
        with open("dataset", "rb") as fo:
            dataset = pickle.load(fo, encoding='bytes')

            return [image.reshape(3,32,32).transpose for image in dataset]

if __name__ == "__main__":
    benchmark = Benchmark()

    def save(format):
        return lambda image, path: image.save(path,format=format, lossless=True)
    
    preprocess = lambda image: Image.fromarray(image)
    load = lambda path: np.assarray(Image.open(path))

    benchmark.set_model("PNG", model=None, 
                    preprocess=preprocess,
                    save=save("PNG"),
                    load=load,
                    compare=None)

    benchmark.set_model("WebP", model=None,
                    preprocess=preprocess,
                    save=save("WebP"),
                    load=load,
                    compare=None)
    
    benchmark.benchmark(1000)
```