from src.llimcobe import Llimcobe as LCB
import numpy as np


class Test(LCB):

    def __init__(self):
        super().__init__()

    def prepare_dataset(self):
        return np.random.rand(25, 25, 10)


def test_execution():
    test = Test()
    assert test is not None
