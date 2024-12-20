# -*- coding: utf-8 -*-

import torch
import unittest

from numpy.testing import assert_equal
from numpy.testing import assert_raises
from numpy.testing import assert_warns

from fedgraph.train_func import accuracy


class TestFunc(unittest.TestCase):

    def test_accuracy(self):
        output = torch.tensor([[0.1, 0.9], [0.2, 0.8]])
        labels = torch.tensor([1, 0])
        acc = accuracy(output, labels)
        assert (acc == .5)
