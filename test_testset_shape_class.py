from CrazyflieSimulationPython.cfSimulator import zTest
from ControlBasedTesting.shapeTestSet import shapeTestSet
from ControlBasedTesting.NLthUpperbound import NLthUpperbound
import matplotlib.pyplot as plt


nlth = NLthUpperbound(0.5, 0.5, 1, 5)
nlth.add_sample(1,    4,   5)
nlth.add_sample(1.5,  4,   5)
nlth.add_sample(2.5,  3.5, 4.5)
nlth.add_sample(2.25, 3,   4)
nlth.add_sample(3.25, 2.5, 2.6)
nlth.add_sample(4,    3,   3.5)
nlth.add_sample(5,    0.9, 1)

sts = shapeTestSet(zTest, 'trapezoidal', nlth, 0.01)

# sts.plot_test(2, 1)
# sts.plot_test(sts.t_max, sts.a_max)
# sts.plot_test(sts.t_min, sts.a_max)

sts.generate_test_set()
sts.plot_test_set()

plt.show()
