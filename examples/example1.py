# # coding: utf-8
# import matplotlib.pyplot as plt
# from evalys.jobset import JobSet

# #matplotlib.use('WX')

# js = JobSet.from_csv('/Users/vhafener/repos/lanl-ccu/evalys/examples/jobs.csv')
# print(js.df.describe())

# js.df.hist()

# fig, axe = plt.subplots()
# js.gantt()
# plt.show()

from evalys.jobset import JobSet
import matplotlib.pyplot as plt

js = JobSet.from_csv("/Users/vhafener/repos/lanl-ccu/evalys/examples/jobs.csv")
js.plot(with_details=True)
plt.show()