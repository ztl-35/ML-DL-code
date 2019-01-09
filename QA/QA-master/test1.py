import matplotlib.pyplot as plt

X = ['1', '2','3','4','5']
Y = [0.28200000000000003, 0.18, 0.766, 0.47400000000000003, 0.5720000000000001]
fig = plt.figure()
plt.bar(X, Y, 0.4, color="green")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("５代的准确率训练集结果")

plt.show()
plt.savefig("barChart.jpg")