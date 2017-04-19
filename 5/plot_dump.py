import matplotlib.pyplot as plt
import seaborn

d_range = list()
accuracy = list()

with open('dump.txt') as f:
	for line in f:
		d, acc = map(float, line.rstrip().split(':'))
		d_range.append(d)
		accuracy.append(acc)

plt.plot(d_range, accuracy)
plt.figure(figsize=(15,10))
plt.plot(d_range, accuracy)
plt.ylabel('Accuracy')
plt.xlabel('droptout rate')
plt.savefig('result/dropout_accuracy.png')