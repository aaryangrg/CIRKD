import re
import matplotlib.pyplot as plt


f = open("./local_logs/lr0001.txt","r")
lines = f.readlines()
f.close()

# Extracting data
x = []
y = []
for line in lines:
    if "mIoU" in line :
        match = re.search(r"(\d+) mIoU = (\d+\.\d+)", line)
        if match:
            x.append(int(match.group(1)))
            y.append(float(match.group(2)))

# Plotting the graph
plt.plot(x, y, marker='o')
plt.title('mIoU vs. Iterations')
plt.xlabel('Iterations')
plt.ylabel('mIoU')
# plt.show()
plt.savefig("./graphs/lr0001.png")


#scp aaryang@crcv.eecs.ucf.edu:/home/aaryang/experiments/CIRKD/logs/pretrained/normalized/kd_l1_pretrained_b0_cityscapes_batch_4_lr_0.0001_decay_200_log.txt decay200.txt
#scp aaryang@crcv.eecs.ucf.edu:/home/aaryang/experiments/CIRKD/logs/pretrained/normalized/kd_l1_pretrained_b0_cityscapes_batch_4_lr_0.0001_log.txt lr0001.txt
#scp aaryang@crcv.eecs.ucf.edu:/home/aaryang/experiments/CIRKD/logs/pretrained/normalized/kd_l1_pretrained_b0_cityscapes_batch_4_lr_0.0001_decay_500_log.txt decay500.txt
#scp aaryang@crcv.eecs.ucf.edu:/home/aaryang/experiments/CIRKD/logs/pretrained/normalized/kd_l1_pretrained_b0_cityscapes_batch_4_lr_5e-05_decay_1000_log.txt decay1000.txt