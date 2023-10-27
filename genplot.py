import re
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

import re

def extract_info_from_line(line):
    match = re.match(r".*Iters: (\d+)/(\d+) \|\| Lr: (\d+\.\d+) \|\| Task Loss: (\d+\.\d+) \|\| KD Loss: (\d+\.\d+).*", line)
    if match:
        iters_current, iters_total, lr, task_loss, kd_loss = match.groups()
        return int(iters_current), int(iters_total), float(lr), float(task_loss), float(kd_loss)
    else:
        return None
    
#Create two subplot --> this would be easier to view probably
def plot_loss(file_name) :
    f = open("./local_logs/{}.txt".format(file_name),"r")
    lines = f.readlines()
    f.close()
    x = []
    y_task = []
    y_kd = []
    for line in lines :
        if "Task" in line :
            iters_current, _, __, task_loss, kd_loss = extract_info_from_line(line)
            x.append(iters_current)
            y_task.append(task_loss)
            y_kd.append(kd_loss)
    
    fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 1]})
    plt.rcParams['lines.linewidth'] = 0.5  # Adjust line width
    # Plot the first line on the first subplot
    axs[0].plot(x, y_kd, color='red', label='kd loss')
    axs[0].set_title('KD Loss - KL Divergence')
    axs[0].set_xlabel('Iterations')
    axs[0].set_ylabel('Loss')
    axs[0].legend()

    axs[1].plot(x, y_task, color='green', label='task loss')
    axs[1].set_title('Task Loss - Cross Entropy')
    axs[1].set_xlabel('Iterations')
    axs[1].set_ylabel('Loss')
    axs[1].legend()

    # plt.plot(x, y_kd, marker='', color = 'red', label = 'kd loss')
    # plt.plot(x, y_task, marker='', color = 'green', label = 'task loss')
    # plt.title('mIoU vs. Iterations')
    # plt.xlabel('Iterations')
    # plt.ylabel('mIoU')
    plt.tight_layout()
    plt.savefig("./graphs/{}_loss.png".format(file_name))



def plot_mIoU(file_name) :
    f = open("./local_logs/{}.txt".format(file_name),"r")
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

    max_index = np.argmax(y[1:])
    max_x = x[max_index+1]
    max_y = y[max_index+1]

    plt.annotate(f'{max_y:.2f}', xy=(max_x, max_y), xytext=(max_x, max_y + 1.5),
                arrowprops=dict(facecolor='green', shrink=0.5))

    plt.plot(x, y, marker='')
    plt.title('mIoU vs. Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('mIoU')
    plt.savefig("./graphs/{}_mIoU.png".format(file_name))

def plot_combined(file_name) :
    f = open("./local_logs/{}.txt".format(file_name),"r")
    lines = f.readlines()
    f.close()
    fig = plt.figure(figsize=(10, 6))
    gs = GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 2])
    plt.rcParams['lines.linewidth'] = 0.5  # Adjust line width
    # Create subplots
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])
    x = []
    y_task = []
    y_kd = []
    for line in lines :
        if "Task" in line :
            iters_current, _, __, task_loss, kd_loss = extract_info_from_line(line)
            x.append(iters_current)
            y_task.append(task_loss)
            y_kd.append(kd_loss)

    # Plot the first line on the first subplot
    ax1.plot(x, y_kd, color='red', label='kd loss')
    ax1.set_title('KD Loss - KL Divergence')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Loss')
    ax1.legend()

    ax2.plot(x, y_task, color='green', label='task loss')
    ax2.set_title('Task Loss - Cross Entropy')
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('Loss')
    ax2.legend()
    # Extracting data
    x = []
    y = []
    for line in lines:
        if "mIoU" in line :
            match = re.search(r"(\d+) mIoU = (\d+\.\d+)", line)
            if match:
                x.append(int(match.group(1)))
                y.append(float(match.group(2)))

    max_index = np.argmax(y[1:])
    max_x = x[max_index+1]
    max_y = y[max_index+1]
    ax3.plot(x, y, color='blue', label='mIoU')
    ax3.annotate(f'{max_y:.2f}', xy=(max_x, max_y), xytext=(max_x, max_y + 0.75),
                arrowprops=dict(facecolor='green', shrink=0.5))
    ax3.set_title('mIoU vs Iterations')
    ax3.set_xlabel('Iterations')
    ax3.set_ylabel('mIoU')

    plt.tight_layout()
    plt.savefig("./graphs/{}_combined.png".format(file_name))


file_list = [
            # "lr0001",
            #  "decay200",
            #  "decay500",
            #  "decay1000",
            #  "task_lambda",
            #  "task_lambda_200",
            #  "task_lambda_4000_iters",
            #  "b2_basic",
            #  "irregular_lr",
            # "b1_b0_025",
            # "b1_b0_50",
            # "b3_b0_25",
            "b2_b0_50",
            "b2_b0_50_lr_lower",
            "b3_b0_50",
            "b1_b0_50_lowered_lr"
             ]

for file in file_list :
    # plot_loss(file)
    # plot_mIoU(file)
    plot_combined(file)

#scp aaryang@crcv.eecs.ucf.edu:/home/aaryang/experiments/CIRKD/logs/pretrained/normalized/kd_l1_pretrained_b0_cityscapes_batch_4_lr_0.0001_decay_200_log.txt decay200.txt
#scp aaryang@crcv.eecs.ucf.edu:/home/aaryang/experiments/CIRKD/logs/pretrained/normalized/kd_l1_pretrained_b0_cityscapes_batch_4_lr_0.0001_log.txt lr0001.txt
#scp aaryang@crcv.eecs.ucf.edu:/home/aaryang/experiments/CIRKD/logs/pretrained/normalized/kd_l1_pretrained_b0_cityscapes_batch_4_lr_0.0001_decay_500_log.txt decay500.txt
#scp aaryang@crcv.eecs.ucf.edu:/home/aaryang/experiments/CIRKD/logs/pretrained/normalized/kd_l1_pretrained_b0_cityscapes_batch_4_lr_5e-05_decay_1000_log.txt decay1000.txt

#scp aaryang@crcv.eecs.ucf.edu:/home/aaryang/experiments/CIRKD/logs/pretrained/normalized/kd_l1_pretrained_b0_cityscapes_batch_8_lr_0.0001_decay_1_task_lambda_0.25_log.txt task_lambda.txt
#scp aaryang@crcv.eecs.ucf.edu:/home/aaryang/experiments/CIRKD/logs/pretrained/normalized/kd_l1_pretrained_b0_cityscapes_batch_4_lr_0.0001_decay_200_task_lambda_0.25_log.txt task_lambda_200.txt
#scp aaryang@crcv.eecs.ucf.edu:/home/aaryang/experiments/CIRKD/logs/pretrained/normalized/kd_l1_pretrained_b0_cityscapes_batch_8_lr_0.0001_decay_1_task_lambda_0.25_iters_40000_log.txt task_lambda_4000_iters.txt

#scp aaryang@crcv.eecs.ucf.edu:/home/aaryang/experiments/CIRKD/logs/pretrained/normalized/kd_b2_pretrained_b0_cityscapes_batch_6_lr_0.0001_decay_1_task_lambda_0.25_iters_35000_log.txt b2_basic.txt
#scp aaryang@crcv.eecs.ucf.edu:/home/aaryang/experiments/CIRKD/logs/pretrained/irregular/kd_l1_pretrained_b0_cityscapes_batch_8_lr_0.0001_decay_1_task_lambda_0.25_iters_40000_log.txt irregular_lr.txt

#scp aaryang@crcv.eecs.ucf.edu:/home/aaryang/experiments/CIRKD/logs/pretrained/normalized/kd_b1_pretrained_b0_cityscapes_batch_6_lr_0.0001_decay_1_task_lambda_0.25_iters_35000_log.txt b1_b0_025.txt
#scp aaryang@crcv.eecs.ucf.edu:/home/aaryang/experiments/CIRKD/logs/pretrained/normalized/kd_b1_pretrained_b0_cityscapes_batch_6_lr_0.0001_decay_1_task_lambda_0.5_iters_35000_log.txt  b1_b0_50.txt
#scp aaryang@crcv.eecs.ucf.edu:/home/aaryang/experiments/CIRKD/logs/pretrained/normalized/kd_b3_pretrained_b0_cityscapes_batch_8_lr_0.0001_decay_1_task_lambda_0.25_iters_35000_log.txt b3_b0_25.txt


#scp aaryang@crcv.eecs.ucf.edu:/home/aaryang/experiments/CIRKD/logs/pretrained/normalized/kd_b2_pretrained_b0_cityscapes_batch_6_lr_0.0001_decay_1_task_lambda_0.5_iters_35000_log.txt b2_b0_50.txt
#scp aaryang@crcv.eecs.ucf.edu:/home/aaryang/experiments/CIRKD/logs/pretrained/normalized/kd_b2_pretrained_b0_cityscapes_batch_6_lr_5e-05_decay_1_task_lambda_0.5_iters_35000_log.txt b2_b0_50_lr_lower.txt
#scp aaryang@crcv.eecs.ucf.edu:/home/aaryang/experiments/CIRKD/logs/pretrained/normalized/kd_b3_pretrained_b0_cityscapes_batch_8_lr_0.0001_decay_1_task_lambda_0.5_iters_35000_log.txt b3_b0_50.txt
#scp aaryang@crcv.eecs.ucf.edu:/home/aaryang/experiments/CIRKD/logs/pretrained/normalized/kd_b1_pretrained_b0_cityscapes_batch_6_lr_1e-05_decay_1_task_lambda_0.5_iters_35000_log.txt b1_b0_50_lowered_lr.txt
