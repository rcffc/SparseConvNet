import matplotlib.pyplot as plt

def train_loss(path):
    with open(path) as file:
        lines = [s.strip().split(' ') for s in file.readlines() if 'Train loss' in s]
        X = [int(line[0]) for line in lines]
        Y = [float(line[3]) for line in lines]
        plt.plot(X, Y)
        plt.xlabel('Epoch')
        plt.ylabel('Train loss')
        plt.savefig('./train_loss.png')

def iou():
    with open('/app/results/scores.out') as file:
        lines = [s.strip().split('\t') for s in file.readlines() if 'Mean IoU' in s]
        print(lines)
        X = range(188,160,-1)
        Y = [float(line[1]) for line in lines if len(line)==2]
        plt.plot(X, Y)
        plt.xlabel('Epoch')
        plt.ylabel('Val Mean IoU')
        plt.savefig('./val_iou.png')

def accuracy():
    with open('/app/results/scores.out') as file:
        lines = [s.strip().split('\t') for s in file.readlines() if 'Mean Accuracy' in s]
        X = range(188,160,-1)
        Y = [float(line[1]) for line in lines if len(line)==2]
        plt.plot(X, Y)
        plt.xlabel('Epoch')
        plt.ylabel('Val Mean Accuracy')
        plt.savefig('./val_accuracy.png')


# train_loss('/app/results/nohup_finetune_181.out')
# iou()
accuracy()