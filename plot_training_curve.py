import matplotlib.pyplot as plt, matplotlib.patches as mpatches


if __name__ == '__main__':
    num_iterations = []
    val_accs = []
    test_accs = []
    with open('log.txt', 'r') as ins:
        for line in ins:
            iterations, val_acc, test_acc = line.strip().split()
            num_iterations.append(int(iterations))
            val_accs.append(float(val_acc))
            test_accs.append(float(test_acc))

    fig, ax = plt.subplots()
    ax.plot(num_iterations, val_accs, 'b--', num_iterations, test_accs, 'r--')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Accuracy')

    red_patch = mpatches.Patch(color='red', label='Testing One-shot Accuracy')
    blue_patch = mpatches.Patch(color='blue', label='Validation One-shot Accuracy')
    ax.legend(handles=[red_patch, blue_patch])

    plt.axis([0, 300000, 0, 100])
    plt.show()
