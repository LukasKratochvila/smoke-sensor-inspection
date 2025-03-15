import os
import math
import pandas
import numpy as np
import matplotlib.pyplot as plt
import argparse
"""
Script for creating learning curves.
"""
 
def main(args):
    log_csv=pandas.read_csv(args.file, header=None, sep="brake")
    log_csv=log_csv.values.tolist()
    #print(log_csv[0][0])
    trainLines = [line[0] for line in log_csv if "Epoch:" in line[0] and "eta" in line[0]]

    fileName = os.path.basename(args.file).split(".")[0]
    modelName = [mName.split("'")[-2] for mName in log_csv[0][0].split(",") if "model" in mName][0]
    datasetName = [dName.split("'")[-2] for dName in log_csv[0][0].split(",") if "data_path" in dName][0]
    printFreq = int([pFreq.split("=")[-1] for pFreq in log_csv[0][0].split(",") if "print_freq" in pFreq][0])
    batchCount = int(trainLines[0].split("]  [")[-1].split("]")[0].split("/")[-1])
    nParts = math.ceil(batchCount/printFreq)
    if (batchCount % printFreq != 1):
        nParts += 1
    #print(nParts)
    
    datasetName = datasetName.replace("/","-")

    #print(log_csv[21][0].split("  ")) 'loss', 'bbox_regression', 'classification'
    log_loss = [[float(line.split("  ")[4].split(" ")[1]), float(line.split("  ")[5].split(" ")[1]), float(line.split("  ")[6].split(" ")[1])] 
                for line in trainLines]
    log_prec = [float(line[0].split(" = ")[-1]) for line in log_csv if "Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]" in line[0]]
    log_rec = [float(line[0].split(" = ")[-1]) for line in log_csv if "Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]" in line[0]]
    
    #print(log_loss)
    na=np.array(log_loss)
    ne=np.arange(0, len(log_loss)/nParts, 1/nParts)
    #print(na, ne)
    
    num_p=np.array(log_prec)
    num_r=np.array(log_rec)
    #print(num_p, num_r)

    _, ax = plt.subplots(figsize=(12, 12))
    ax.plot(ne, na[:,0])
    ax.plot(ne, na[:,1])
    ax.plot(ne, na[:,2])
    ax.autoscale(True)
    ax.set_title("Train curve for experiment: {0} on {1} dataset".format(fileName, datasetName))
    ax.set_xlabel("Epoch[-]")
    ax.set_ylabel("[-]")
    ax.set_yscale("log")
    ax.legend(['loss', 'bbox_regression', 'classification'])
    
    _, ax2 = plt.subplots(figsize=(12, 12))
    ax2.plot(ne[::nParts], num_p)
    ax2.plot(ne[::nParts], num_r)
    ax2.autoscale(True)
    ax2.set_title("Validation precission and recal for experiment: {0} on {1} dataset".format(fileName, datasetName))
    ax2.set_xlabel("Epoch[-]")
    ax2.set_ylabel("[-]")
    #ax2.set_yscale("log")
    ax2.legend(['Average Precision (AP)', 'Average Recall (AR)'])

    # Get model with maximum precision with step index
    max_prec = np.max(num_p, 0)
    max_step = np.where(num_p == max_prec)[0][0]
    print(f"Best model at: {max_step} step with {max_prec}[-] precision")

    if args.save:
        ax.figure.savefig('training_curve_{}_{}_{}'.format(modelName, fileName, datasetName[:-1]), bbox_inches='tight', pad_inches=0.1)
        ax2.figure.savefig('ap_val_curve_{}_{}_{}'.format(modelName, fileName, datasetName[:-1]), bbox_inches='tight', pad_inches=0.1)
        with open(os.path.join("model_select.csv"), "w") as f:
            f.write("type, step, precision\n")
            f.write(f"best, {max_step}, {max_prec}\n")
    else:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog = 'plot.py',
                    description = 'Script for plotting training curves.',
                    epilog = 'Folder is required.')
    parser.add_argument('-f', '--file', type=str, help='path_to_log_file', required=True) # "../log/biscuit_23_9" #"../log/my_test"
    parser.add_argument("--save", help="Save figures", action="store_true")
    args = parser.parse_args()

    main(args)
