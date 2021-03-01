import os
import json 
import matplotlib.pyplot as plt

# Github: https://github.com/sujitmandal
# Pypi : https://pypi.org/user/sujitmandal/
# LinkedIn : https://www.linkedin.com/in/sujit-mandal-91215013a/ 

if not os.path.exists('plot/modelPerformance'):
    os.mkdir('plot/modelPerformance')

def ploModel(modelPerformance):
    plt.figure(figsize=(15,8))
    plt.title('Mean Values')
    plt.bar(modelPerformance.get('mean').keys(), modelPerformance.get('mean').values())
    plt.xlabel('Model Name')
    plt.ylabel('Model Mean Values')
    plt.grid(True)
    plt.savefig('plot/modelPerformance/Mean.pdf')
    plt.show(block=False)
    plt.pause(5)
    plt.close()
    
    plt.figure(figsize=(15,8))
    plt.title('Standard Deviation Values')
    plt.bar(modelPerformance.get('std').keys(), modelPerformance.get('std').values())
    plt.xlabel('Model Name')
    plt.ylabel('Model Standard Deviation Values')
    plt.grid(True)
    plt.savefig('plot/modelPerformance/Std.pdf')
    plt.show(block=False)
    plt.pause(5)
    plt.close()

    plt.figure(figsize=(15,8))
    plt.title('Mean vs. Standard Deviation')
    plt.bar(modelPerformance.get('mean').keys(), modelPerformance.get('mean').values())
    plt.bar(modelPerformance.get('std').keys(), modelPerformance.get('std').values())
    plt.xlabel('Model Name')
    plt.ylabel('Model Standard Deviation Values')
    plt.grid(True)
    plt.savefig('plot/modelPerformance/MeanStd.pdf')
    plt.pause(5)
    plt.close()