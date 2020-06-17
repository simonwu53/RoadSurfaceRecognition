'''
Paper Title: Road Surface Recognition Based on DeepSense Neural Network using Accelerometer Data
Created by ITS Lab, Institute of Computer Science, University of Tartu
'''

from model import DeepSense
import CrossValidation


if __name__ == '__main__':
    ds = DeepSense.DeepSenseTS(preprocess=True)
    cv = CrossValidation.CV(network=ds)
    cv.create_folds()
    cv.train_on_cv()
