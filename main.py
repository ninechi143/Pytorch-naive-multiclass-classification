import torch
import os
import argparse

from trainer import multiclass_trainer


def parse_args():

    parser = argparse.ArgumentParser(description='multiclass classification trainer')

    parser.add_argument("-l" , "--lr" , type = float , default=1e-3)
    parser.add_argument('--batch_size' , type=int , default=128)
    parser.add_argument('--epochs' , type=int , default=15)
    parser.add_argument('--optimizer' , type=str , default='adam')
    parser.add_argument("--normalize" , action= "store_true" , default=True)
    parser.add_argument("--resume" , type=str , default = None)

    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = parse_args()
    # print(args.batch_size)
    # print(args.normalize)


    trainer = multiclass_trainer(args)

    trainer.load_data()  # prepare dataset and dataLoader
    trainer.setup()      # define our model, loss function, and optimizer
    trainer.train()      # define training pipeline and execute the training process
    trainer.save()       # save model after training

    print("\nDone.\n")
