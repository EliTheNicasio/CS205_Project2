import numpy as np


def featureSearch(data):
    print(data.shape)


def main():
    dataSmall = np.loadtxt('CS205_small_testdata__12.txt', dtype=np.single)
    # dataLarge = np.loadtxt('CS205_large_testdata__45', dtype=np.single)
    featureSearch(dataSmall)


if __name__ == '__main__':
    main()
