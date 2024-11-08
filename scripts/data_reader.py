

if __name__ == "__main__":
    import numpy
    import pickle
    import lzma
    import matplotlib.pyplot as plt

    with lzma.open("record_0.npz", "rb") as file:
        data = pickle.load(file)

        print("Read", len(data), "snapshotwas")
        print(data[0].image)
        # print([e.current_controls for e in data])

        if data[0].image is not None:
            plt.imshow(data[0].image)
            plt.show()
