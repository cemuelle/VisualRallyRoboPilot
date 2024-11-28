if __name__ == "__main__":
    import numpy
    import pickle
    import lzma
    import matplotlib.pyplot as plt
    import numpy as np

    with lzma.open("./data/cyan/record_cyan_0.npz", "rb") as file:
        data = pickle.load(file)

        print("Read", len(data), "snapshotwas")
        print(data[0].image)
        # print([e.current_controls for e in data])

        print(type(data[0].image))

        if data[0].image is not None:
            plt.imshow(data[0].image)
            plt.show()

            plt.imshow(np.fliplr(data[0].image))
            plt.show()
