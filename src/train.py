import matplotlib.pyplot as plt
from dataset import Custom_Food101
from model import GAN


def main():
    print("=> Script started")
    print("=> Loading dataset...")
    food_dataset = Custom_Food101()
    food_dataset.prepare_data()
    print("=> Dataset loaded")
    data_loader = food_dataset.get_dataloader()
    gan = GAN()
    print("=> Training started")
    gan.training_loop(data_loader, debug_mode=True, debug_grad=False)
    print("=> Training finished")

    print("=> Generating example images...")
    print("Donuts generation...")
    grid = gan.generate(31)
    plt.axis("off")
    plt.imshow(grid)
    plt.show()

    # print("Hot dogs generation...")
    # grid = gan.generate(55)
    # plt.axis("off")
    # plt.imshow(grid)
    # plt.show()
    # print("Apple pies generation...")
    # grid = gan.generate(0)
    # plt.axis("off")
    # plt.imshow(grid)
    # plt.show()


if __name__ == "__main__":
    main()
