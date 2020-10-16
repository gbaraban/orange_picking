import os
import pickle
import argparse

def main():
        bag_order = {}
        parser = argparse.ArgumentParser()
        parser.add_argument('bag_dir', help="bag dir") 
        args = parser.parse_args()

        bag_list = os.listdir(args.bag_dir)
        #print(bag_list)

        for bag_name in bag_list:
                if not bag_name.endswith(".bag"):
                        continue
                print(bag_name)
                bag_order[bag_name] = len(bag_order)
        print(bag_order)
        pickle.dump(bag_order, open("bag_order_lambda.pickle", "wb"))


if __name__ == "__main__":
    main()
