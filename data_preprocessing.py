# python program to keep only 100000 lines in a cleaned_Laptops.csv file
def ModifyFile():
    with open("cleaned_Laptops.csv", "r") as f:
        lines = f.readlines()
    with open("cleaned_Laptops.csv", "w") as f:
        for line in lines[:100000]:
            f.write(line)

#main
if __name__ == "__main__":
    ModifyFile()
