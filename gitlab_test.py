import sys

def main():
    f = open("metrics.txt")
    s = f.readline()
    if float(s.split()[2]) < 0.9:
        sys.exit("accuracy is lower than treshold")

if __name__ == "__main__":
    main()
