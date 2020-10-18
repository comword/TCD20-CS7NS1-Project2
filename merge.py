import argparse

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('-a', '--alpha', type=str, required=True,
                        help='first file')
    args.add_argument('-b', '--beta', type=str, required=True,
                    help='second file')
    args.add_argument('-o', '--output', type=str, required=True,
                    help='file to output')

    arg_parsed = args.parse_args()
    
    a = list()
    b = list()

    with open(arg_parsed.alpha, "r") as f:
        for line in f:
            line = line.rstrip('\n')
            fname, code = line.split(",")
            a.append((fname, code))
    
    with open(arg_parsed.beta, "r") as f:
        for line in f:
            line = line.rstrip('\n')
            fname, code = line.split(",")
            b.append((fname, code))
    
    for idx, item in enumerate(a):
        for sub in b:
            if item[0]==sub[0]:
                a[idx]=sub

    with open(arg_parsed.output, "w") as f:
        for l in a:
            f.write("%s,%s\n" % (l[0],l[1]))