def genSubsets(L):
    if len(L) == 0:
        return [[]]  # list of empty list

    smaller = genSubsets(L[:-1])
    print('smaller:', smaller)
    extra = L[-1:]
    print('extra:', extra)

    new = []
    for small in smaller:
        print('small:', small)
        print('extra:', extra)
        new.append(small + extra)

    print('new', new)
    print('smaller', smaller)
    return smaller + new

if __name__ == "__main__":
    arr = [1,2,3]

    print(genSubsets(arr))



#{}
#0,1,2,3
#0,1 - 0,2 - 0,3 - 1,2 - 1,3 - 2,3
#(0,1,2) (0,1,3), (0,2,3), (1,2,3)
#{0,1,2,3}