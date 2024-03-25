with open("thamdu.txt","r+") as f:
    myDatalist = f.readlines()
    print(myDatalist)
    for line in myDatalist:
        print(line)
        a = line.split(",")
        print(a)