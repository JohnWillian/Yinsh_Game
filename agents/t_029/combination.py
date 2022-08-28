# INFORMATION ------------------------------------------------------------------------------------------------------- #


# Author:  Yu Zhang
# Date:    24/05/2022
# Purpose: Generate all possible combinations, use as H value.


# IMPORTS AND CONSTANTS ----------------------------------------------------------------------------------------------#


import csv

def getHeuristicValue(id,comb):
    if id == 1:
        count = 2
    else:
        count = 0
    
    empty = "0" 
    ring = str(1 + count)
    piece = str(2 + count)
    o_ring = str(3 - count)
    o_piece = str(4 - count)

    if comb.count(ring) > 4:
        return 5
    elif comb.count(o_ring) > 0:
        return 51
    elif comb.count(empty) > 4:
        return 6
    elif comb.count(o_piece) == 0:
        return comb.count(ring) + comb.count(empty)
    else:
        if comb.count(ring) == 0:
            if comb.count(o_piece) != 5:
                return comb.count(o_piece) + comb.count(empty)
            else:
                return 51
        else:
            return comb.count(ring) + comb.count(empty)
            

def combiniation():
    csv_file = open("agents/t_029/combination.csv",
                    'a', encoding='utf-8', newline='')

    write = csv.writer(csv_file)
    # header = ('id','comb','h-value')
    for p0 in range(5):
        for p1 in range(5):
            for p2 in range(5):
                for p3 in range(5):
                    for p4 in range(5):
                        temp = str(p0) + str(p1) + str(p2) + str(p3) + str(p4)
                        # Consider multiple players
                        h_value_0 = getHeuristicValue(0, temp)
                        h_value_1 = getHeuristicValue(1, temp)
                        write.writerow(("0", temp, h_value_0))
                        write.writerow(("1", temp, h_value_1))
    csv_file.close()


combiniation()
