total = [0, 6, 27, 120, 534, 2256, 8969, 33058, 114149, 360508, 930588, 1350852, 782536, 90280, 276]

avg = 0
for i in range(len(total)):
    avg = avg + total[i]*i

avg = avg/sum(total)
print(avg)