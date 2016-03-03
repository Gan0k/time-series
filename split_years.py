#/usr/bin/env python

import fileinput

print("Start...")
i=0
last_year = -1
f = None
for line in fileinput.input():
    if i != 0:
        idx = line.rfind('/')+1
        year=line[idx:idx+4]
        if year != last_year:
            if f is not None:
                f.close()
            f = open(year+".txt","w")
            last_year = year
            f.write(first_line)

        f.write(line)

    else:
        first_line = line
        i = i+1
print(i)
