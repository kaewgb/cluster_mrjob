f = open('E001.lst', 'r')
lines = f.readlines()
output = "["
for l in lines[0:100]:
	l = l.rstrip()
	output += "\'"+l[-13:-4]+"\', "
output = output[:-1]+']'
print output
