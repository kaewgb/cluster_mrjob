dataset = 'E001'
f = open(dataset+'.lst', 'r')
lines = f.readlines()
output = "["
for l in lines[0:10]:
	l = l.rstrip()
	#output += "\'"+l[-13:-4]+"\', "
	output += "\'"+dataset+"/"+l[-13:-4]+"\' "
output = output[:-1]+']'
print output
