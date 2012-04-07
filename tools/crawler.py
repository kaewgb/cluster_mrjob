def findHVC(lines):
	for l in lines:
		if l[0:3] == "HVC":
			return l.strip()
	raise ValueError("Cannot find testset name\n")

def update_timedict(timedict, t):
	if t[-3:] == 'sec':
		timedict['sec'] = int(t[:-3])
	elif t[-4:] == 'mins':
		timedict['mins'] = int(t[:-4])
	elif t[-3:] == 'hrs':
		timedict['hrs'] = int(t[:-3])
	else:
		raise ValueError(t)

from urllib import urlopen
f = urlopen("http://localhost:50030/jobtaskshistory.jsp?logFile=file:/n/shokuji/da/penpornk/local/hadoop/logs/history/done/version-1/squid1_1333701959021_/2012/04/06/000000/job_201204060145_0002_1333703009620_penpornk_streamjob4016883624523835860.jar&taskType=MAP&status=all")
lines = f.readlines()
content = lines[12].split('href=\"')
output = open('out', 'w')
for c in content[1:]:
	idx = c.find('\"')
	url = c[:idx]
	taskid = c[idx+2:idx+33]

	start = c.find('(')
	end = c.find(')')
	timestring = c[start+1:end]
	timestring = timestring.split(',')
	timestring = map(lambda(x): x.strip(), timestring)
	timedict = {}
	timedict['hrs'] = 0
	timedict['mins'] = 0
	timedict['sec'] = 0
	for t in timestring:
		update_timedict(timedict, t)
	
	idx = c.find('6/04')
	start = c[idx+5:idx+13]
	c = c[idx+13:]	
	idx = c.find('6/04')
	end = c[idx+5:idx+13]

	time_formula = "={0}*3600+{1}*60+{2}".format(timedict['hrs'], timedict['mins'], timedict['sec'])
	g = urlopen('http://localhost:50030/'+url)
	gg = g.readlines()
	url2 = gg[15].split('href=\"')[3]
	url2 = url2[:url2.find('\"')]
	
	node = url2[7:13]
	h = urlopen(url2)
	meeting_name = findHVC(h.readlines())
	print >> output, "{0}\t{1}\t{2}\t{3}\t{4}\t{5}".format(meeting_name, taskid, node, start, end, time_formula)
	print node

output.close()


