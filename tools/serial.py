def findtime(lines):
	for l in lines:
		if l[0:4] == "real":
			return l[5:].strip()

#E004
meeting_names = ['HVC002194', 'HVC006498', 'HVC017259', 'HVC017564', 'HVC020563', 'HVC028110', 'HVC032638', 'HVC032883', 'HVC037683', 'HVC041266', 'HVC046116', 'HVC053311', 'HVC062818', 'HVC064403', 'HVC069023', 'HVC072350', 'HVC074600', 'HVC080138', 'HVC086950', 'HVC087373', 'HVC096396', 'HVC099975', 'HVC104148', 'HVC118260', 'HVC122310', 'HVC128242', 'HVC129154', 'HVC132154', 'HVC135119', 'HVC141088', 'HVC143703', 'HVC144025', 'HVC145707', 'HVC151805', 'HVC156787', 'HVC160818', 'HVC169716', 'HVC173306', 'HVC187134', 'HVC191915', 'HVC199590', 'HVC207621', 'HVC216485', 'HVC217451', 'HVC224183', 'HVC227619', 'HVC227629', 'HVC238985', 'HVC239981', 'HVC255020', 'HVC256953', 'HVC257153', 'HVC260610', 'HVC270837', 'HVC273688', 'HVC289649', 'HVC290509', 'HVC314148', 'HVC314622', 'HVC320733', 'HVC344261', 'HVC344981', 'HVC352641', 'HVC354277', 'HVC357043', 'HVC373590', 'HVC386315', 'HVC387362', 'HVC388379', 'HVC393854', 'HVC408909', 'HVC418580', 'HVC421799', 'HVC454515', 'HVC459112', 'HVC477496', 'HVC478853', 'HVC501898', 'HVC501995', 'HVC504909', 'HVC516867', 'HVC533832', 'HVC534412', 'HVC534905', 'HVC552683', 'HVC564882', 'HVC566346', 'HVC566365', 'HVC573726', 'HVC602555', 'HVC606182', 'HVC607421', 'HVC618343', 'HVC622940', 'HVC631950', 'HVC636230', 'HVC641088', 'HVC647848', 'HVC657915', 'HVC665516', 'HVC670060', 'HVC673789', 'HVC679045', 'HVC689791', 'HVC691867', 'HVC694478', 'HVC696623', 'HVC701564', 'HVC726255', 'HVC729977', 'HVC734829', 'HVC742023', 'HVC756575', 'HVC757202', 'HVC774967', 'HVC775405', 'HVC796410', 'HVC804283', 'HVC808980', 'HVC830046', 'HVC832838', 'HVC841343', 'HVC868773', 'HVC872705', 'HVC877282']
#E001
#meeting_names = ['HVC006045', 'HVC006184', 'HVC011409', 'HVC022974', 'HVC026971', 'HVC027850', 'HVC029485', 'HVC031151', 'HVC032158', 'HVC036790', 'HVC040785', 'HVC042692', 'HVC049437', 'HVC064215', 'HVC067270', 'HVC068504', 'HVC087057', 'HVC090414', 'HVC091740', 'HVC095303', 'HVC098807', 'HVC103302', 'HVC103932', 'HVC105766', 'HVC108320', 'HVC109456', 'HVC123288', 'HVC134406', 'HVC134634', 'HVC146032', 'HVC148793', 'HVC152448', 'HVC169936', 'HVC174999', 'HVC198184', 'HVC201655', 'HVC203988', 'HVC218344', 'HVC218765', 'HVC240490', 'HVC243157', 'HVC255452', 'HVC257987', 'HVC259599', 'HVC263724', 'HVC268521', 'HVC271903', 'HVC283780', 'HVC292292', 'HVC293678', 'HVC298568', 'HVC309504', 'HVC317804', 'HVC319297', 'HVC319600', 'HVC320560', 'HVC334254', 'HVC359216', 'HVC362146', 'HVC364230', 'HVC365963', 'HVC377296', 'HVC396658', 'HVC397279', 'HVC398674', 'HVC402970', 'HVC403628', 'HVC417926', 'HVC418407', 'HVC425927', 'HVC429818', 'HVC434449', 'HVC450989', 'HVC460343', 'HVC462262', 'HVC462508', 'HVC463620', 'HVC467645', 'HVC468477', 'HVC473973', 'HVC479361', 'HVC486704', 'HVC498099', 'HVC499900', 'HVC506183', 'HVC507955', 'HVC513054', 'HVC515040', 'HVC516366', 'HVC520461', 'HVC528762', 'HVC528929', 'HVC529613', 'HVC532992', 'HVC532993', 'HVC539647', 'HVC541506', 'HVC542481', 'HVC549861', 'HVC553302', 'HVC561733', 'HVC562777', 'HVC570651', 'HVC573643', 'HVC579741', 'HVC591147', 'HVC597104', 'HVC602688', 'HVC605240', 'HVC606315', 'HVC615626', 'HVC616948', 'HVC620320', 'HVC631691', 'HVC637907', 'HVC646345', 'HVC652043', 'HVC658517', 'HVC672564', 'HVC676565', 'HVC680956', 'HVC681821', 'HVC682794', 'HVC683835', 'HVC687278', 'HVC690639', 'HVC699748', 'HVC705168', 'HVC707236', 'HVC711253', 'HVC717615', 'HVC718274', 'HVC719228', 'HVC726373', 'HVC730049', 'HVC730081', 'HVC733376', 'HVC742499', 'HVC749106', 'HVC766498', 'HVC776608', 'HVC786536', 'HVC788532', 'HVC789034', 'HVC792251', 'HVC803719', 'HVC811971', 'HVC822060', 'HVC823377', 'HVC823657', 'HVC829585', 'HVC834162', 'HVC836424', 'HVC846668', 'HVC878334', 'HVC881736', 'HVC883718', 'HVC887091', 'HVC888814', 'HVC891523']


for m in meeting_names:
	try:
		f = open('/n/shokuji/da/penpornk/all/serial/E004/'+m+'.log', 'r')
		lines = f.readlines()
		print "{0}\t{1}".format(m, findtime(lines))
	except:
		print "{0}\tFile not found.".format(m)
