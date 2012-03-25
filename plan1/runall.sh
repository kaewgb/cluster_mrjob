for meeting in \
	'IS1000a' 'IS1000b' 'IS1000c' 'IS1000d' \
	'IS1001a' 'IS1001b' 'IS1001c' \
        'IS1003b' 'IS1003d' \
        'IS1006b' 'IS1006d' \
        'IS1008a' 'IS1008b' 'IS1008c' 'IS1008d' 
do
	echo python cluster.py -c conf/${meeting}.cfg
	python cluster.py -c conf/${meeting}.cfg
done
