for meeting in \
	'IS1000a' 'IS1000b' 'IS1000c' 'IS1000d' \
	'IS1001a' 'IS1001b' 'IS1001c' \
        'IS1003b' 'IS1003d' \
        'IS1006b' 'IS1006d' \
        'IS1008a' 'IS1008b' 'IS1008c' 'IS1008d' 
do
	echo diff -q serial/${meeting}.rttm hadoop/${meeting}.rttm
	diff -q serial/${meeting}.rttm hadoop/${meeting}.rttm
	echo sleep 2
	sleep 2
	echo python gmm_matchall.py serial/${meeting}.gmm hadoop/${meeting}.gmm
	python gmm_matchall.py serial/${meeting}.gmm hadoop/${meeting}.gmm
	echo sleep 3
	sleep 3
done

