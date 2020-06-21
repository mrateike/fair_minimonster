#!/bin/bash

if [ ! $# -eq 2 ]; then
	echo "Usage: $0 summary_file exp_folder"
        exit 0
fi

if [ -e "$1" ]; then
	read -p "Do you want to delete the file ${1}? ([Y]/n)" ans
	if [ -z "${ans}" ] || [ "$ans" == "Y" ] || [ "$ans" == "y" ]; then
		rm "$1"
	else
		exit 0
	fi
fi

echo "batch;bsize;data;fair;alpha;beta;seed;acc;DP;TPR;axis" > "$1"

for file in $(find "$2" -name "*evaluation.json*"); do

	# batch (none, lin, exp)
	echo "$file" | cut -d "_" -f3 | tr -d "\n" >> "$1"
	echo -n ";" >> "$1"

	# batchsize
	echo "$file" | cut -d "_" -f6 |  cut -d "/" -f1 | tr -d "\n" >> "$1"
	echo -n ";" >> "$1"

	# dataset (Uncalibrated, FICO) (for us 8)
	echo "$file" | cut -d "_" -f4  | tr -d "\n" >> "$1"
	echo -n ";" >> "$1"

	# fairness (DP, EO)
	echo "$file" | cut -d "_" -f5 | cut -d "/" -f1 | tr -d "\n" >> "$1"
	echo -n ";" >> "$1"

	# alpha
	echo "$file" | cut -d "_" -f9 | cut -d "/" -f1  | tr -d "\n" >> "$1"
	echo -n ";" >> "$1"

	# beta
	echo "$file" | cut -d "_" -f9 | tr -d "\n" >> "$1"
	echo -n ";" >> "$1"

	# seed
	echo "$file" | cut -d "_" -f6 | cut -d "/" -f1  | tr -d "\n" >> "$1"
	echo -n ";" >> "$1"

	#

	# acc
	cat "$file" | cut -d "[" -f 2 | cut -d ']' -f 1| tr -d "\n" >> "$1"
	echo -n ";" >> "$1"

	# DP
	cat "$file" | cut -d "[" -f 3 | cut -d ']' -f 1| tr -d "\n" >> "$1"
	echo -n ";" >> "$1"

	# util_TQ
	cat "$file" | cut -d "[" -f 4 | cut -d ']' -f 1 | tr -d "\n" >> "$1"
	echo -n ";" >> "$1"

	# axis
	cat "$file" | cut -d "[" -f 5 | cut -d ']' -f 1 | tr -d "\n" | tr -d "\n" >> "$1"





	echo "" >> "$1"
done
