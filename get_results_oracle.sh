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

echo "data;fair;seed;beta;acc" > "$1"

for file in $(find "$2" -name "*evaluation.json*"); do
	
	# dataset (Uncalibrated, FICO)
	echo "$file" | cut -d "_" -f1 | cut -d "/" -f4 | tr -d "\n" >> "$1"
	echo -n ";" >> "$1"

	# fairness (DP, EO)
	echo "$file" | cut -d "_" -f4 | cut -d "/" -f1 | tr -d "\n" >> "$1"
	echo -n ";" >> "$1"

	# seed 
	echo "$file" | cut -d "_" -f6 | cut -d "/" -f1 |tr -d "\n" >> "$1"
	echo -n ";" >> "$1"

	# eps 
	echo "$file" | cut -d "_" -f7 | cut -d "-" -f1 |tr -d "\n" >> "$1"
	echo -n ";" >> "$1"

	# err_mean
	cat "$file" | cut -d "," -f 1 | cut -d ' ' -f 2| tr -d "\n" >> "$1"
	# echo -n ";" >> "$1"

	# # err_std
	# cat "$file" | cut -d "," -f2 | cut -d ' ' -f 3| tr -d "\n" >> "$1"


	echo "" >> "$1"
done
