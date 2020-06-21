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

echo "data;fair;alpha;seed;regt;regT;xaxis" > "$1"

for file in $(find "$2" -name "*regret.json*"); do

	# dataset
	echo "$file" | cut -d "_" -f4 | cut -d "/" -f1 | tr -d "\n" >> "$1"
	echo -n ";" >> "$1"

	# fair
	echo "$file" | cut -d "_" -f5 | cut -d "/" -f1 | tr -d "\n" >> "$1"
	echo -n ";" >> "$1"

	# alpha
	echo "$file" | cut -d "_" -f7 | cut -d "/" -f1 | tr -d "\n" >> "$1"
	echo -n ";" >> "$1"

	# seed
	echo "$file" | cut -d "_" -f8 | cut -d "/" -f1 | tr -d "\n" >> "$1"
	echo -n ";" >> "$1"

	

	#----- measures file

	# Rt
	cat "$file" | cut -d "[" -f2 | cut -d ']' -f 1| tr -d "\n" >> "$1"
	echo -n ";" >> "$1"

	# RT
	cat "$file" | cut -d "[" -f3 | cut -d ']' -f 1| tr -d "\n" >> "$1"
	echo -n ";" >> "$1"

	# xaxis
	cat "$file" | cut -d "[" -f4 | cut -d ']' -f 1 | tr -d "\n" >> "$1"

	echo "" >> "$1"
done
