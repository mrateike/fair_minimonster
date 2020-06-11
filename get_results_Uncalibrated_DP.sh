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

echo "uncal;DP;eps;acc;mean_pred;utility;DP" > "$1"

for file in $(find "$2" -name "evaluation_mean.json"); do
	# uncal
    echo "$file" | cut -d "_" -f 3 | tr -d "\n" >> "$1"
	echo -n ";" >> "$1"

	# DP
	echo "$file" | cut -d "_" -f 4 | cut -d "/" -f 1 | tr -d "\n" >> "$1"
	echo -n ";" >> "$1"

	# eps
	echo "$file" | cut -d "_" -f 5 | cut -d "/" -f 1 | tr -d "\n" >> "$1"
	echo -n ";" >> "$1"

	# acc
	cat "$file" | cut -d "," -f1  | cut -d " " -f 2| tr -d "\n" >> "$1"
	echo -n ";" >> "$1"

	# mean_pred
	cat "$file" | cut -d "," -f2 | cut -d " " -f3| tr -d "\n" >> "$1"
	echo -n ";" >> "$1"

	# utility
	cat "$file" | cut -d "," -f3 | cut -d " " -f3| tr -d "\n" >> "$1"
	echo -n ";" >> "$1"

	# DP
	cat "$file" | cut -d "," -f4 | cut -d " " -f3 | cut -d "}" -f1| tr -d "\n" >> "$1"


	echo "" >> "$1"
done
