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

echo "data;fair;seed;eps;util_mean;util_FQ;util_TQ;acc_mean;acc_FQ;acc_TQ;DP_mean;DP_FQ;DP_TQ;FPR_mean;FPR_FQ;FPR_TQ" > "$1"

for file in $(find "$2" -name "evaluation_mean.json"); do
	# uncal
    echo "$file" |cut -d "_" -f3 | tr -d "\n" >> "$1"
	echo -n ";" >> "$1"

	# DP
	echo "$file" | cut -d "_" -f4 | cut -d "/" -f1 | tr -d "\n" >> "$1"
	echo -n ";" >> "$1"

	# seed
	echo "$file" | cut -d "_" -f5 | cut -d "/" -f1 | tr -d "\n" >> "$1"
	echo -n ";" >> "$1"

	# eps
	echo "$file" | cut -d "_" -f6 | cut -d "/" -f1 | cut -d "-" -f1 | tr -d "\n" >> "$1"
	echo -n ";" >> "$1"

	# util_mean
	cat "$file" | cut -d "," -f 1 | cut -d ' ' -f 2| tr -d "\n" >> "$1"
	echo -n ";" >> "$1"

	# util_FQ
	cat "$file" | cut -d "," -f2 | cut -d ' ' -f 3| tr -d "\n" >> "$1"
	echo -n ";" >> "$1"

	# util_TQ
	cat "$file" | cut -d "," -f3 | cut -d ' ' -f 3 | tr -d "\n" >> "$1"
	echo -n ";" >> "$1"

	# acc_mean
	cat "$file" | cut -d "," -f4 | cut -d ' ' -f 3 | tr -d "\n" >> "$1"
	echo -n ";" >> "$1"

	# acc_FQ
	cat "$file" | cut -d "," -f5 | cut -d ' ' -f 3 | tr -d "\n" >> "$1"
	echo -n ";" >> "$1"

	# acc_TQ
	cat "$file" | cut -d "," -f6 | cut -d ' ' -f 3 | tr -d "\n" >> "$1"
	echo -n ";" >> "$1"

	# DP_mean
	cat "$file" | cut -d "," -f7 | cut -d ' ' -f 3| tr -d "\n" >> "$1"
	echo -n ";" >> "$1"

	# DP_FQ
	cat "$file" | cut -d "," -f8 | cut -d ' ' -f 3| tr -d "\n" >> "$1"
	echo -n ";" >> "$1"

	# DP_TQ
	cat "$file" | cut -d "," -f9 | cut -d ' ' -f 3 | tr -d "\n" >> "$1"
	echo -n ";" >> "$1"

	# FPR_mean
	cat "$file" | cut -d "," -f10 | cut -d ' ' -f 3| tr -d "\n" >> "$1"
	echo -n ";" >> "$1"

	# FPR_FQ
	cat "$file" | cut -d "," -f11 | cut -d ' ' -f 3| tr -d "\n" >> "$1"
	echo -n ";" >> "$1"

	# DP
	cat "$file" | cut -d "," -f12 | cut -d " " -f3 | cut -d "}" -f1| tr -d "\n" >> "$1"


	echo "" >> "$1"
done
