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

echo "batch;file;data;fair;alpha;seed;eps;mu;nu;N;\
reg_type;RT;regt_mean;regt_FQ;regt_TQ;regt_STD;regt_Q025;regt_Q975; \
regt_cum_mean;regt_cum_FQ;regt_cum_TQ;regt_cum_STD;regt_cum_Q025;regt_cum_Q975; \
regT_mean;regT_FQ;regT_TQ;regT_STD;regT_Q025;regT_Q975;\
regT_cum_mean;regT_cum_FQ;regT_cum_TQ;regT_cum_STD;regT_cum_Q025;regT_cum_Q975;" > "$1"

for file in $(find "$2" -name "*regret_evaluation.json*"); do

	# batch (no = no)
	echo "$file" | cut -d "_" -f1 | cut -d "/" -f3 | tr -d "\n" >> "$1"
	echo -n ";" >> "$1"

	#oracle, var, loss, dec, reg
	echo "$file" | cut -d "_" -f18 | cut -d "/" -f2 | tr -d "\n" >> "$1"
	echo -n ";" >> "$1"

	# # uncal
 #    echo "$file" |cut -d "_" -f3 | tr -d "\n" >> "$1"
	# echo -n ";" >> "$1"

	# dataset (Uncalibrated, FICO) (for us 8)
	echo "$file" | cut -d "_" -f6 | cut -d "/" -f1 | tr -d "\n" >> "$1"
	echo -n ";" >> "$1"

	# fairness (DP, EO)
	echo "$file" | cut -d "_" -f7 | cut -d "/" -f1 | tr -d "\n" >> "$1"
	echo -n ";" >> "$1"

	# alpha
	echo "$file" | cut -d "_" -f8 | cut -d "/" -f1 | tr -d "\n" >> "$1"
	echo -n ";" >> "$1"

	# seed
	echo "$file" | cut -d "_" -f9 | cut -d "/" -f1 |tr -d "\n" >> "$1"
	echo -n ";" >> "$1"

	# eps
	echo "$file" | cut -d "_" -f10 | cut -d "/" -f1 | cut -d "-" -f1 | tr -d "\n" >> "$1"
	echo -n ";" >> "$1"

	#  mu
	echo "$file" | cut -d "_" -f12 | cut -d "/" -f1 | cut -d "-" -f1 | tr -d "\n" >> "$1"
	echo -n ";" >> "$1"

	#  nu
	echo "$file" | cut -d "_" -f14 | cut -d "/" -f1 | tr -d "\n" >> "$1"
	echo -n ";" >> "$1"

	#  N
	echo "$file" | cut -d "_" -f16 | cut -d "/" -f1 | tr -d "\n" >> "$1"
	echo -n ";" >> "$1"

	#----- measures file

	# RT
	cat "$file" | cut -d "," -f 1 | cut -d ' ' -f 2| tr -d "\n" >> "$1"
	echo -n ";" >> "$1"

	# regt mean
	cat "$file" | cut -d "," -f2 | cut -d ' ' -f 4| tr -d "\n" >> "$1"
	echo -n ";" >> "$1"

	# regt FQ
	cat "$file" | cut -d "," -f3 | cut -d ' ' -f 3 | tr -d "\n" >> "$1"
	echo -n ";" >> "$1"

	# regt_TQ
	cat "$file" | cut -d "," -f4 | cut -d ' ' -f 3 | tr -d "\n" >> "$1"
	echo -n ";" >> "$1"

	# regt_STD
	cat "$file" | cut -d "," -f5 | cut -d ' ' -f 3 | tr -d "\n" >> "$1"
	echo -n ";" >> "$1"

	# regt_Q025
	cat "$file" | cut -d "," -f6 | cut -d ' ' -f 3 | tr -d "\n" >> "$1"
	echo -n ";" >> "$1"

	# regt_Q975
	cat "$file" | cut -d "," -f7 | cut -d ' ' -f 3| cut -d "}" -f1| tr -d "\n" >> "$1"
	echo -n ";" >> "$1"

	#

	# regt_cum mean
	cat "$file" | cut -d "," -f8 | cut -d ' ' -f 4| tr -d "\n" >> "$1"
	echo -n ";" >> "$1"

	# regt_cum FQ
	cat "$file" | cut -d "," -f9 | cut -d ' ' -f 3 | tr -d "\n" >> "$1"
	echo -n ";" >> "$1"

	# regt_cum_TQ
	cat "$file" | cut -d "," -f10 | cut -d ' ' -f 3 | tr -d "\n" >> "$1"
	echo -n ";" >> "$1"

	# regt_cum_STD
	cat "$file" | cut -d "," -f11 | cut -d ' ' -f 3 | tr -d "\n" >> "$1"
	echo -n ";" >> "$1"

	# regt_cum_Q025
	cat "$file" | cut -d "," -f12 | cut -d ' ' -f 3 | tr -d "\n" >> "$1"
	echo -n ";" >> "$1"

	# regt_cum_Q975
	cat "$file" | cut -d "," -f13 | cut -d ' ' -f 3| cut -d "}" -f1| tr -d "\n" >> "$1"
	echo -n ";" >> "$1"

	#

	# regt_cum mean
	cat "$file" | cut -d "," -f14 | cut -d ' ' -f 4| tr -d "\n" >> "$1"
	echo -n ";" >> "$1"

	# regt_cum FQ
	cat "$file" | cut -d "," -f15 | cut -d ' ' -f 3 | tr -d "\n" >> "$1"
	echo -n ";" >> "$1"

	# regt_cum_TQ
	cat "$file" | cut -d "," -f16 | cut -d ' ' -f 3 | tr -d "\n" >> "$1"
	echo -n ";" >> "$1"

	# regt_cum_STD
	cat "$file" | cut -d "," -f17 | cut -d ' ' -f 3 | tr -d "\n" >> "$1"
	echo -n ";" >> "$1"

	# regt_cum_Q025
	cat "$file" | cut -d "," -f18 | cut -d ' ' -f 3 | tr -d "\n" >> "$1"
	echo -n ";" >> "$1"

	# regt_cum_Q975
	cat "$file" | cut -d "," -f19 | cut -d ' ' -f 3| cut -d "}" -f1| tr -d "\n" >> "$1"
	echo -n ";" >> "$1"

	#

	# regt_cum mean
	cat "$file" | cut -d "," -f20 | cut -d ' ' -f 4| tr -d "\n" >> "$1"
	echo -n ";" >> "$1"

	# regt_cum FQ
	cat "$file" | cut -d "," -f21 | cut -d ' ' -f 3 | tr -d "\n" >> "$1"
	echo -n ";" >> "$1"

	# regt_cum_TQ
	cat "$file" | cut -d "," -f22 | cut -d ' ' -f 3 | tr -d "\n" >> "$1"
	echo -n ";" >> "$1"

	# regt_cum_STD
	cat "$file" | cut -d "," -f23 | cut -d ' ' -f 3 | tr -d "\n" >> "$1"
	echo -n ";" >> "$1"

	# regt_cum_Q025
	cat "$file" | cut -d "," -f24 | cut -d ' ' -f 3 | tr -d "\n" >> "$1"
	echo -n ";" >> "$1"

	# regt_cum_Q975
	cat "$file" | cut -d "," -f25 | cut -d " " -f3 | cut -d "}" -f1| tr -d "\n" >> "$1"


	echo "" >> "$1"
done
