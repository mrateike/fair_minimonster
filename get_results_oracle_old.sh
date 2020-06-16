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

echo "file;data;fair;alpha;seed;eps;mu;nu;N;util_mean;util_FQ;util_TQ;acc_mean;acc_FQ;acc_TQ;DP_mean;DP_FQ;DP_TQ;FPR_mean;FPR_FQ;FPR_TQ;util_std;acc_std;DP_std;FPR_std;util_Q025;util_Q975;acc_Q025;acc_Q975;DP_Q025;DP_Q975;FPR_Q025;FPR_Q975" > "$1"

for file in $(find "$2" -name "*evaluation.json*"); do
	
	#oracle, var, loss, dec
	echo "$file" | cut -d "_" -f10 | cut -d "/" -f2 | tr -d "\n" >> "$1"
	echo -n ";" >> "$1"

	# # uncal
 #    echo "$file" |cut -d "_" -f3 | tr -d "\n" >> "$1"
	# echo -n ";" >> "$1"

	# dataset (Uncalibrated, FICO)
	echo "$file" | cut -d "_" -f3 | cut -d "/" -f1 | tr -d "\n" >> "$1"
	echo -n ";" >> "$1"

	# fairness (DP, EO)
	echo "$file" | cut -d "_" -f4 | cut -d "/" -f1 | tr -d "\n" >> "$1"
	echo -n ";" >> "$1"

	# alpha
	echo "$file" | cut -d "_" -f5 | cut -d "/" -f1 | tr -d "\n" >> "$1"
	echo -n ";" >> "$1"
 
	# seed 
	echo "$file" | cut -d "_" -f6 | cut -d "/" -f1 |tr -d "\n" >> "$1"
	echo -n ";" >> "$1"

	# eps
	echo "$file" | cut -d "_" -f7 | cut -d "/" -f1 | cut -d "-" -f1 | tr -d "\n" >> "$1"
	echo -n ";" >> "$1"

	#  mu
	echo "$file" | cut -d "_" -f8 | cut -d "/" -f1 | cut -d "-" -f1 | tr -d "\n" >> "$1"
	echo -n ";" >> "$1"

	#  nu
	echo "$file" | cut -d "_" -f9 | cut -d "/" -f1 | cut -d "N" -f1 | rev | cut -c 2- | rev | tr -d "\n" >> "$1"
	echo -n ";" >> "$1"

	#  N
	echo "$file" | cut -d "_" -f10 | cut -d "/" -f1 | cut -d "-" -f1 | tr -d "\n" >> "$1"
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

	# FPR_TQ
	cat "$file" | cut -d "," -f12 | cut -d " " -f3 | tr -d "\n" >> "$1"
	echo -n ";" >> "$1"

	# Util STD
	cat "$file" | cut -d "," -f13 | cut -d " " -f3 | tr -d "\n" >> "$1"
	echo -n ";" >> "$1"

	# Acc STD
	cat "$file" | cut -d "," -f14 | cut -d " " -f3 | tr -d "\n" >> "$1"
	echo -n ";" >> "$1"

	# DP STD
	cat "$file" | cut -d "," -f15 | cut -d " " -f3 | tr -d "\n" >> "$1"
	echo -n ";" >> "$1"

	# FPR STD
	cat "$file" | cut -d "," -f16 | cut -d " " -f3 | tr -d "\n" >> "$1"
	echo -n ";" >> "$1"

	# UTIL_Q025
	cat "$file" | cut -d "," -f17 | cut -d " " -f3 | tr -d "\n" >> "$1"
	echo -n ";" >> "$1"

	# UTIL_Q975
	cat "$file" | cut -d "," -f18 | cut -d " " -f3 | tr -d "\n" >> "$1"
	echo -n ";" >> "$1"

	# Acc_Q025
	cat "$file" | cut -d "," -f19 | cut -d " " -f3 | tr -d "\n" >> "$1"
	echo -n ";" >> "$1"

	# Acc_Q975
	cat "$file" | cut -d "," -f20 | cut -d " " -f3 | tr -d "\n" >> "$1"
	echo -n ";" >> "$1"

	# DP_Q025
	cat "$file" | cut -d "," -f21 | cut -d " " -f3 | tr -d "\n" >> "$1"
	echo -n ";" >> "$1"

	# DP_Q975
	cat "$file" | cut -d "," -f22 | cut -d " " -f3 | tr -d "\n" >> "$1"
	echo -n ";" >> "$1"

	# FPR_Q025
	cat "$file" | cut -d "," -f23 | cut -d " " -f3 | tr -d "\n" >> "$1"
	echo -n ";" >> "$1"

	# FPR_Q975
	cat "$file" | cut -d "," -f24 | cut -d " " -f3 | cut -d "}" -f1| tr -d "\n" >> "$1"


	echo "" >> "$1"
done
