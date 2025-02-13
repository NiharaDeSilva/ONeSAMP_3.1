#!/bin/bash
#SBATCH --job-name=oneSamp    # Job name
#SBATCH --mail-type=ALL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=suhashidesilva@ufl.edu     # Where to send mail
#SBATCH --ntasks=1		      # Number of tasks
#SBATCH --cpus-per-task=64	      # Number of cores per task
#SBATCH --mem=10gb                     # Job memory request
#SBATCH --time=24:00:00               # Time limit hrs:min:sec
#SBATCH --output=serial_test_%j.log   # Standard output and error log
pwd; hostname; date

module load R/4.1
chmod +rwx /blue/boucher/suhashidesilva/2025/ONeSAMP_3.1/build/OneSamp

echo "Running plot script on multiple CPU cores"

#python /blue/boucher/suhashidesilva/ONeSAMP_3.1/ONeSAMP_3/main.py --s 1000 --o /blue/boucher/suhashidesilva/ONeSAMP_3.1/ONeSAMP_3/exampleData/genePop5Ix5L > /blue/boucher/suhashidesilva/ONeSAMP_3.1/ONeSAMP_3/genePop5Ix5L.out

folder="/blue/boucher/suhashidesilva/2025/ONeSAMP_3.1/data"
output="/blue/boucher/suhashidesilva/2025/ONeSAMP_3.1/output"


#Iterate through the files in the folder
for file in "$folder"/*; do
    if [ -f "$file" ]; then
        filename=$(basename -- "$file")
        filename_no_extension="${filename%.*}"
        output_file="$output/${filename_no_extension}"
        python /blue/boucher/suhashidesilva/2025/ONeSAMP_3.1/main.py --s 20000 --o "$file" > "$output_file"
        echo "Processed $file and saved output to $output_file"
    fi
done

date
