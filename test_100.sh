#!/bin/bash
#SBATCH --job-name=oneSamp    # Job name
#SBATCH --mail-type=ALL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=suhashidesilva@ufl.edu     # Where to send mail
#SBATCH --ntasks=1		      # Number of tasks
#SBATCH --cpus-per-task=2	      # Number of cores per task
#SBATCH --mem=10gb                     # Job memory request
#SBATCH --time=48:00:00               # Time limit hrs:min:sec
#SBATCH --output=serial_test_%j.log   # Standard output and error log
pwd; hostname; date

#module load R/4.1
module load conda

# Ensure Conda is properly initialized for SLURM
eval "$(conda shell.bash hook)"
conda activate "/blue/boucher/suhashidesilva/conda_envs/my_env"

echo "Python path: $(which python)"
python -c "import numba; print(numba.__version__)"

'''
module load conda/25.3.1
source activate /blue/boucher/suhashidesilva/conda_envs/my_env

which python
python -c "import numba; print(numba.__version__)"
'''
chmod +rwx /blue/boucher/suhashidesilva/Latest/ONeSAMP_3.1/build/OneSamp
export PYTHONPATH=$PYTHONPATH:/blue/boucher/suhashidesilva/2025/WFsim/

echo "Running plot script on multiple CPU cores"

#python /blue/boucher/suhashidesilva/ONeSAMP_3.1/ONeSAMP_3/main.py --s 10000 --o /blue/boucher/suhashidesilva/ONeSAMP_3.1/ONeSAMP_3/exampleData/genePop5Ix5L > /blue/boucher/suhashidesilva/ONeSAMP_3.1/ONeSAMP_3/genePop5Ix5L.out

folder="/blue/boucher/suhashidesilva/Latest/ONeSAMP_3.1/data_70"
output="/blue/boucher/suhashidesilva/Latest/ONeSAMP_3.1/output_70"


#Iterate through the files in the folder
for file in "$folder"/*; do
    if [ -f "$file" ]; then
        filename=$(basename -- "$file")
        filename_no_extension="${filename%.*}"
        output_file="$output/${filename_no_extension}"
        python /blue/boucher/suhashidesilva/Latest/ONeSAMP_3.1/main.py --s 100 --o "$file" > "$output_file"
        echo "Processed $file and saved output to $output_file"
    fi
done

date
