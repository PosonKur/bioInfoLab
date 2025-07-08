#!/usr/bin/zsh 

### Job Parameters 

#SBATCH --time=01:00:00         # Run time of 15 minutes
#SBATCH --job-name=uni_notebook  # Sets the job name
#SBATCH --output=/work/jo666642/UNI/notebooks/results/output.out   # Replace with your project-id or delete the line
#SBATCH --gres=gpu:1
source  /rwthfs/rz/cluster/home/jo666642/conda/bin/activate
conda activate UNI
python uni_walk.py