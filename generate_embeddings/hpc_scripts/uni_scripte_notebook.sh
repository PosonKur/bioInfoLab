#!/usr/bin/zsh 

### Job Parameters 

#SBATCH --time=01:00:00         # Run time of 15 minutes
#SBATCH --job-name=uni_notebook  # Sets the job name
#SBATCH --output=/work/jo666642/UNI/notebooks/results/output.out   # Replace with your project-id or delete the line
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu 50G
source  /rwthfs/rz/cluster/home/jo666642/conda/bin/activate
conda activate UNI
python uni_walk.py
#export CONDA_ROOT=$HOME/mambaforge
#. $CONDA_ROOT/etc/profile.d/conda.sh
#export PATH="$CONDA_ROOT/bin:$PATH"

echo "------------------------------------------------------------"
echo "SLURM JOB ID: $SLURM_JOBID"
echo "Running on nodes: $SLURM_NODELIST"
echo "------------------------------------------------------------"

NOTEBOOKPORT=shuf -i 8000-8500 -n 1

TUNNELPORT=shuf -i 8501-9000 -n 1

TOKEN=cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 49 | head -n 1

echo "On your local machine, run:"
echo ""
echo "ssh -L8888:localhost:$TUNNELPORT $USER@$SLURM_SUBMIT_HOST -N -4"
echo ""
echo "and point your browser to http://localhost:8888/?token=$TOKEN"
echo "Change '8888' to some other value if this port is already in use on your PC,"
echo "for example, you have more than one remote notebook running."
echo "To stop this notebook, run 'scancel $SLURM_JOB_ID'"

ssh -R$TUNNELPORT:localhost:$NOTEBOOKPORT $SLURM_SUBMIT_HOST -N -f

srun -n1 /rwthfs/rz/cluster/home/jo666642/conda/envs/UNI/bin/jupyter-lab --no-browser --port=$NOTEBOOKPORT --NotebookApp.token=$TOKEN --log-level WARN --notebook-dir /work/project/YOURS/scripts

#singularity exec --bind /hpcwork/fp015734/DBit-seq/rstudio:/mnt/rstudio,/hpcwork/fp015734/DBit-seq/jupyter:/mnt/jupyter,/hpcwork/fp015734/DBit-seq/var-lib-rstudio-server:/var/lib/rstudio-server,/hpcwork/fp015734/DBit-seq/database.conf:/etc/rstudio/database.conf /hpcwork/YOURS/Software/singularity/jupyter_rstudio1.sif jupyter lab --notebook-dir=/mnt/jupyter --no-browser --port=$NOTEBOOKPORT --NotebookApp.token=$TOKEN --log-level WARN