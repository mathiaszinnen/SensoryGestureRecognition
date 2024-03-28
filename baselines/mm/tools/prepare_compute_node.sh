mkdir ${TMPDIR}/$SLURM_JOB_ID
cd ${TMPDIR}/$SLURM_JOB_ID

cp -r $HOME/sniffyart/sniffyart-experiments .

cd sniffyart-experiments

echo "start copying data"

mkdir -p ./data/sniffyart
tar xvf /home/janus/iwi5-datasets/sniffyart/sniffyart.tar -C ./data/sniffyart