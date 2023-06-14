python generate.py --nvocab 100 --ngram 3 --raw --outdir data

python generate.py --nvocab 50000 --ngram 1 --raw --outdir data


singularity exec --nv -B /scratch/$USER/public/can-wikipedia-help-offline-rl-old/ngram:/ngram -B /scratch/$USER/sing/dt-sandbox/opt/conda/lib/python3.8/site-packages/mujoco_py/:/opt/conda/lib/python3.8/site-packages/mujoco_py/ -B /scratch/$USER/public/can-wikipedia-help-offline-rl-old/code/checkpoints:/checkpoints /scratch/$USER/sing/dt-sandbox bash -c "
cd /ngram
export PYTHONPATH=$PYTHONPATH:/code
nvidia-smi
echo $PATH
echo $LD_LIBRARY_PATH
python generate.py --nvocab 10 --ngram 1 --raw --outdir data_online_new --num_workers 2 --online
python generate.py --nvocab 100 --ngram 1 --raw --outdir data_online_new --num_workers 2 --online
python generate.py --nvocab 1000 --ngram 1 --raw --outdir data_online_new --num_workers 2 --online
python generate.py --nvocab 10000 --ngram 1 --raw --outdir data_online_new --num_workers 2 --online
"


python generate.py --nvocab 10 --ngram 1 --raw --outdir data_online_new_new --num_workers 8 --batch_size 256 --iterations 500 --online
python generate.py --nvocab 100 --ngram 1 --raw --outdir data_online_new_new --num_workers 8 --batch_size 256 --iterations 500 --online
python generate.py --nvocab 1000 --ngram 1 --raw --outdir data_online_new_new --num_workers 8 --batch_size 256 --iterations 500 --online


python generate.py --nvocab 10000 --ngram 1 --raw --outdir data_online_new --num_workers 32 --batch_size 1024 --iterations 125 --online
python generate.py --nvocab 100000 --ngram 1 --raw --outdir data_online_new --num_workers 48 --batch_size 1536 --iterations 84 --online
