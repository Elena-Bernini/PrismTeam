# Running Prism on AImageLab-HPC (UNIMORE)

Based on the cluster documentation at
<https://ailb-web.ing.unimore.it/coldfront/documentation>. Check the account
name and paths in `cluster/config.sh` and in the `#SBATCH --account` lines
(`cvcs2026` is a guess taken from the `/work/cvcs2026/...` path: check your
project name on ColdFront).

## Storage layout

| What | Where | Why |
|---|---|---|
| Code | `/work/cvcs2026/PrismTeam/prism` | shared by the team |
| Raw + processed data | `/work/cvcs2026/PrismTeam/data` | BeeGFS, large files |
| Artifacts, outputs | `/work/cvcs2026/PrismTeam/{artifacts,outputs}` | shared results |
| Model caches | `/work/cvcs2026/PrismTeam/cache` | `/homes` quota is 100 GB |
| Python venv | `/homes/$USER/envs/prism` | docs: venvs on `/homes` (NFS), not `/work` |

Nothing on `/work` is backed up. Keep the raw dataset archive somewhere else too.

BeeGFS is slow with many small files. Phase 05 with `--pack` writes
`packed/layouts.jsonl.gz`, `graphs.jsonl.gz` and `images.zip`. GPU jobs
(`stage_packed` in `slurm/common.sh`) copy those few files to node-local
`/tmp` and read from there.

## One-time setup (login node)

```bash
cd /work/cvcs2026/PrismTeam
unzip prism.zip                       # or git clone
cd prism
nano cluster/config.sh                # check paths / account
bash cluster/setup_env.sh             # venv + CUDA torch + models + env check
mkdir -p logs
```

Put the RPLAN files under
`/work/cvcs2026/PrismTeam/data/raw/rplan/Interface/static/Data/`
(`data_train_converted.pkl`, `snapshot_train/`).

## Jobs

Always submit from the code folder (SLURM writes logs to `logs/`):

```bash
cd /work/cvcs2026/PrismTeam/prism

# 1) Look at everything on a small subset first (debug QoS, 1 h, 1 GPU)
sbatch cluster/slurm/preview_all.sbatch
#    -> open outputs/rplan/preview/*/report.html (scp them to your laptop)

# 2) Full run, one stage at a time (each depends on the previous one)
j1=$(sbatch --parsable cluster/slurm/01_05_preprocess.sbatch)
j2=$(sbatch --parsable --dependency=afterok:$j1 cluster/slurm/06_train_gnn.sbatch)
j3=$(sbatch --parsable --dependency=afterok:$j2 cluster/slurm/07_08_index.sbatch)
sbatch --dependency=afterok:$j3 cluster/slurm/09_12_retrieval_eval.sbatch
```

Each job can also be run in preview mode on its own:

```bash
sbatch --qos=all_qos_dbg --time=00:30:00 --export=ALL,MODE=preview cluster/slurm/06_train_gnn.sbatch
```

Pass extra options to every phase with `EXTRA`:

```bash
sbatch --export=ALL,EXTRA="--set graph.model=gat --set graph.objective=both" cluster/slurm/06_train_gnn.sbatch
sbatch --export=ALL,VISION="dinov2_base resnet50 clip_b32" cluster/slurm/07_08_index.sbatch
```

## Interactive session (debugging)

```bash
srun --partition=all_usr_prod --qos=all_qos_dbg --gres=gpu:1 --time=01:00:00 \
     --account=cvcs2026 --pty /bin/bash
source cluster/config.sh && module load $PRISM_PYTHON_MODULE && source $PRISM_VENV/bin/activate
python scripts/00_check_env.py --env cluster
```

## Useful commands

```bash
squeue -u $USER            # my jobs
scancel <job_id>           # cancel
squota                     # quota on /work (do not use du on BeeGFS)
tail -f logs/prism_06_gnn_<id>.out
```

## Resources

- `all_usr_prod`: 24 h max. Each GPU reserves 6 CPU cores and 18 GB of RAM by default.
- `all_qos_dbg`: 1 h, 1 GPU, for preview runs.
- Students get their QoS assigned automatically (no `--qos` needed for production jobs).
- `boost_usr_prod` (L40S / A40 / RTX PRO 6000) can be used by changing `--partition`.
