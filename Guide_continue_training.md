# Guide : Reprendre l'entraînement multi-GPU avec le scheduler

Ce document remplace les instructions de l'ancienne méthode décrite dans `LSTM_Experiment_Guide_2026.docx`
(section *"Note2: how to relaunch training if it didn't reach 30 epochs"*).

---

## Ce qui a changé par rapport à l'ancienne méthode

| Ancienne méthode | Nouvelle méthode |
|---|---|
| Commenter/décommenter des lignes dans `train_ensemble.ipynb` | Même script, nouveau mode `continue_training` |
| Modifier `epochs:` dans chaque seed file avant chaque reprise | Inutile — `epochs: 30` signifie maintenant "atteindre l'epoch 30 au total" |
| Charger l'ancien environnement `nh_gpu` (mamba) | Le nouvel environnement fonctionne maintenant |
| Relancer un modèle à la fois | 4 modèles en parallèle sur 4 GPUs |
| Copier manuellement `model_epoch030.pt` dans le dossier parent | Inutile — le scheduler trouve les checkpoints dans les sous-dossiers |

---

## Référence complète des arguments

### `nh_run_scheduler.py`

```
python nh_run_scheduler.py <mode> --directory <chemin> --gpu-ids <ids> --runs-per-gpu <n> [--target-epochs <n>]
```

| Argument | Type | Requis | Description |
|---|---|---|---|
| `mode` | positional | oui | `train`, `evaluate`, `finetune`, ou `continue_training` |
| `--directory` | str | oui | Chemin vers le dossier contenant les runs (voir détail par mode ci-dessous) |
| `--gpu-ids` | int(s) | oui | IDs des GPUs à utiliser, séparés par des espaces. Ex: `0 1 2 3` |
| `--runs-per-gpu` | int | oui | Nombre de runs lancés simultanément sur un seul GPU. Mettre `1` pour les LSTMs lourds. |
| `--target-epochs` | int | non | **Seulement pour `continue_training`.** Force une cible d'epoch différente de celle dans les `config.yml`. Utile si on veut s'arrêter à 20 au lieu de 30 pour tester. |

**Ce que `--directory` doit contenir selon le mode :**

- `train` / `finetune` → dossier contenant les fichiers `.yml` (ex: `experiments/example/ens_configs/`)
- `evaluate` → dossier contenant les sous-dossiers de runs (ex: `experiments/example/runs/`)
- `continue_training` → **même dossier que pour `evaluate`** : les sous-dossiers de runs (ex: `experiments/example/runs/`)

---

### `nh_run.py` (appelé par le scheduler, mais peut aussi être utilisé directement)

```
python nh_run.py continue_training --run-dir <chemin_run> [--config-file <yml>] [--gpu <id>]
```

| Argument | Type | Requis | Description |
|---|---|---|---|
| `continue_training` | positional | oui | Mode de reprise |
| `--run-dir` | str | oui | Chemin vers le dossier du run à reprendre (ex: `runs/V31_30ep-256N_ens4_2605_132454/`) |
| `--config-file` | str | non | Fichier `.yml` dont les valeurs écraseront celles du `config.yml` d'origine. Utile pour changer `epochs` ou `learning_rate` ponctuellement. |
| `--gpu` | int | non | ID du GPU. Écrase la valeur dans le config. Valeur négative = CPU. |

---

## Logique de détection des runs incomplets

Quand `continue_training` est utilisé, le scheduler scanne chaque sous-dossier de `--directory` et :

1. Cherche les fichiers `model_epoch*.pt` dans le dossier du run **et** dans ses sous-dossiers `continue_training_from_epoch*/`
2. Prend le numéro d'epoch le plus élevé trouvé comme "dernier epoch complété"
3. Lit `epochs:` dans le `config.yml` du run pour connaître la cible (ou utilise `--target-epochs` si fourni)
4. Si dernier epoch < cible → le run est mis en queue
5. Si dernier epoch >= cible → le run est ignoré (déjà terminé)

Un run sans aucun checkpoint (epoch 0) est considéré incomplet et sera aussi mis en queue.

---

## Version 1 : Ligne de commande directe

Depuis le dossier de l'expérience, après avoir activé l'environnement :

```bash
source /home/ega001/store8/machine_learning/neuralhydrology/.venv/bin/activate

python /home/ega001/store8/machine_learning/neuralhydrology/neuralhydrology/nh_run_scheduler.py \
    continue_training \
    --directory /chemin/vers/experiments/example/runs/ \
    --gpu-ids 0 1 2 3 \
    --runs-per-gpu 1
```

Sortie attendue :
```
Scanning /chemin/vers/experiments/example/runs/ for incomplete runs...
  V31_30ep-256N_ens1_2205_093012: 15/30 epochs done — queuing for continuation
  V31_30ep-256N_ens2_2205_094501: 30/30 epochs, skipping
  V31_30ep-256N_ens3_2205_095022: 18/30 epochs done — queuing for continuation
  V31_30ep-256N_ens4_2205_095544: 0/30 epochs done — queuing for continuation
Starting run 1/3: python ... continue_training --run-dir .../ens1_... --gpu 0
Starting run 2/3: python ... continue_training --run-dir .../ens3_... --gpu 1
Starting run 3/3: python ... continue_training --run-dir .../ens4_... --gpu 2
Finished run ...
Done
```

Pour forcer une cible différente (ex: tu veux juste atteindre epoch 20 pour l'instant) :

```bash
python nh_run_scheduler.py continue_training \
    --directory experiments/example/runs/ \
    --gpu-ids 0 1 2 3 \
    --runs-per-gpu 1 \
    --target-epochs 20
```

---

## Version 2 : Via le notebook `train_ensemble.ipynb` (méthode PBS/qsub)

C'est l'équivalent de ce qui etait fait avant, adapté au nouveau code.

### Contenu du notebook `train_ensemble.ipynb`

Remplacer le contenu du notebook par ceci. Le bloc `if/else` permet de basculer entre lancement initial et reprise sans changer de fichier.

```python
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path.cwd().parent.parent.parent))  # ajuster selon la structure du projet
from neuralhydrology.nh_run_scheduler import schedule_runs
from neuralhydrology.nh_run import continue_run, start_run

RUNS_DIR   = Path("runs/")          # dossier où sont stockés les runs
CONFIG_DIR = Path("ens_configs/")   # dossier contenant les seed_X.yml

if torch.cuda.is_available():
    # --- Lancement initial (ou reprise des incomplets) ---
    # Changer mode='train' pour un premier lancement depuis les seed files.
    # Changer mode='continue_training' pour reprendre les runs incomplets.
    schedule_runs(
        mode='continue_training',   # <-- changer en 'train' pour un premier lancement
        directory=RUNS_DIR,         # <-- pour 'train', mettre CONFIG_DIR ici
        gpu_ids=[0, 1, 2, 3],
        runs_per_gpu=1,
        # target_epochs=20,         # <-- décommenter si tu veux forcer une cible différente
    )
else:
    # Fallback CPU — entraîne un seul modèle à la fois
    start_run(config_file=CONFIG_DIR / "seed_1.yml", gpu=-1)
```

**Pour un premier lancement depuis les seed files :**
```python
schedule_runs(mode='train', directory=CONFIG_DIR, gpu_ids=[0,1,2,3], runs_per_gpu=1)
```

**Pour reprendre les runs incomplets :**
```python
schedule_runs(mode='continue_training', directory=RUNS_DIR, gpu_ids=[0,1,2,3], runs_per_gpu=1)
```

**Pour reprendre un seul run spécifique manuellement :**
```python
continue_run(
    run_dir=Path("runs/V31_30ep-256N_ens4_2605_132454/"),
    gpu=0  # GPU à utiliser
)
```

### Script PBS `qsub_GPU_train_ensemble3.pbs`

Rien ne change dans le script PBS lui-même. La seule modification est dans le notebook. Le script PBS charge l'environnement et lance `ipython train_ensemble.ipynb` comme avant.

S'assurer que le script PBS utilise le nouvel environnement (pas l'ancien `nh_gpu` mamba) :

```bash
# Dans qsub_GPU_train_ensemble3.pbs — remplacer l'ancien bloc d'environnement par :
source /home/ega001/store8/machine_learning/neuralhydrology/.venv/bin/activate
```

Lancement :
```bash
qsub qsub_GPU_train_ensemble3.pbs
```

---

## Cas d'usage courants

### Cas 1 : Première session (lancement initial des 10 modèles)

Les 10 `seed_X.yml` sont dans `ens_configs/`. On lance 4 à la fois.

```python
schedule_runs(mode='train', directory=Path('ens_configs/'), gpu_ids=[0,1,2,3], runs_per_gpu=1)
```

Les configs traitées sont déplacées automatiquement dans `ens_configs/processed/`.

### Cas 2 : Reprise après une session interrompue (6h de walltime)

Les runs sont dans `runs/`. Certains ont atteint 15 epochs, d'autres 18, d'autres 0 (si la session GPU n'a pas démarré à temps). On reprend tous les incomplets en parallèle.

```python
schedule_runs(mode='continue_training', directory=Path('runs/'), gpu_ids=[0,1,2,3], runs_per_gpu=1)
```

Le scheduler lit `epochs: 30` dans chaque `config.yml` et relance uniquement ceux qui n'ont pas atteint 30.

### Cas 3 : Reprendre plusieurs jours plus tard

Même commande que le Cas 2. Le scheduler cherche récursivement le dernier checkpoint, peu importe s'il est dans `runs/V31_ens1_XXXX/` ou dans `runs/V31_ens1_XXXX/continue_training_from_epoch015/`. Il reprend depuis le bon epoch automatiquement.

Pour reprendre depuis une epoch antérieure à une epoch precise, il faut renommer ou supprimer les fichiers qui sont après cette epoch antérieure.

### Cas 4 : Vérifier l'état de tous les runs sans rien lancer

```bash
python - << 'EOF'
from pathlib import Path
from neuralhydrology.nh_run_scheduler import _get_last_completed_epoch, _get_target_epochs_from_config

runs_dir = Path("experiments/example/runs/")
for run_dir in sorted(runs_dir.iterdir()):
    if not run_dir.is_dir() or run_dir.name == "processed":
        continue
    last = _get_last_completed_epoch(run_dir)
    try:
        target = _get_target_epochs_from_config(run_dir)
    except Exception:
        target = "?"
    status = "OK" if isinstance(target, int) and last >= target else "INCOMPLET"
    print(f"{run_dir.name}: {last}/{target} epochs  [{status}]")
EOF
```

### Cas 5 : Un seul run à la fois sur un seul GPU (ancienne méthode manuelle)

```python
from neuralhydrology.nh_run import continue_run
from pathlib import Path

continue_run(
    run_dir=Path("runs/V31_30ep-256N_ens4_2605_132454/"),
    gpu=0
)
```

Aucune modification du seed file requise. Le scheduler reprend depuis le dernier checkpoint et s'arrête à `epochs: 30` dans le config.

---

## Points d'attention

**Ne plus modifier `epochs:` dans les seed files** avant une reprise. Cette valeur est maintenant la cible totale, pas le nombre d'epochs supplémentaires. Si le seed file dit `epochs: 30` et qu'on est à l'epoch 15, le code entraîne automatiquement les 15 epochs restants.

**Les fichiers `model_epoch*.pt` ne sont plus à copier manuellement** depuis les sous-dossiers `continue_training_from_epoch*/`. Le scheduler et le code de reprise trouvent le bon checkpoint tout seuls.

**En cas d'erreur "Already at epoch X, target is Y"** dans les logs, c'est que le modèle a déjà atteint sa cible. Le scheduler l'aurait normalement détecté et ignoré — si ce message apparaît, c'est que le run a été lancé manuellement avec `continue_run()` sur un run déjà terminé.
