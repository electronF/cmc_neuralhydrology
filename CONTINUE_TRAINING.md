# Reprendre un entraînement (`continue_training`)

> Ce document explique comment reprendre l'entraînement d'un modèle qui a été interrompu (walltime PBS dépassé, crash, arrêt manuel), pour un seul run ou pour un ensemble sur plusieurs GPUs. Il documente aussi ce qui a changé dans le code et pourquoi.
>
> Pour les paramètres qui contrôlent la durée/l'arrêt d'un entraînement (early stopping, learning rate dynamique, `epochs:`, `compile_model`), voir [OPTIONS_ENTRAINEMENT.md](OPTIONS_ENTRAINEMENT.md).

---

## Le principe en une phrase

`epochs:` dans le fichier de config **n'est plus** "le nombre d'epochs à faire" — c'est **la cible totale**. Si le run est déjà à l'epoch 13 et que `epochs: 30`, reprendre l'entraînement fait les epochs 14 à 30, pas 30 de plus. On ne touche jamais à `epochs:` à la main entre deux sessions.

---

## Reprendre un seul run

```python
from neuralhydrology.nh_run import continue_run
from pathlib import Path

continue_run(
    run_dir=Path("runs/V31_30ep-256N_ens4_2605_132454/"),
    gpu=0
)
```

`run_dir` pointe vers le dossier du run **existant** (celui créé par le premier `start_run`), pas vers un fichier `.yml`. Le code va :

1. Lire `run_dir/config.yml` pour connaître la config d'origine.
2. Chercher le dernier checkpoint (`model_epoch*.pt`) dans `run_dir` pour savoir à quel epoch reprendre.
3. Continuer l'entraînement à partir de là, jusqu'à `epochs:` (la cible du config).
4. Écrire les nouveaux `model_epoch0XX.pt` / `optimizer_state_epoch0XX.pt` **directement dans `run_dir`**, et continuer à écrire dans le même `output.log`.

Si `run_dir` n'a aucun checkpoint (aucun `model_epoch*.pt`), ça échoue avec une erreur claire (`FileNotFoundError`) — ça ne repart jamais silencieusement de zéro.

---

## Reprendre plusieurs runs en parallèle (multi-GPU)

```python
from neuralhydrology.nh_run_scheduler import schedule_runs
from pathlib import Path

schedule_runs(
    mode='continue_training',
    directory=Path("runs/"),      # dossier qui contient les sous-dossiers de runs
    gpu_ids=[0, 1, 2, 3],
    runs_per_gpu=1,
)
```

Le scheduler scanne chaque sous-dossier de `directory` :

- lit l'epoch du dernier checkpoint sauvegardé,
- lit la cible `epochs:` dans le `config.yml` du run (ou utilise `target_epochs=` si fourni en argument),
- met en file d'attente uniquement les runs dont `dernier epoch < cible`,
- ignore (sans erreur) les runs déjà à leur cible,
- lance `continue_run` sur chacun, répartis sur les GPUs disponibles.

**Important : `directory` doit pointer vers le dossier des runs (`runs/`), pas vers celui des seed files (`ens_configs/`).** Pointer vers le mauvais dossier, ou vers un dossier vide, ne produit **aucune erreur** avec `mode='continue_training'` — le scheduler affiche juste `"All runs are already complete. Nothing to do."` et s'arrête. C'est la cause la plus probable si vous ne voyez "rien se passer" sans message d'erreur.

### Mode strict : `continue_training_only`

Si vous voulez être certain de ne jamais tomber dans ce silence (mauvais chemin, dossier vide, rien n'a jamais été lancé), utilisez `continue_training_only` à la place de `continue_training` :

```python
schedule_runs(
    mode='continue_training_only',
    directory=Path("runs/"),
    gpu_ids=[0, 1, 2, 3],
    runs_per_gpu=1,
)
```

Comportement identique, sauf qu'il lève un `RuntimeError` explicite si **aucun** run du dossier n'a jamais sauvegardé le moindre checkpoint — au lieu de se taire.

---

## Vérifier l'état de tous les runs sans rien lancer

```python
from pathlib import Path
from neuralhydrology.nh_run_scheduler import _get_last_completed_epoch, _get_target_epochs_from_config

runs_dir = Path("runs/")
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
```

---

## Ce qui a changé (et pourquoi c'était cassé avant)

Avant, chaque reprise créait un **nouveau sous-dossier** `run_dir/continue_training_from_epochXXX/`, et y redirigeait aussi bien le `output.log` que les nouveaux poids. Résultat : en regardant le `output.log` du dossier de base, on ne voyait jamais rien bouger — l'activité réelle était écrite ailleurs, dans un sous-dossier facile à ne pas remarquer. Ce n'était pas un blocage : l'entraînement reprenait bien au bon epoch, mais dans un endroit différent de celui qu'on surveillait.

Depuis le correctif, une reprise écrit **directement dans le dossier du run d'origine** : plus de sous-dossier, `output.log` s'accumule (append) au même endroit, les checkpoints suivent la même séquence de noms de fichiers (`model_epoch014.pt`, `015`, ...) que ceux d'avant.

Les runs qui ont déjà un vieux sous-dossier `continue_training_from_epochXXX/` (créés avant ce correctif) restent lisibles : la recherche du dernier checkpoint (dans le trainer, dans le scheduler, et dans l'évaluation) regarde toujours à la fois la racine du run **et** ces anciens sous-dossiers. Seuls les epochs futurs iront directement à la racine.

Autres correctifs déjà en place dans le code actuel :
- `torch.load(..., weights_only=False)` — évite le crash `_pickle.UnpicklingError` avec PyTorch ≥ 2.6 lors du chargement de l'état de l'optimizer.
- Recherche récursive du dernier checkpoint (`run_dir` + anciens sous-dossiers `continue_training_from_epoch*/`) dans le trainer, le scheduler, **et** l'évaluation (`evaluation/tester.py`) — avant, l'évaluation ne regardait que la racine et pouvait charger le mauvais (ancien) checkpoint sur un run repris.

---

## Le bug `torch.compile()` / `_orig_mod.` (clés de checkpoint qui ne correspondent plus)

### Symptôme

```
RuntimeError: Error(s) in loading state_dict for OptimizedModule:
	Missing key(s) in state_dict: "_orig_mod.lstm.weight_ih_l0", "_orig_mod.lstm.weight_hh_l0", ...
	Unexpected key(s) in state_dict: "lstm.weight_ih_l0", "lstm.weight_hh_l0", ...
```
en essayant de reprendre un run (`continue_run` / `continue_training`).

### Ce qu'est `torch.compile()`

Le correctif de performance du 6 juillet a ajouté `torch.compile(self.model)` pour accélérer l'entraînement. `torch.compile()` ne modifie pas le modèle — il l'enveloppe dans un objet `OptimizedModule` qui stocke le vrai modèle dans un attribut interne `_orig_mod`. Conséquence directe : le `state_dict()` de l'objet enveloppé a toutes ses clés préfixées par `_orig_mod.` :

| Modèle non compilé | Modèle compilé (`torch.compile`) |
|---|---|
| `lstm.weight_ih_l0` | `_orig_mod.lstm.weight_ih_l0` |
| `head.net.0.weight` | `_orig_mod.head.net.0.weight` |

C'est littéralement le même modèle, seul le nommage des clés change.

### Pourquoi ça a cassé une reprise précise

Chronologie typique (vue sur le run `V31_30ep-256N_ens7`) :

1. **Session 1** : le job tourne avec le code **avant** l'ajout de `torch.compile()`. Le modèle est sauvegardé "brut" → `model_epoch012.pt` contient des clés simples (`lstm.weight_ih_l0`, ...).
2. **Session 2, quelques jours plus tard** : on reprend l'entraînement avec le code **après** l'ajout de `torch.compile()`. Le code construit le modèle, le compile (devient un `OptimizedModule`), puis essaie de charger `model_epoch012.pt` dedans.
3. PyTorch compare les clés attendues par l'objet compilé (`_orig_mod.lstm...`) à celles du fichier (`lstm...`, sans préfixe) → aucune ne correspond → `RuntimeError`.

Ce n'est donc pas une histoire d'epoch, de dossier ou de fichier corrompu — uniquement un désaccord de **nommage des clés**, causé par le changement de version de code entre les deux sessions.

### Pourquoi ce n'était pas juste "un problème pour cette fois"

Le même bug aurait resurgi ailleurs, dans l'autre sens : une fois `continue_training` en train de sauvegarder de nouveaux checkpoints (epoch 13, 14, ...) via le modèle compilé, ces fichiers auraient eu des clés préfixées `_orig_mod.`. Puis, en évaluant ce run avec `evaluation/tester.py` — qui, lui, ne compile jamais le modèle — le chargement aurait de nouveau échoué, cette fois dans le sens inverse (modèle brut essayant de lire des clés préfixées).

### Le correctif

Dans [basetrainer.py](neuralhydrology/training/basetrainer.py), une méthode `_raw_model()` retourne toujours le modèle **sans** son enveloppe de compilation :

```python
def _raw_model(self) -> torch.nn.Module:
    return getattr(self.model, '_orig_mod', self.model)
```

`getattr(x, '_orig_mod', x)` veut dire : *"si `self.model` a un attribut `_orig_mod` (donc s'il est compilé), utilise-le ; sinon utilise `self.model` tel quel."*

Toutes les sauvegardes et tous les chargements de poids (`checkpoint_path`, finetuning, `_restore_training_state` pour `continue_training`, et la sauvegarde à chaque epoch) passent maintenant par `_raw_model()`. Résultat : les fichiers `.pt` sur disque ont toujours des clés simples, que le modèle soit compilé ou non au moment de la sauvegarde/du chargement — donc compatibles entre anciennes sessions, nouvelles sessions, `continue_training`, et l'évaluation, indéfiniment.

---

## Le bug `FileExistsError: config.yml` (effet de bord du correctif du sous-dossier)

### Symptôme

```
File ".../neuralhydrology/training/logger.py", line 41, in __init__
    cfg.dump_config(folder=self.log_dir)
File ".../neuralhydrology/utils/config.py", line 135, in dump_config
    raise FileExistsError(yml_path)
FileExistsError: .../runs/V31_30ep-256N_ens1_2407_173608/config.yml
```
alors que ce `config.yml` est légitimement présent — ce n'est pas un fichier corrompu ou en trop.

### Pourquoi

`Logger.__init__()` sauvegarde une copie de la config dans le dossier du run à chaque démarrage, via `dump_config()`. Cette méthode a toujours eu un garde-fou : si `config.yml` existe déjà dans le dossier cible, elle refuse d'écrire et lève une erreur plutôt que d'écraser silencieusement un fichier existant (utile pour éviter d'effacer un vieux run par erreur).

Avant le correctif du sous-dossier (section plus haut), ce garde-fou ne posait jamais de problème pour `continue_training`, puisque chaque reprise écrivait dans un **nouveau** sous-dossier `continue_training_from_epochXXX/` qui n'avait encore jamais de `config.yml`. Une fois ce correctif appliqué — la reprise écrit maintenant directement dans le dossier d'origine — le `config.yml` de la toute première session s'y trouve déjà, et `dump_config()` refuse de l'écraser. C'est un effet de bord du premier correctif que je n'avais pas anticipé.

### Le correctif

`dump_config()` accepte maintenant un paramètre `overwrite` :

```python
def dump_config(self, folder: Path, filename: str = 'config.yml', overwrite: bool = False):
    yml_path = folder / filename
    if overwrite or not yml_path.exists():
        ...  # écrit le fichier
    else:
        raise FileExistsError(yml_path)
```

Et `Logger` l'active automatiquement pour les reprises :

```python
cfg.dump_config(folder=self.log_dir, overwrite=cfg.is_continue_training)
```

Le garde-fou reste actif pour tous les autres cas (`train`, `finetune`) — seul `continue_training`, où l'on sait qu'on réécrit légitimement dans le même dossier, l'ignore. En bonus, `config.yml` se retrouve à jour à chaque reprise avec le `commit_hash` et le `package_version` de la session qui a fait la reprise — utile pour savoir avec quelle version du code chaque portion de l'entraînement a réellement tourné (voir aussi la section suivante sur les clones multiples).

---

## Attention aux clones multiples sur le cluster

Un piège récurrent, indépendant du code : si plusieurs copies de ce dépôt existent sur le cluster (ex: `neuralhydrology/`, `cmc_neuralhydrology-test/`, `cmc_neuralhydrology-test-new/`), rien ne garantit qu'un `.pbs` donné active la version que vous croyez. Deux sessions consécutives peuvent utiliser deux clones différents — avec des correctifs différents — sans qu'aucune erreur ne le signale, jusqu'à ce qu'un des deux plante pour une raison que l'autre avait déjà réglée.

À vérifier si un comportement semble incohérent d'une session à l'autre :
```bash
grep "commit_hash\|package_version" runs/<nom_du_run>/output.log
```
Ces deux valeurs sont maintenant écrites à chaque session (voir section précédente) — si elles changent de manière inattendue entre deux reprises du même run, c'est le signe que le `.pbs` (ou le venv qu'il active) pointe vers des clones différents. Le plus sûr est de n'avoir qu'un seul clone de référence sur le cluster, et de vérifier que le `.pbs` l'active explicitement par son chemin complet.

---

## Erreurs courantes

| Ce que vous voyez | Cause probable |
|---|---|
| `"All runs are already complete. Nothing to do."` alors que rien n'a jamais tourné | `directory` pointe vers le mauvais dossier (ex: `ens_configs/` au lieu de `runs/`), ou vers un dossier vide. Utilisez `mode='continue_training_only'` pour avoir une erreur explicite à la place. |
| `FileNotFoundError: No model checkpoint found in ...` | Le `run_dir` donné n'a jamais sauvegardé de checkpoint — la reprise ne peut pas savoir où continuer. Vérifiez que l'entraînement initial a bien tourné jusqu'à au moins un `save_weights_every`. |
| `"Already at epoch X, target is Y. Nothing to train."` | Le run a déjà atteint sa cible `epochs:`. Normal, rien à faire — sauf si vous voulez pousser plus loin, auquel cas augmentez `epochs:` dans le `config.yml` du run avant de reprendre. |
| Rien de nouveau dans `output.log` pendant des heures, mais le job PBS tourne | Symptôme de l'ancien bug (voir section précédente) — corrigé. Si ça persiste après mise à jour du code, vérifier que le job n'est pas bloqué au chargement des données (`nvidia-smi`, `ps`, `py-spy dump`). |
| `RuntimeError: ... Missing key(s) ... "_orig_mod.lstm..." / Unexpected key(s) ... "lstm..."` | Bug `torch.compile()` / `_orig_mod.` — voir section dédiée ci-dessus. Corrigé ; ne devrait plus apparaître une fois le code à jour déployé. |
| `FileExistsError: .../config.yml` en pleine reprise, alors que ce fichier est légitime | Effet de bord du correctif du sous-dossier — voir section dédiée ci-dessus. Corrigé ; ne devrait plus apparaître une fois le code à jour déployé. |
| Comportement différent entre deux reprises du même run, sans raison apparente | Le `.pbs` a probablement activé deux clones/venvs différents du dépôt entre les deux sessions. Comparer `commit_hash` / `package_version` dans `output.log` entre les deux — voir section « Attention aux clones multiples » ci-dessus. |

---

## Vérifier que le code déployé est à jour

Ce correctif vit dans ce dépôt Git. S'il est utilisé depuis un autre emplacement (ex: un clone séparé sur un cluster de calcul), il faut que cet emplacement soit synchronisé avec ce dépôt pour bénéficier des correctifs :

```bash
git -C /chemin/vers/le/clone/sur/le/cluster log -1 --oneline
grep -n "weights_only=False" neuralhydrology/training/basetrainer.py
```

Si la commande `git log` ne montre pas les commits récents, ou si le `grep` ne trouve rien, le code déployé est une version antérieure au correctif — il faut le mettre à jour (`git pull` ou resynchronisation) avant de relancer une reprise.

---

## Tester rapidement sans attendre des heures

Pour valider que `continue_training` fonctionne, inutile d'attendre un vrai walltime de plusieurs heures. Simulez une interruption :

1. Dans un seed file de test, mettez `epochs: 2` (et si possible un `train_basin_file` réduit à 1-2 bassins pour aller vite).
2. Lancez-le normalement avec `start_run` — le run se termine, produit `model_epoch001.pt` et `model_epoch002.pt`.
3. Éditez `epochs: 4` dans le `config.yml` généré dans le dossier du run.
4. Lancez `continue_run(run_dir=...)` dessus.
5. Vérifiez : l'entraînement reprend à l'epoch 3 (pas 1), les nouveaux poids apparaissent dans le même dossier, `output.log` s'allonge au lieu d'être remplacé.

Ça exerce exactement le même chemin de code que la vraie reprise, en quelques minutes.
