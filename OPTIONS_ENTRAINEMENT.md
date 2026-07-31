# Options d'entraînement : nombre d'epochs, early stopping, learning rate dynamique, compile_model

> Ce document couvre les paramètres qui contrôlent *combien de temps* et *comment* un modèle s'entraîne — indépendamment de la reprise (`continue_training`, voir [CONTINUE_TRAINING.md](CONTINUE_TRAINING.md)).

---

## `epochs:` — le plafond, toujours actif

```yaml
epochs: 30
```

C'est le nombre maximum d'epochs. Que vous utilisiez l'early stopping ou non, l'entraînement ne dépassera jamais cette valeur. Avec `continue_training`, c'est la **cible totale**, pas le nombre d'epochs additionnels — voir [CONTINUE_TRAINING.md](CONTINUE_TRAINING.md) pour les détails.

Si vous ne voulez **pas** d'early stopping (juste un maximum fixe), c'est le seul paramètre à définir : laissez `early_stopping` absent ou à `False`.

---

## Early stopping — arrêter avant `epochs:` si ça ne s'améliore plus

### Ce qu'il vérifie

La **perte de validation** (`avg_total_loss` calculée sur le jeu de validation), pas la perte d'entraînement. Concrètement (`basetrainer.py`) :

```python
if self._early_stopping and epoch > self._minimum_epochs_before_early_stopping \
        and early_stopper.check_early_stopping(valid_metrics['avg_total_loss']):
    LOGGER.info(f"Early stopping triggered at epoch {epoch} ...")
    break
```

`EarlyStopper` (dans `training/earlystopper.py`) garde en mémoire la meilleure perte de validation vue jusqu'ici. À chaque nouvelle validation : si la perte s'améliore d'au moins `min_delta` (fixé à `0.0001` dans le code, pas configurable pour l'instant), le compteur de patience repart à zéro ; sinon il s'incrémente. Dès que le compteur atteint `patience_early_stopping`, l'entraînement s'arrête.

### Paramètres

```yaml
early_stopping: True
patience_early_stopping: 5                  # epochs sans amélioration avant d'arrêter
minimum_epochs_before_early_stopping: 10     # epochs à faire au minimum, quoi qu'il arrive
validate_every: 1                            # obligatoire, voir ci-dessous
```

`early_stopping: False` (ou absent) → comportement actuel, aucun changement, seul `epochs:` compte.

### Pourquoi `validate_every: 1` est obligatoire

Le code lève une erreur explicite si `early_stopping: True` et `validate_every != 1` :
```
ValueError: Early stopping can only be used if validation is performed every epoch (validate_every=1).
```
Raison : `patience_early_stopping` est compté en **epochs**, mais l'early stopping n'est vérifié qu'aux epochs de validation. Si `validate_every: 3`, une patience de "5" voudrait dire tantôt 5 epochs, tantôt 15 (5 validations × 3), selon comment on l'interprète — le code élimine l'ambiguïté en forçant une validation à chaque epoch.

**Coût réel** : vos seed files actuels ont `validate_every: 3` (validation sur 584 bassins une epoch sur trois). Passer à `validate_every: 1` **triple** ce coût. Sur vos runs, une validation prend dans les 15-20 minutes (ex: epoch 3 du run `ens1`, 19h37→19h56) — à multiplier par le nombre d'epochs. À peser contre le temps économisé en arrêtant plus tôt un modèle qui ne s'améliore plus.

### Limite avec `continue_training`

L'état de l'early stopping (compteur de patience, meilleure perte vue) **n'est pas sauvegardé sur disque** — chaque reprise le réinitialise à zéro. Vous verrez dans le log :
```
Early stopping state is reset.
```
Concrètement : un run repris après interruption regagne `patience_early_stopping` epochs de grâce, même s'il était sur le point de s'arrêter juste avant la coupure. Ce n'est pas un bug à proprement parler, plutôt une fonctionnalité non encore implémentée (persister l'état de l'`EarlyStopper` dans un fichier, comme les poids du modèle).

---

## Learning rate dynamique — réduire le LR automatiquement en cas de plateau

### Ce que ça fait

Utilise `torch.optim.lr_scheduler.ReduceLROnPlateau` : si la perte de validation ne s'améliore plus pendant `patience_dynamic_learning_rate` epochs, le learning rate est multiplié par `factor_dynamic_learning_rate`.

### Paramètres

```yaml
dynamic_learning_rate: True
patience_dynamic_learning_rate: 3
factor_dynamic_learning_rate: 0.5            # divise le LR par 2 à chaque déclenchement
validate_every: 1                            # obligatoire, même raison que pour early_stopping
```

### Incompatible avec le `learning_rate:` fixe de vos seed files actuels

Vos configs définissent un planning de LR fixe par epoch :
```yaml
learning_rate: {0: 0.0005, 20: 0.0001, 30: 5e-05}
```
Ce planning n'est appliqué **que si `dynamic_learning_rate` est False** (`basetrainer.py`) :
```python
if not self._dynamic_learning_rate:
    if epoch in self.cfg.learning_rate.keys():
        ...  # applique le LR fixe de cet epoch
```
Si vous activez `dynamic_learning_rate: True`, ce dictionnaire `learning_rate:` est **complètement ignoré** — le scheduler automatique prend seul le contrôle du LR à partir de sa valeur initiale (`learning_rate[0]`). Ce n'est pas un bug, c'est le comportement voulu (les deux mécanismes sont mutuellement exclusifs), mais gardez-le en tête si vous activez l'un en pensant garder l'autre.

### Un bug corrigé au passage

Avant le 31 juillet 2026, `dynamic_learning_rate` dans `config.py` lisait par erreur la clé `"early_stopping"` (copier-coller). Conséquence : `dynamic_learning_rate: True` seul ne faisait **rien**, et `early_stopping: True` seul activait **silencieusement aussi** le scheduler de LR dynamique, sans le demander. Corrigé — chaque option lit maintenant sa propre clé, indépendamment de l'autre. Si vous aviez déjà `early_stopping: True` quelque part avant cette date en pensant n'activer que l'early stopping, sachez que le LR a peut-être aussi été ajusté dynamiquement sans que ce soit voulu — vérifiez `output.log` pour d'éventuelles lignes de réduction de LR sur ces runs-là.

### Même limite avec `continue_training`

Même chose que l'early stopping : l'état du scheduler (l'historique de plateau) n'est pas sauvegardé, log `"Scheduler state is reset."` à chaque reprise.

---

## `compile_model` — désactiver `torch.compile()` si besoin

```yaml
compile_model: False   # défaut : True
```

Ajouté suite au crash `torch._inductor.exc.InductorError: OSError: Disk quota exceeded` (Triton, qui compile les kernels GPU, a besoin d'un cache disque). Deux raisons de mettre `False` :
- **Quota disque** : si le cache Triton (`TRITON_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR`) n'est pas redirigé vers un espace avec assez de place — voir [CONTINUE_TRAINING.md](CONTINUE_TRAINING.md), section clones multiples et quota.
- **LSTM peu adapté au compile** : `CudaLSTM` utilise `torch.nn.LSTM`, déjà optimisé via des kernels cuDNN fusionnés. `torch.compile()`/Inductor apporte surtout des gains sur des architectures type Transformer/CNN ; sur du LSTM, le gain est parfois nul, voire négatif, une fois l'overhead de compilation pris en compte. Si vous constatez qu'une reprise avec compile est *plus lente* qu'une session équivalente sans, comparez avec `compile_model: False`.

---

## Exemple complet pour un `seed_x.yml`

```yaml
epochs: 30
early_stopping: True
patience_early_stopping: 5
minimum_epochs_before_early_stopping: 10
validate_every: 1
compile_model: True     # ou False si le cache Triton pose problème / pas de gain observé sur LSTM
```

Si vous préférez garder votre planning de LR fixe actuel (`learning_rate: {0: ..., 20: ..., 30: ...}`), laissez `dynamic_learning_rate` absent ou `False` — les deux options ne coexistent pas.

---

## Erreurs courantes

| Ce que vous voyez | Cause |
|---|---|
| `ValueError: Early stopping can only be used if validation is performed every epoch (validate_every=1).` | `early_stopping: True` avec `validate_every` différent de `1`. Passez `validate_every: 1`. |
| `ValueError: Dynamic learning rate can only be used if validation is performed every epoch (validate_every=1).` | Même chose pour `dynamic_learning_rate: True`. |
| Le planning `learning_rate: {...}` semble ignoré | `dynamic_learning_rate: True` est actif — il prend le contrôle exclusif du LR, le planning fixe n'est appliqué que si `dynamic_learning_rate` est `False`. |
| `"Early stopping state is reset."` / `"Scheduler state is reset."` dans le log | Normal lors d'un `continue_training` — ces deux états ne sont pas persistés entre sessions (voir sections dédiées ci-dessus). |
