# Recette de cuisine — GRIP-GL LSTM sur les Grands Lacs

> **Comment utiliser ce document :** Lis chaque étape dans l'ordre. Ne passe pas à l'étape suivante avant que la vérification soit faite. Les blocs de code sont des commandes à copier-coller dans le terminal.
>
> **Durée totale estimée :** 2 à 5 jours (dominé par les étapes 7 et 9 qui tournent en batch)
>
> **Pré-requis :** Un compte Science ECCC avec accès aux serveurs ppp7/ppp8 et aux GPU (Moira ou Conrad).

---

## Variables de départ — définis-les une seule fois

> **Copie ce bloc au début de chaque nouvelle session de terminal.** Ces variables sont utilisées dans toutes les commandes qui suivent.

```bash
export USERID=$(whoami)                          # ton identifiant Science (ex: ovk001)
export SSTORE="store8"                           # ton sitestore (adapter si différent : store86, store5...)
export PROJ_DIR="/home/$USERID/$SSTORE/GripGl_LSTM/new_bassin"
export EXPERIMENT_NAME="Gl_Lstm_Exp1"
export NH_DIR="/home/$USERID/$SSTORE/machine_learning/neuralhydrology"
```

> **Pourquoi :** Ces variables évitent de réécrire les longs chemins à chaque commande. Si ton sitestore n'est pas `store8`, change-le ici et tout le reste s'adapte.

---

## ÉTAPE 1 — Préparer l'environnement de calcul

> **Ce que ça fait :** Réserver une machine de calcul, activer Internet, installer Python et NeuralHydrology.
> **À faire :** Une seule fois, jamais à refaire (sauf si l'environnement est supprimé).
> **Durée :** ~30–45 minutes.

### 1.1 — Réserver une session interactive

Connecte-toi d'abord au serveur de connexion Science, puis réserve une machine de calcul :

```bash
qsub -I -lselect=1:ncpus=1:mpiprocs=1:ompthreads=1:mem=32gb -l walltime=6:0:0
```

> Attends que le terminal affiche une invitation comme `[ovk001@ppp8]`. Tu es maintenant sur une machine de calcul dédiée. **Ne fais jamais d'installation depuis le nœud de connexion.**

### 1.2 — Activer Internet et redéfinir les variables

```bash
# Redéfinir les variables (nouveau shell = variables perdues)
export USERID=$(whoami)
export SSTORE="store8"
export PROJ_DIR="/home/$USERID/$SSTORE/GripGl_LSTM/new_bassin"
export NH_DIR="/home/$USERID/$SSTORE/machine_learning/neuralhydrology"

# Activer Internet
Internet

# Configurer les proxys
export https_proxy=http://webproxy.science.gc.ca:8888/
export http_proxy=http://webproxy.science.gc.ca:8888/
```

### 1.3 — Rediriger les caches volumineux vers le sitestore

> Sans ça, l'installation remplit ton quota `$HOME` et échoue à mi-chemin.

```bash
mkdir -p "/home/$USERID/$SSTORE/machine_learning/uv_cache"
mkdir -p "/home/$USERID/$SSTORE/machine_learning/uv_local/share/uv"
mkdir -p "$HOME/.cache"
mkdir -p "$HOME/.local/share"

ln -sf "/home/$USERID/$SSTORE/machine_learning/uv_cache" "$HOME/.cache/uv"
ln -sf "/home/$USERID/$SSTORE/machine_learning/uv_local/share/uv" "$HOME/.local/share/uv"

# Lien pour conda (optionnel mais recommandé)
ln -s "/home/$USERID/$SSTORE/.conda" "$HOME/.conda"
```

### 1.4 — Installer `uv` et cloner NeuralHydrology

```bash
# Installer uv (gestionnaire de paquets Python rapide)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Créer le dossier et cloner NeuralHydrology
mkdir -p "/home/$USERID/$SSTORE/machine_learning"
cd "/home/$USERID/$SSTORE/machine_learning"
git clone https://github.com/neuralhydrology/neuralhydrology.git

# Créer l'environnement virtuel Python avec toutes les dépendances
cd neuralhydrology
uv sync --all-groups

# Activer l'environnement et installer les extras
source .venv/bin/activate
uv pip install -e .
uv pip install regex openpyxl ipython
```

### 1.5 — Vérification ✅

```bash
python -c "import neuralhydrology; print('NeuralHydrology OK')"
python -c "import torch; print('PyTorch', torch.__version__)"
```

> **Attendu :** Pas d'erreur. Si tu vois `NeuralHydrology OK`, l'environnement est prêt.

---

## ACTIVATION À RÉPÉTER À CHAQUE SESSION

> **Important :** Chaque fois que tu ouvres un nouveau terminal ou une nouvelle session `qsub`, tu dois répéter ces 4 lignes avant de lancer quoi que ce soit.

```bash
export USERID=$(whoami)
export SSTORE="store8"
export PROJ_DIR="/home/$USERID/$SSTORE/GripGl_LSTM/new_bassin"
export NH_DIR="/home/$USERID/$SSTORE/machine_learning/neuralhydrology"
source $HOME/.local/bin/env
source "$NH_DIR/.venv/bin/activate"
```

---

## ÉTAPE 2 — Créer la structure du projet et centraliser les données

> **Ce que ça fait :** Crée tous les dossiers nécessaires et copie les données depuis les comptes des autres chercheurs (ega001, mba002, shyd500) vers ton propre espace.
> **À faire :** Une seule fois par nouveau projet.
> **Durée :** ~1–2 heures (dominé par la copie des fichiers NetCDF CaSR, plusieurs dizaines de Go).

### 2.1 — Créer l'arborescence des dossiers

```bash
echo "-> Création de la structure dans : $PROJ_DIR"

mkdir -p "$PROJ_DIR/data/basins"
mkdir -p "$PROJ_DIR/data/shape_files"
mkdir -p "$PROJ_DIR/data/NETCDF_FOLDER"
mkdir -p "$PROJ_DIR/data/grid_weights"
mkdir -p "$PROJ_DIR/data/discharge_data"
mkdir -p "$PROJ_DIR/scripts"
mkdir -p "$PROJ_DIR/data/source_shapes/CaPA"
mkdir -p "$PROJ_DIR/data/source_shapes/WSC_shapes_new"
mkdir -p "$PROJ_DIR/data/source_shapes/old_212_shapefiles"
mkdir -p "$PROJ_DIR/data/source_shapes/US_shapes_all"
mkdir -p "$PROJ_DIR/EXPERIMENT/$EXPERIMENT_NAME/attributes"
mkdir -p "$PROJ_DIR/EXPERIMENT/$EXPERIMENT_NAME/basins"
mkdir -p "$PROJ_DIR/EXPERIMENT/$EXPERIMENT_NAME/time_series"
mkdir -p "$PROJ_DIR/EXPERIMENT/$EXPERIMENT_NAME/time_series_csv"
mkdir -p "$PROJ_DIR/EXPERIMENT/Apply_$EXPERIMENT_NAME/ens_configs/processed"
mkdir -p "$PROJ_DIR/EXPERIMENT/Apply_$EXPERIMENT_NAME/results"
mkdir -p "$PROJ_DIR/EXPERIMENT/Apply_$EXPERIMENT_NAME/runs"
mkdir -p "$PROJ_DIR/geophys/grid_weights"
mkdir -p "$PROJ_DIR/geophys/shape_files"
mkdir -p "$PROJ_DIR/geophys/stat_attrs"
mkdir -p "$PROJ_DIR/listings"
mkdir -p "$PROJ_DIR/time_series_csv"

echo "[OK] Arborescence créée."
```

### 2.2 — Copier les métadonnées, scripts et listes de stations

```bash
echo "-> Copie des métadonnées et listes..."
cp /home/ega001/store8/machine_learning/cal_val_CaSR3.2_EG/data/LSTM_CaSR3.2_forcings_2000_2023/basins/all_212_basins.txt "$PROJ_DIR/data/basins/"
cp /home/mba002/LSTM_scripts/marie_scripts/StationID_NHSvsQC.xlsx "$PROJ_DIR/data/basins/"
cp /home/mba002/LSTM_scripts/marie_scripts/new_bassin/*.obs "$PROJ_DIR/data/basins/"
cp /home/ega001/store8/Qobs/Hydat/WSC_gauges_GRIPGL.csv "$PROJ_DIR/data/basins/"

echo "-> Copie des scripts de calcul..."
cp /home/mba002/Gitlab/lstm/derive_grid_weights_new_WSC.py "$PROJ_DIR/scripts/"
cp /home/ega001/store8/machine_learning/prep_static_attrs/*.ipynb "$PROJ_DIR/scripts/"
cp /home/ega001/store8/machine_learning/prep_static_attrs/*.py "$PROJ_DIR/scripts/"
```

### 2.3 — Copier les fichiers NetCDF CaSR (gros fichiers — peut être long)

```bash
echo "-> Copie des forçages NetCDF CaSR 3.2 (2000–2024)..."
echo "   Ceci peut prendre 1–2 heures selon la taille..."
cp -rp /home/ega001/store8/sps/experiments/CaSR_extract/CaSR3.2_GLS_nc/* "$PROJ_DIR/data/NETCDF_FOLDER/"

echo "[OK] NetCDF copiés. Vérifie : $(ls $PROJ_DIR/data/NETCDF_FOLDER/ | wc -l) fichiers"
```

> **Attendu :** ~8 000+ fichiers `.nc` (un par jour de 2000 à 2024).

### 2.4 — Copier les débits observés et les shapefiles sources

```bash
echo "-> Copie des débits observés (déjà en format 12-12 UTC)..."
cp /home/mba002/LSTM/new_bassin/data/discharge_data/* "$PROJ_DIR/data/discharge_data/"

echo "-> Copie des banques de shapefiles..."
cp -r /home/shyd500/data/sitestore7/Maestro/shapefiles/CaPA/* "$PROJ_DIR/data/source_shapes/CaPA/"
cp -r /home/ega001/store8/geophys_data/WSC_shapes_new/MDA_ADP_02_split_new/* "$PROJ_DIR/data/source_shapes/WSC_shapes_new/"
cp -r /home/mba002/LSTM/cal_val_CaSR3.2_MB/data/shape_files/* "$PROJ_DIR/data/source_shapes/old_212_shapefiles/"
cp -r /home/ega001/store8/geophys_data/US_shapefiles/all/* "$PROJ_DIR/data/source_shapes/US_shapes_all/"
# Ajouter aussi les autres familles WSC si nécessaire (MDA_ADP_01, etc.)
cp -r /home/ega001/store8/geophys_data/WSC_shapes_new/MDA_ADP_01_split_new/* "$PROJ_DIR/data/source_shapes/WSC_shapes_new/"

echo "[OK] Centralisation terminée."
```

### 2.5 — Vérification ✅

```bash
echo "Débits :  $(ls $PROJ_DIR/data/discharge_data/ | wc -l) fichiers"
echo "NetCDF :  $(ls $PROJ_DIR/data/NETCDF_FOLDER/ | wc -l) fichiers"
echo "Shapefiles CA :  $(ls $PROJ_DIR/data/source_shapes/WSC_shapes_new/ | wc -l) fichiers"
echo "Shapefiles US :  $(ls $PROJ_DIR/data/source_shapes/US_shapes_all/ | wc -l) fichiers"
```

---

## ÉTAPE 3 — Préparer les shapefiles et les listes de stations

> **Ce que ça fait :** Sélectionne le bon shapefile pour chacune des 212 stations, les copie dans `data/shape_files/`, et génère les listes de stations utilisées par tous les scripts suivants.
> **Durée :** ~15–30 minutes.

### 3.1 — Ouvrir une session interactive et activer l'environnement

```bash
qsub -I -lselect=1:ncpus=1:mpiprocs=1:ompthreads=1:mem=32gb -l walltime=6:0:0
# (attendre la session)

export USERID=$(whoami)
export SSTORE="store8"
export PROJ_DIR="/home/$USERID/$SSTORE/GripGl_LSTM/new_bassin"
export NH_DIR="/home/$USERID/$SSTORE/machine_learning/neuralhydrology"
source $HOME/.local/bin/env
source "$NH_DIR/.venv/bin/activate"

cd "$PROJ_DIR/scripts"
```

### 3.2 — Modifier les chemins dans le notebook avant de le lancer

Avant d'exécuter `01a_merge_bassin.ipynb`, ouvre-le dans VS Code et vérifie/modifie ces variables dans les premières cellules :

```python
# Dans 01a_merge_bassin.ipynb — adapter ces chemins :
old_basin_list = "$PROJ_DIR/data/basins/all_212_basins.txt"
obs_dir = "$PROJ_DIR/data/basins/"                      # dossier des fichiers .obs
station_excel = "$PROJ_DIR/data/basins/StationID_NHSvsQC.xlsx"

destination_repo_list = "$PROJ_DIR/data/basins"         # où sauver les listes de stations
destination_repo = "$PROJ_DIR/data/shape_files"         # où copier les shapefiles

# Dossiers de shapefiles sources (dans l'ordre de priorité)
wsc_dir     = "$PROJ_DIR/data/source_shapes/WSC_shapes_new/"
usgs_dir    = "$PROJ_DIR/data/source_shapes/US_shapes_all/"
capa_dir    = "$PROJ_DIR/data/source_shapes/CaPA/"
old_dir     = "$PROJ_DIR/data/source_shapes/old_212_shapefiles/"
```

### 3.3 — Lancer le notebook

```bash
ipython 01a_merge_bassin.ipynb
```

### 3.4 — Vérification ✅

```bash
echo "Shapefiles copiés : $(ls $PROJ_DIR/data/shape_files/*.shp | wc -l)"
echo "Liste de stations :"
ls $PROJ_DIR/data/basins/*.txt
```

> **Attendu :** ~212 fichiers `.shp` dans `data/shape_files/`. Si des shapefiles manquent, le script le mentionne — note les stations manquantes et cherche leurs shapefiles manuellement (voir la section "Si un shapefile manque" plus bas).

### SI UN SHAPEFILE MANQUE

Si un bassin canadien n'est pas trouvé, sa "famille" WSC n'est peut-être pas encore dans `source_shapes/WSC_shapes_new/`. Solution :

1. Sur Windows, ouvre le gros fichier WSC de la famille dans QGIS
   - Ex: `MDA_ADP_05_DrainageBasin.shp` pour les stations `05XXXX`
2. Exporte en `EPSG:4326 (WGS 84)`, type `Polygon`
3. Processing Toolbox → Vector general → **Split Vector Layer**
   - Champ clé : `StationNum`
   - Format de sortie : `.shp` (Settings → Options → Processing → default output = shp)
4. Transfère les shapefiles individuels vers Science :
   ```
   sftp://hpcr6-in.science.gc.ca
   ```
5. Copie vers `$PROJ_DIR/data/source_shapes/WSC_shapes_new/`
6. Relance le notebook

---

## ÉTAPE 4 — Données de débit observé

> **Ce que ça fait :** Extrait les mesures de débit journalier pour chaque station (Hydat pour le Canada, USGS pour les États-Unis) et les convertit au format 12h UTC → 12h UTC.
>
> **Si les débits existent déjà :** Tu les as copiés à l'étape 2.4 depuis `mba002`. Vérifie avec :
> ```bash
> ls $PROJ_DIR/data/discharge_data/ | head -5
> ```
> Si tu vois des fichiers, **passe directement à l'étape 5.**

### 4.1 — Si les débits doivent être régénérés (stations canadiennes)

```bash
cd "$PROJ_DIR/scripts"

# Extraire les débits depuis Hydat (adapter START_DATE et END_DATE)
./01b1_get_obs_hydat_EG_new.py \
    "$PROJ_DIR/data/basins/WSC_gauges_GRIPGL.csv" \
    FLOW \
    /home/ega001/store8/Qobs/Hydat/Hydat_20260116.sqlite3 \
    20000101 20241231 GLS_flow

# Convertir en journalier 12-12 UTC (fuseau Est UTC-5)
./01b2_convert_daily_WSC_in_12-12_UTC.py
```

> **Attention fuseau horaire :** Ce script suppose UTC-5 (heure normale de l'Est). Pour la Colombie-Britannique (UTC-7), modifier la ligne ~47 du script.

### 4.2 — Si les débits doivent être régénérés (stations américaines)

```bash
# Récupérer les données USGS depuis internet (proxy requis)
Internet
export https_proxy=http://webproxy.science.gc.ca:8888/
export http_proxy=http://webproxy.science.gc.ca:8888/

python 01c1_USGS_txtfiles_reading.py
python 01c2_convert_daily_USGS_in_12-12_UTC.py
```

### 4.3 — Vérification ✅

```bash
echo "Débits disponibles : $(ls $PROJ_DIR/data/discharge_data/ | wc -l) fichiers"
ls $PROJ_DIR/data/discharge_data/ | head -3
```

---

## ÉTAPE 5 — Calculer les attributs statiques

> **Ce que ça fait :** Calcule les caractéristiques physiques de chaque bassin (sol, végétation, pente, superficie) moyennées sur l'aire du bassin.
> **Note importante :** Les données géophysiques sont sur une grille **différente** de CaSR. Cette étape a ses propres poids de grille.
> **Durée :** ~1–2 heures.

### 5.1 — Installer GDAL (requis pour le calcul des poids géophysiques)

```bash
# Activer Internet d'abord
Internet
export https_proxy=http://webproxy.science.gc.ca:8888/
export http_proxy=http://webproxy.science.gc.ca:8888/

# Installer la version exacte correspondant aux serveurs ECCC
uv pip install gdal==3.11.3
uv pip install shapely geopandas fiona
```

### 5.2 — Modifier et lancer le notebook de poids géophysiques

Ouvre `01_grid_weight_geophy.ipynb` dans VS Code et modifie la **2e cellule** après les importations :

```python
# Dans 01_grid_weight_geophy.ipynb — 2e cellule :
nc_file    = "/home/ega001/store8/machine_learning/prep_static_attrs/geophys.nc"
station_list = "$PROJ_DIR/data/basins/all_212_basins.txt"
shape_dir  = "$PROJ_DIR/data/shape_files/"
out_dir    = "$PROJ_DIR/geophys/grid_weights/"
```

```bash
cd "$PROJ_DIR/scripts"
ipython 01_grid_weight_geophy.ipynb
```

> **Vérifie après :** Le script génère aussi `main.txt` avec les superficies de drainage. **Inspecte ce fichier** et confirme que les identifiants de stations sont corrects. Une superficie à zéro ou manquante causera une erreur à l'étape 5.4.

### 5.3 — Calculer les attributs pondérés

Modifie les chemins dans ces 3 fichiers :
- `weight_geophys_master.ipynb` : liste de bassins, répertoire `geophys/stat_attrs/`, lien vers `weight_geophys.txt`
- `weight_geophys.txt` : chemin vers l'activation de l'environnement Python
- `weight_geophys.ipynb` : chemins des données géophysiques et de sortie

```bash
ipython 02_weight_geophys_master.ipynb
```

> Ce script soumet un job par bassin. Les résultats apparaissent dans `geophys/stat_attrs/`.

### 5.4 — Combiner tous les attributs en un seul fichier

Modifie les chemins au début de `03_comb_stat_attrs_and_add_others.py` puis :

```bash
ipython 03_comb_stat_attrs_and_add_others.py
```

### 5.5 — Vérification ✅

```bash
# Le message de succès doit apparaître dans la sortie :
# "Latitudes were found for all stations. The unique static attributes file should be complete"

ls "$PROJ_DIR/geophys/stat_attrs/static_attributes.csv"
head -3 "$PROJ_DIR/geophys/stat_attrs/static_attributes.csv"
```

> **Si le message de latitudes manquantes apparaît :** Identifie les stations concernées dans le fichier de sortie indiqué par la variable `outf_store_miss_lats` dans le script. Trouve leurs coordonnées sur les sites WSC ou USGS et fournis un fichier complémentaire via la variable `additional_file`.

---

## ÉTAPE 6 — Calculer les poids de grille pour les forçages CaSR

> **Ce que ça fait :** Calcule quelle fraction de chaque pixel CaSR (10km × 10km) appartient à chaque bassin. Ces poids servent à calculer la précipitation/température "sur" chaque bassin.
> **Durée :** ~30–60 minutes.

### 6.1 — Modifier les chemins dans le notebook

Ouvre `02_grid_weight.ipynb` et modifie la **1ère cellule** :

```python
# Dans 02_grid_weight.ipynb — 1ère cellule :
station_list = "$PROJ_DIR/data/basins/all_212_basins.txt"
shape_dir    = "$PROJ_DIR/data/shape_files/"
nc_file      = "$PROJ_DIR/data/NETCDF_FOLDER/2000010212.nc"   # n'importe quel fichier .nc
out_dir      = "$PROJ_DIR/data/grid_weights/"
grid_weight_script = "$PROJ_DIR/scripts/derive_grid_weights_new_WSC.py"
```

> **Note sur l'argument `-c` :** La commande Python dans le notebook ressemble à :
> ```python
> !python {grid_weight_script} -i {nc_file} -d "rlon,rlat" -v "lon,lat" -r {shape} -a -c "StationNum" -o {out}
> ```
> L'argument `-c` doit correspondre à la colonne d'ID dans le shapefile :
> - Stations **canadiennes WSC** → `-c "StationNum"`
> - Stations **américaines USGS** → `-c "GAGE_ID"`
> - Shapefiles **CaPA** → `-c "SubId"`
>
> **Si tes 212 bassins mélangent CA et US :** Fais cette étape deux fois avec deux listes séparées (une CA, une US) et l'argument `-c` adapté à chaque fois.

### 6.2 — Lancer le calcul

```bash
cd "$PROJ_DIR/scripts"
ipython 02_grid_weight.ipynb
```

### 6.3 — Vérification ✅

```bash
echo "Poids de grille calculés : $(ls $PROJ_DIR/data/grid_weights/ | wc -l) fichiers"
```

> **Attendu :** Autant de fichiers que de stations dans ta liste (212 si tu as toutes les 212 stations).

---

## ÉTAPE 7 — Générer les cartes de poids (weight maps)

> **Ce que ça fait :** Crée les séries temporelles horaires de forçages pour chaque bassin en appliquant les poids de grille à chaque fichier NetCDF. C'est l'étape la **plus longue et la plus coûteuse** en calcul.
> **Durée :** Plusieurs heures à plusieurs jours selon le nombre de bassins et la disponibilité du cluster.
> **Méthode :** Un job `ord_soumet` indépendant est soumis pour chaque bassin.

### 7.1 — Modifier les 3 fichiers de configuration

**Fichier 1 : `03a_weight_map_master.ipynb`**
```python
# Dans le notebook maître, modifier :
basin_file  = "$PROJ_DIR/data/basins/all_212_basins.txt"
listing_dir = "$PROJ_DIR/listings/"

# Dans la commande ord_soumet, vérifier les chemins vers :
# - weight_map.txt (script de soumission)
# - le répertoire de travail courant
```

**Fichier 2 : `weight_map.txt`**
```bash
# Dans weight_map.txt, changer la ligne d'activation de l'environnement :
source /home/$USERID/$SSTORE/machine_learning/neuralhydrology/.venv/bin/activate

# Et vérifier que le chemin vers ipython dans ce script pointe vers ton répertoire scripts
```

**Fichier 3 : `weight_map_basin.ipynb`**
```python
# Dans weight_map_basin.ipynb :
nc_dir       = "$PROJ_DIR/data/NETCDF_FOLDER/"
weights_dir  = "$PROJ_DIR/data/grid_weights/"
output_dir   = "$PROJ_DIR/time_series_csv/"   # séries horaires intermédiaires

# Période à traiter :
start_year = 2000;  end_year = 2024   # traite jusqu'à 2023 inclus
start_month = 1;    end_month = 13    # 1 à 13 = tous les 12 mois
start_day = 1;      end_day = 32      # 1 à 32 = tous les jours
```

### 7.2 — TEST AVEC UN SEUL BASSIN AVANT DE TOUT LANCER

> **Recommandation forte :** Lance d'abord le notebook pour **un seul bassin** pour vérifier que tout fonctionne, avant de soumettre les 212 jobs.

Modifie temporairement `03a_weight_map_master.ipynb` pour n'utiliser qu'un seul bassin (ex: `02GA010`), lance-le, puis vérifie le fichier de listing pour ce bassin dans `listings/`. Si pas d'erreur, modifie la liste pour inclure tous les bassins.

### 7.3 — Lancer pour tous les bassins

```bash
cd "$PROJ_DIR/scripts"
ipython 03a_weight_map_master.ipynb
```

> Chaque bassin génère son propre job. Tu peux suivre l'avancement avec les fichiers dans `listings/`.

### 7.4 — Cas spécial : Grands bassins (02KF005, 02KF009)

Ces deux bassins sont trop grands pour être traités dans un seul job. Traite-les séparément :

```bash
ipython 03b1_weight_map_bigbasin_master.ipynb   # traite par morceaux
ipython 03b2_merge_bigbasin_chunks.ipynb         # fusionne les morceaux
```

> **Attention :** Utilise la version **corrigée** de `merge_bigbasin_chunks.ipynb` depuis :
> `/home/ega001/bin/package_scripts/LSTM_scripts/EG_main_scripts/`
> La version de Marie Babineau peut produire des valeurs incorrectes lors de la fusion.

> **Si un job dépasse le walltime :** Tu recevras un email avec l'ID du bassin comme nom de session. Relance uniquement ce bassin.

### 7.5 — Vérification ✅

```bash
echo "Séries horaires générées : $(ls $PROJ_DIR/time_series_csv/ | wc -l) fichiers"
# Attendu : autant de fichiers que de stations
```

---

## ÉTAPE 8 — Générer les forçages dynamiques journaliers

> **Ce que ça fait :** Convertit les séries horaires en moyennes journalières (période 12h → 12h UTC) et génère aussi les indices climatiques. C'est la dernière étape de prétraitement avant l'entraînement.
> **Durée :** ~1–3 heures selon le nombre de bassins.

### 8.1 — Copier les fichiers nécessaires dans la structure d'expérience

```bash
# Attributs statiques
cp "$PROJ_DIR/geophys/stat_attrs/static_attributes.csv" \
   "$PROJ_DIR/EXPERIMENT/$EXPERIMENT_NAME/attributes/"

# Liste des bassins
cp "$PROJ_DIR/data/basins/all_212_basins.txt" \
   "$PROJ_DIR/EXPERIMENT/$EXPERIMENT_NAME/basins/"
```

### 8.2 — Modifier le notebook `04_forcings.ipynb`

Ouvre `04_forcings.ipynb` et adapte la **1ère cellule** :

```python
# Dans 04_forcings.ipynb :
data_dir       = "$PROJ_DIR/EXPERIMENT/$EXPERIMENT_NAME"   # dossier avec attributes/, basins/, time_series/
hourly_csv_dir = "$PROJ_DIR/time_series_csv/"              # séries horaires de l'étape 7
discharge_dir  = "$PROJ_DIR/data/discharge_data/"          # débits observés

# Noms des fichiers de débits (vérifier les noms exacts) :
# ligne ~25 de la 2e cellule — adapter si les noms diffèrent
```

### 8.3 — Lancer dans une session interactive

```bash
qsub -I -lselect=1:ncpus=1:mpiprocs=1:ompthreads=1:mem=32gb -l walltime=6:0:0
# (attendre la session puis réactiver l'environnement)
export USERID=$(whoami)
export SSTORE="store8"
source $HOME/.local/bin/env
source "/home/$USERID/$SSTORE/machine_learning/neuralhydrology/.venv/bin/activate"

cd "$PROJ_DIR/scripts"
ipython 04_forcings.ipynb
```

### 8.4 — Gérer le fichier climate_indices.csv

Le script génère automatiquement `climate_indices.csv` dans `attributes/`. Ce fichier est calculé sur **toute la période disponible**. Si tu veux des indices climatiques uniquement sur la période de calibration (recommandé pour une expérience propre) :

```bash
# Créer un sous-dossier pour les différentes versions
mkdir -p "$PROJ_DIR/EXPERIMENT/$EXPERIMENT_NAME/attributes/climate_variants/"

# Renommer la version calculée sur toute la période
mv "$PROJ_DIR/EXPERIMENT/$EXPERIMENT_NAME/attributes/climate_indices.csv" \
   "$PROJ_DIR/EXPERIMENT/$EXPERIMENT_NAME/attributes/climate_variants/climate_indices_2000_2024.csv"

# Créer un lien symbolique vers la version voulue
ln -s "$PROJ_DIR/EXPERIMENT/$EXPERIMENT_NAME/attributes/climate_variants/climate_indices_2000_2024.csv" \
      "$PROJ_DIR/EXPERIMENT/$EXPERIMENT_NAME/attributes/climate_indices.csv"
```

> **Pourquoi :** NeuralHydrology plante s'il trouve plusieurs fichiers `climate_indices*.csv` dans le dossier `attributes/`. Le lien symbolique garantit qu'il n'en voit qu'un.

### 8.5 — Vérification ✅

```bash
echo "Séries journalières :"
ls "$PROJ_DIR/EXPERIMENT/$EXPERIMENT_NAME/time_series_csv/" | wc -l

echo "Attributs statiques :"
ls "$PROJ_DIR/EXPERIMENT/$EXPERIMENT_NAME/attributes/"

echo "Vérification structure complète :"
ls "$PROJ_DIR/EXPERIMENT/$EXPERIMENT_NAME/"
```

> **Attendu :** Les dossiers `attributes/`, `basins/`, `time_series/`, `time_series_csv/` doivent tous exister et contenir des fichiers.

---

## ÉTAPE 9 — Entraîner le modèle LSTM

> **Ce que ça fait :** Entraîne 10 modèles LSTM (10 "membres d'ensemble") pendant 30 epochs chacun sur GPU. Les serveurs GPU ont un walltime de 6–10h (~15–20 epochs par session), donc des reprises sont souvent nécessaires.
> **Durée :** 2–3 sessions GPU de 6h pour 10 membres × 30 epochs.
> **Où ça tourne :** Sur les serveurs GPU (Moira ou Conrad), pas sur ppp7/ppp8.

### 9.1 — Récupérer les fichiers seed (configs des 10 membres)

```bash
# Demander les bons fichiers seed à Étienne Gaborit !
# Voici où les trouver :
cp /home/ega001/bin/package_scripts/LSTM_scripts/EG_main_scripts/seed_*.yml \
   "$PROJ_DIR/EXPERIMENT/Apply_$EXPERIMENT_NAME/ens_configs/"
```

### 9.2 — Adapter chaque seed file

> **Attention :** Tu dois modifier **les 10 fichiers** `seed_1.yml` à `seed_10.yml`. Les paramètres clés à changer dans chacun :

```yaml
# Dans chaque seed_X.yml — adapter ces lignes :
data_dir: /home/TON_USER/TON_SSTORE/GripGl_LSTM/new_bassin/EXPERIMENT/Gl_Lstm_Exp1
                          # ^ chemin ABSOLU vers le dossier contenant attributes/, time_series/, etc.

experiment_name: V31_30ep-256N_ens1   # changer le numéro (ens1 à ens10)

train_basin_file: /home/TON_USER/TON_SSTORE/GripGl_LSTM/new_bassin/EXPERIMENT/Gl_Lstm_Exp1/basins/all_212_basins.txt
validation_basin_file: # même chemin
test_basin_file: # même chemin

train_start_date: "01/01/2000"
train_end_date:   "31/12/2017"      # période de calibration
validation_start_date: "01/01/2000"
validation_end_date:   "31/12/2017"
test_start_date: "01/01/2018"
test_end_date:   "31/12/2024"       # période de validation

epochs: 30          # ← NE PAS MODIFIER avant une reprise (c'est la CIBLE TOTALE)
seq_length: 365     # fenêtre temporelle de 1 an
hidden_size: 256    # taille du modèle
```

> **Règle absolue sur `epochs: 30` :** Cette valeur est la **cible totale d'epochs**, pas le nombre d'epochs supplémentaires. Si l'entraînement s'arrête à l'epoch 15 et que tu relances, le code calculera automatiquement qu'il reste 15 epochs à faire. **Ne change jamais cette valeur avant une reprise.**

### 9.3 — Préparer le script d'entraînement `train_ensemble.ipynb`

Remplace le contenu du notebook `train_ensemble.ipynb` par :

```python
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path("/home/TON_USER/TON_SSTORE/machine_learning/neuralhydrology")))
from neuralhydrology.nh_run_scheduler import schedule_runs
from neuralhydrology.nh_run import start_run

CONFIG_DIR = Path("ens_configs/")   # dossier avec les seed_X.yml
RUNS_DIR   = Path("runs/")          # dossier où sont stockés les runs

if torch.cuda.is_available():
    # === PREMIER LANCEMENT (depuis les seed files) ===
    schedule_runs(
        mode='train',
        directory=CONFIG_DIR,
        gpu_ids=[0, 1, 2, 3],   # adapter selon les GPUs disponibles
        runs_per_gpu=1,
    )
else:
    # Fallback CPU — un modèle à la fois
    start_run(config_file=CONFIG_DIR / "seed_1.yml", gpu=-1)
```

### 9.4 — Préparer le script PBS `qsub_GPU_train_ensemble3.pbs`

Copie le script PBS depuis :
```bash
cp /home/ega001/bin/package_scripts/LSTM_scripts/EG_main_scripts/qsub_GPU_train_ensemble3.pbs \
   "$PROJ_DIR/EXPERIMENT/Apply_$EXPERIMENT_NAME/"
```

Modifie le script PBS pour utiliser le **nouvel environnement** :
```bash
# Dans qsub_GPU_train_ensemble3.pbs — REMPLACER l'ancien bloc d'environnement par :
source /home/$USERID/$SSTORE/machine_learning/neuralhydrology/.venv/bin/activate

# Et vérifier que le chemin vers ipython et train_ensemble.ipynb est correct
# Le script doit se trouver dans Apply_Gl_Lstm_Exp1/
```

### 9.5 — Lancer l'entraînement sur GPU

```bash
# Se connecter au serveur GPU
ssh moira   # ou: ssh conrad

# Aller dans le dossier de l'expérience
cd "$PROJ_DIR/EXPERIMENT/Apply_$EXPERIMENT_NAME/"

# Lancer le job
qsub qsub_GPU_train_ensemble3.pbs

# Vérifier que le job est bien en queue
jobst -u $USER
```

> **Si le job reste en queue trop longtemps :** Les GPUs sont peut-être tous occupés. Alternative : entraîner sur CPU (plus lent mais fonctionne).

### 9.6 — Alternative CPU (si pas de GPU disponible)

```bash
# Sur ppp7 ou ppp8
qsub -I -lselect=1:ncpus=12:mpiprocs=1:ompthreads=1:mem=10gb -l walltime=12:0:0
source $HOME/.local/bin/env
source "/home/$USERID/$SSTORE/machine_learning/neuralhydrology/.venv/bin/activate"

cd "$PROJ_DIR/EXPERIMENT/Apply_$EXPERIMENT_NAME/"

# Dans train_ensemble.ipynb, utiliser le fallback CPU (le else du bloc) :
# start_run(config_file=CONFIG_DIR / "seed_1.yml", gpu=-1)
# Répéter pour seed_2.yml ... seed_10.yml
ipython train_ensemble.ipynb
```

### 9.7 — Reprendre un entraînement interrompu (continue training)

Après 6h de walltime, certains membres n'auront pas atteint les 30 epochs. Pour reprendre :

**Modifier `train_ensemble.ipynb`** — changer `mode='train'` en `mode='continue_training'` et `directory=CONFIG_DIR` en `directory=RUNS_DIR` :

```python
if torch.cuda.is_available():
    # === REPRISE DES RUNS INCOMPLETS ===
    schedule_runs(
        mode='continue_training',   # ← changer ici
        directory=RUNS_DIR,         # ← et ici (pointer vers runs/, pas ens_configs/)
        gpu_ids=[0, 1, 2, 3],
        runs_per_gpu=1,
    )
```

Puis relancer :
```bash
qsub qsub_GPU_train_ensemble3.pbs
```

> Le scheduler détecte automatiquement quels membres n'ont pas atteint 30 epochs et les relance. Ceux qui sont déjà terminés sont ignorés.

### 9.8 — Vérifier l'avancement de l'entraînement

```bash
# Voir l'état de chaque membre :
python3 - << 'EOF'
from pathlib import Path
from neuralhydrology.nh_run_scheduler import _get_last_completed_epoch, _get_target_epochs_from_config

runs_dir = Path("runs/")
for run_dir in sorted(runs_dir.iterdir()):
    if not run_dir.is_dir():
        continue
    last   = _get_last_completed_epoch(run_dir)
    target = _get_target_epochs_from_config(run_dir)
    status = "✅ COMPLET" if last >= target else f"⏳ {last}/{target} epochs"
    print(f"{run_dir.name}: {status}")
EOF
```

### 9.9 — Vérification finale ✅

Avant de passer à l'évaluation, confirme que **tous les 10 membres** ont atteint 30 epochs :

```bash
ls "$PROJ_DIR/EXPERIMENT/Apply_$EXPERIMENT_NAME/runs/" | wc -l
# Attendu : 10 dossiers

# Pour chaque dossier, vérifier la présence de model_epoch030.pt :
for d in "$PROJ_DIR/EXPERIMENT/Apply_$EXPERIMENT_NAME/runs/"*/; do
    last=$(ls "$d"model_epoch*.pt 2>/dev/null | sort | tail -1)
    echo "$(basename $d): dernier checkpoint = $(basename ${last:-AUCUN})"
done
```

> Si certains membres ont des checkpoints dans des sous-dossiers `continue_training_from_epochXXX/`, le nouvel environnement les trouvera automatiquement. **Ne copie pas les `.pt` manuellement** (ancienne méthode, plus nécessaire).

---

## ÉTAPE 10 — Évaluer le modèle

> **Ce que ça fait :** Applique les 10 modèles entraînés sur la période de validation (données non vues pendant l'entraînement) et calcule les métriques NSE et KGE pour chaque bassin.
> **Durée :** ~1–2 heures sur GPU.

### 10.1 — Modifier les configs de chaque run avant l'évaluation

Dans chaque dossier `runs/V31_30ep-256N_ensX_XXXX/`, il y a un fichier `config.yml` généré automatiquement pendant l'entraînement. Modifie-le pour pointer vers les bonnes données :

```yaml
# Dans runs/V31_30ep-256N_ensX_XXXX/config.yml :
data_dir: /chemin/absolu/vers/EXPERIMENT/Gl_Lstm_Exp1    # vérifier ce chemin
test_basin_file: /chemin/absolu/vers/basins/all_212_basins.txt
test_start_date: "01/01/2018"    # début de la période de validation
test_end_date:   "31/12/2024"    # fin de la période de validation
```

> **Si tu veux évaluer sur la période de calibration :** Change `test_start_date` / `test_end_date` aux dates de calibration.

### 10.2 — Copier et adapter le script d'évaluation

```bash
cp /home/ega001/bin/package_scripts/LSTM_scripts/EG_main_scripts/qsub_GPU_eval_ensemble.pbs \
   "$PROJ_DIR/EXPERIMENT/Apply_$EXPERIMENT_NAME/"

# Modifier le script PBS :
# - chemin vers les runs/
# - chemin vers les résultats results/
# - activer le bon environnement (nouvel .venv)
```

### 10.3 — Lancer l'évaluation sur GPU

```bash
ssh moira   # ou: ssh conrad
cd "$PROJ_DIR/EXPERIMENT/Apply_$EXPERIMENT_NAME/"
qsub qsub_GPU_eval_ensemble.pbs
jobst -u $USER
```

### 10.4 — Vérification des résultats ✅

```bash
ls "$PROJ_DIR/EXPERIMENT/Apply_$EXPERIMENT_NAME/results/"
```

> **Attendu :** Un dossier de résultats par bassin, contenant les débits simulés et les métriques (NSE, KGE) pour la période de test.

### 10.5 — Interpréter les métriques

| NSE | Interprétation |
|-----|----------------|
| > 0.75 | Excellent |
| 0.65 – 0.75 | Bon |
| 0.50 – 0.65 | Acceptable |
| < 0.50 | Médiocre |
| < 0 | Pire que la moyenne historique |

> D'après l'article GRIP-GL (Mai et al., 2022), le LSTM entraîné sur 141 bassins de calibration obtient un NSE médian de ~0.75 sur les bassins de calibration et ~0.65 sur les bassins de validation non vus pendant l'entraînement.

---

## Résumé des commandes à retenir

| Étape | Commande principale | Durée |
|-------|--------------------|----|
| Activation session | `source $HOME/.local/bin/env && source $NH_DIR/.venv/bin/activate` | Instantané |
| 1 — Environnement | `uv sync --all-groups && uv pip install -e .` | 30–45 min |
| 2 — Structure | `mkdir -p ... && cp -rp ...` | 1–2h (NetCDF) |
| 3 — Shapefiles | `ipython 01a_merge_bassin.ipynb` | 15–30 min |
| 4 — Débits | `./01b1_*.py && ./01b2_*.py` | 15 min (si déjà copiés : 0 min) |
| 5 — Attrs statiques | `ipython 01_grid_weight_geophy.ipynb && ...` | 1–2h |
| 6 — Poids grille | `ipython 02_grid_weight.ipynb` | 30–60 min |
| 7 — Weight maps | `ipython 03a_weight_map_master.ipynb` | **Plusieurs heures** |
| 8 — Forçages | `ipython 04_forcings.ipynb` | 1–3h |
| 9 — Entraînement | `qsub qsub_GPU_train_ensemble3.pbs` | **2–3 sessions GPU** |
| 10 — Évaluation | `qsub qsub_GPU_eval_ensemble.pbs` | 1–2h |

---

## Problèmes fréquents et solutions

### "Disk quota exceeded" pendant l'installation
→ Vérifie que les liens symboliques vers le sitestore sont en place (étape 1.3). Supprime les fichiers dans `$HOME/.cache/` et recommence.

### Un shapefile manque pour une station
→ La famille WSC n'a pas encore été subdivisée. Utilise QGIS sur Windows pour subdiviser le gros fichier source, puis transfère via FileZilla/scp.

### Le job d'entraînement se termine après ~15 epochs
→ Normal — le walltime GPU est de 6h. Relance avec `mode='continue_training'` (étape 9.7).

### `continue_training` plante avec PyTorch ≥ 2.6
→ Bug connu. Le correctif est d'ajouter `weights_only=False` dans `basetrainer.py` et `tester.py`. Voir [DIAGNOSTIC_CONTINUE_TRAINING.md](external_docs/DIAGNOSTIC_CONTINUE_TRAINING.md) pour les lignes exactes à modifier.

### Le scheduler ne reconnaît pas `continue_training`
→ Bug connu dans l'ancienne version. Le correctif est documenté dans [DIAGNOSTIC_CONTINUE_TRAINING.md](external_docs/DIAGNOSTIC_CONTINUE_TRAINING.md).

### `climate_indices.csv` — NeuralHydrology plante
→ Il y a plusieurs fichiers `climate_indices*.csv` dans `attributes/`. Déplace les variantes dans un sous-dossier et crée un lien symbolique unique (étape 8.4).

### "Latitudes were found for all stations" — message absent
→ Certains bassins manquent de coordonnées dans `static_attributes.csv`. Identifie-les dans le fichier `outf_store_miss_lats`, trouve leurs coordonnées sur les sites WSC/USGS, et fournis un fichier additionnel via la variable `additional_file` dans `03_comb_stat_attrs_and_add_others.py`.

### Le bassin 02KF005 ou 02KF009 n'a pas de série temporelle
→ Ces bassins sont trop grands pour un seul job. Utilise les scripts `bigbasin` (étape 7.4).
