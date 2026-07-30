# Comprendre le projet GRIP-GL LSTM — du début à la fin

> Ce document ne parle pas de code. Il explique **pourquoi** chaque étape existe, ce qu'elle produit, et pourquoi l'étape suivante en dépend.

---

## C'est quoi ce projet en une phrase ?

On entraîne un modèle de machine learning (LSTM) pour **prédire le débit des rivières** (combien d'eau s'écoule à un point de mesure) à partir des données météo historiques, sur **212 stations de jaugeage** autour des Grands Lacs et du Saint-Laurent.

Ce projet s'inscrit dans une étude scientifique internationale appelée **GRIP-GL** (Great Lakes Runoff Intercomparison Project Phase 4), qui a comparé 13 modèles hydrologiques différents sur ce même domaine géographique. Résultat de l'article : **le LSTM surpasse tous les modèles physiques traditionnels**, même dans les tests de validation les plus exigeants. C'est pour ça qu'on l'implémente.

---

## Les concepts de base à comprendre (le vocabulaire)

Avant les étapes, il faut maîtriser quelques mots-clés qu'on utilise tout le temps :

### Bassin versant (bassin)
C'est la zone géographique dont toute l'eau de pluie finit par s'écouler vers **le même point de mesure** (la station de jaugeage). Si tu imagines un bol inversé posé sur la carte, tout ce qui tombe dedans coule vers le bas du bol. On a 212 de ces "bols" sur le domaine.

### Shapefile
Un fichier qui décrit la **forme géométrique** d'un bassin versant sur une carte — essentiellement un polygone qui trace la frontière du bassin. C'est le format standard en géomatique. Pour chacun de nos 212 bassins, on a un shapefile. Certains viennent du Canada (WSC), d'autres des États-Unis (USGS).

### Grille / Points de grille
Les données météo ne sont pas mesurées partout — elles sont produites sur une grille régulière, comme un quadrillage sur la carte. **CaSR 3.2** (notre source météo) produit des données toutes les 10 km × 10 km. Chaque case du quadrillage est un "point de grille" (ou pixel météo). Le domaine entier est couvert par des milliers de ces pixels.

### Fichiers NetCDF (.nc)
C'est le format de fichier scientifique qui stocke ces données météo sur la grille. Dans notre projet, il y a **un fichier .nc par jour** (de 2000 à 2024), chacun contenant les données horaires de toutes les variables météo (précipitation, température, vent, humidité, etc.) pour tous les pixels de la grille. Ces fichiers sont énormes et c'est pour ça qu'ils sont stockés sur le serveur ECCC.

### CaSR (Canadian Surface Prediction Archive Reanalysis)
C'est la **source des données météo**. CaSR 3.2 est une réanalyse météorologique produite par ECCC : on a reconstitué le temps qu'il a fait de 2000 à 2024 sur tout le Canada et le nord des États-Unis, avec une résolution de 10 km et une fréquence horaire. C'est le "forçage" qu'on va donner au modèle.

### Forçages (forcings)
En hydrologie, les "forçages" sont les variables météo qui **forcent** le cycle de l'eau : précipitation, température, vent, humidité, etc. Ce sont les **entrées** du modèle LSTM. Pour chaque bassin, on a besoin d'une valeur journalière moyenne de chaque forçage.

### Poids de grille (grid weights)
Le problème : la grille météo et les bassins versants ne correspondent pas — un bassin peut couvrir une partie d'un pixel, ou plusieurs pixels entiers. Pour calculer la précipitation **sur le bassin**, il faut faire une **moyenne pondérée** des pixels qui le recouvrent, proportionnellement à la fraction de chaque pixel qui est à l'intérieur du bassin.

Les **poids de grille** = les fractions d'intersection entre chaque pixel météo et chaque bassin. Par exemple : le bassin X est recouvert à 40% par le pixel A, 35% par le pixel B, et 25% par le pixel C.

### Cartes de poids (weight maps)
Une étape plus fine encore : au lieu de faire la pondération pixel-par-pixel sur toute la durée d'un coup (trop lourd en mémoire), on crée des **cartes spatiales** pré-calculées qui encodent ces poids. C'est l'étape la plus coûteuse en calcul.

### Attributs statiques
Ce sont les **caractéristiques physiques** de chaque bassin qui ne changent pas dans le temps : type de sol, proportion de forêt/culture/zones humides, altitude moyenne, pente, superficie, etc. Le LSTM reçoit ces informations en entrée pour "savoir" dans quel type de bassin il se trouve.

### Données de débit observé
Ce sont les **mesures réelles** de débit à chaque station de jaugeage. On les utilise pour entraîner le modèle (il apprend à reproduire ces observations) et pour évaluer ses performances après l'entraînement.

### LSTM
Long Short-Term Memory — un type de réseau de neurones conçu pour apprendre des **séquences temporelles**. Chaque jour, il reçoit les forçages météo du jour + les attributs statiques du bassin, et il prédit le débit. Sa mémoire interne lui permet de "se souvenir" des jours précédents (p. ex. un sol encore saturé d'une pluie de la semaine passée).

### Ensemble (10 membres)
On entraîne **10 copies du LSTM** avec des initialisations aléatoires différentes (seeds différentes). Chacune converge vers une solution légèrement différente. En production, on prend la moyenne des 10 prédictions — c'est plus robuste et ça donne une plage d'incertitude.

### NSE / KGE
Ce sont les **métriques de performance** qu'on calcule pour évaluer le modèle. Le NSE (Nash-Sutcliffe Efficiency) va de −∞ à 1 (1 = parfait, 0 = aussi bon que la moyenne historique). Le KGE (Kling-Gupta Efficiency) est similaire mais plus équilibré.

---

## Les 10 étapes du pipeline — dans l'ordre

```
[Environnement] → [Structure] → [Shapefiles] → [Débits] → [Attributs statiques]
     → [Poids de grille] → [Cartes de poids] → [Forçages] → [Entraînement] → [Évaluation]
```

---

### ÉTAPE 1 — Préparer l'environnement de calcul

**Ce qu'on fait :** On réserve une machine de calcul sur les serveurs ECCC (ppp7 ou ppp8), on configure les proxys pour accéder à internet, on installe Python + toutes les librairies nécessaires (NeuralHydrology, GDAL, etc.).

**Pourquoi c'est nécessaire :** Les serveurs ECCC ont des quotas de disque stricts sur le dossier personnel (`$HOME`). On doit rediriger les gros fichiers vers le `sitestore` (espace étendu). Sans ça, l'installation plante à mi-chemin.

**Fichiers clés :**
- Script : `etape1_init_gestion.sh`
- Environnement virtuel créé dans : `.../machine_learning/neuralhydrology/.venv/`

**Ce que ça produit :** Un environnement Python fonctionnel avec NeuralHydrology installé en mode modifiable (on peut changer le code source).

**Pourquoi l'étape suivante en dépend :** Tous les scripts de prétraitement tournent dans cet environnement. Sans lui, rien ne fonctionne.

---

### ÉTAPE 2 — Structurer le projet et centraliser les données

**Ce qu'on fait :** On crée toute l'arborescence de dossiers du projet et on copie depuis les serveurs d'autres chercheurs (ega001, mba002, shyd500) tous les fichiers dont on va avoir besoin : fichiers NetCDF, shapefiles, listes de stations, fichiers de débit, scripts.

**Pourquoi c'est nécessaire :** Les données sont dispersées sur plusieurs comptes utilisateurs. Si on les laisse là, nos scripts doivent pointer vers des chemins qui peuvent changer ou disparaître. En les centralisant chez nous, on a une copie stable et locale.

**Structure créée :**
```
GripGl_LSTM/new_bassin/
├── data/
│   ├── basins/          ← listes de stations (.txt, .csv)
│   ├── shape_files/     ← shapefiles finaux des 212 bassins
│   ├── NETCDF_FOLDER/   ← un .nc par jour de 2000 à 2024
│   ├── grid_weights/    ← poids de grille calculés à l'étape 6
│   └── discharge_data/  ← débits observés (étape 4)
├── EXPERIMENT/
│   └── Gl_Lstm_Exp1/
│       ├── attributes/  ← attributs statiques (étape 5)
│       ├── basins/      ← liste des bassins de l'expérience
│       └── time_series/ ← forçages journaliers par bassin (étape 8)
└── geophys/             ← intermédiaires géophysiques
```

**Ce que ça produit :** Une arborescence vide mais prête, et les données sources copiées localement.

**Pourquoi l'étape suivante en dépend :** Les étapes 3 à 8 lisent et écrivent dans cette arborescence. Si elle n'existe pas, tout échoue.

---

### ÉTAPE 3 — Préparer les shapefiles et les listes de stations

**Ce qu'on fait :** On sélectionne les bons shapefiles (un par bassin), on les copie dans `data/shape_files/`, et on génère les fichiers texte qui listent les identifiants de stations (par exemple `allbassin_shapefile_GLS.txt`).

**Le défi ici :** Les stations canadiennes (WSC) et américaines (USGS) ont des formats de shapefile différents et des noms de colonnes différents. En plus, certaines nouvelles stations (nouvelles dans l'expérience) ne sont pas encore dans les listes existantes. Le notebook `01a_merge_bassin.ipynb` fusionne les anciennes listes avec les nouvelles et harmonise tout.

#### Comment les shapefiles sont organisés — le système de "familles"

Les shapefiles canadiens WSC sont organisés en **familles** selon les 2 premiers caractères du numéro fédéral de la station. Par exemple, la station `02KF005` appartient à la famille `02` et son shapefile est dans le répertoire `MDA_ADP_02_split_new/`. La base de données WSC originale contient **un seul gros shapefile par famille** (toutes les stations d'une région ensemble). Pour pouvoir utiliser ces shapefiles station par station, il faut les subdiviser — et cette subdivision se fait avec **QGIS sur Windows** (pas disponible directement sur les serveurs Linux ECCC).

**Flux concret :**
1. Télécharger le gros shapefile WSC de la famille sur Windows (ex: `MDA_ADP_02_DrainageBasin.shp`)
2. Dans QGIS : changer le format en WGS84, puis "Split Vector Layer" sur le champ `StationNum`
3. Transférer les shapefiles individuels vers les serveurs Science via FileZilla
4. Les shapefiles US (USGS) sont disponibles directement dans `/home/ega001/store8/geophys_data/US_shapefiles/all/`

#### Le problème des identifiants de stations québécoises

Les stations québécoises ont **deux types de codes** :
- **Code CEHQ** (ex: `050901`) : utilisé par le gouvernement du Québec
- **Code fédéral / NHS** (ex: `02OA050`) : utilisé par Water Survey of Canada (WSC) et dans tous nos shapefiles

Le fichier `StationID_NHSvsQC.xlsx` sert de table de correspondance entre ces deux systèmes. Sans ce fichier, les stations québécoises seraient introuvables dans les shapefiles.

**Fichiers clés :**
- Script : `01a_merge_bassin.ipynb`
- Entrées : shapefiles sources dans `data/source_shapes/`, fichier `.obs` des nouvelles stations, `StationID_NHSvsQC.xlsx`
- Sorties : shapefiles sélectionnés dans `data/shape_files/`, listes de stations dans `data/basins/`, `WSC_gauges_GRIPGL.csv` mis à jour

**Pourquoi c'est nécessaire :** Les étapes de calcul des poids (étapes 5 et 6) prennent les shapefiles en entrée. Si un shapefile est manquant ou mal formaté, le bassin correspondant sera ignoré ou produira une erreur. Tout bassin sans shapefile = bassin absent de l'expérience.

**Pourquoi l'étape suivante en dépend :** La liste de stations produite ici est utilisée pour aller chercher les données de débit à l'étape 4.

---

### ÉTAPE 4 — Rassembler les débits observés

**Ce qu'on fait :** On télécharge les mesures de débit journalier pour chaque station, depuis la base de données Hydat pour le Canada (WSC) et depuis USGS pour les États-Unis. On les convertit en un format standardisé : débit journalier sur la période 12h UTC → 12h UTC (le lendemain).

**Pourquoi 12-12 UTC ?** Les fichiers CaSR émis à 12h UTC contiennent les prévisions des 24h suivantes. Pour aligner parfaitement les forçages météo avec les débits observés, on doit définir la "journée hydrologique" de 12h à 12h UTC plutôt que minuit à minuit.

**Fichiers clés :**
- Scripts : `01b1_get_obs_hydat_EG_new.py`, `01b2_convert_daily_WSC_in_12-12_UTC.py` (Canada)
- Scripts : `01c1_USGS_txtfiles_reading.py`, `01c2_convert_daily_USGS_in_12-12_UTC.py` (USA)
- Sorties dans : `data/discharge_data/`

**Ce que ça produit :** Un fichier de débit par station, couvrant la période 2000–2024, au format attendu par NeuralHydrology.

**Pourquoi l'étape suivante en dépend :** Sans les débits, on ne peut pas entraîner le modèle (il ne saurait pas quoi reproduire) ni l'évaluer.

---

### ÉTAPE 5 — Calculer les attributs statiques

**Ce qu'on fait :** Pour chaque bassin versant, on calcule des caractéristiques physiques moyennées sur sa superficie : type de sol (sable, limon, argile), teneur en matière organique, densité apparente, couverture du sol (forêt, cultures, zones humides), altitude moyenne, pente, superficie drainée, etc.

**Point crucial : une grille différente des forçages.** Les données géophysiques (sol, végétation, topographie) sont sur une grille **différente** de la grille CaSR des forçages météo. La grille géophysique NSRPS a une résolution d'environ 2.5 km. Il faut donc calculer des poids de grille **séparément** pour les attributs statiques, avec leur propre grille de référence.

**Comment ça marche, en 4 sous-étapes :**

**5.1 — Fichier géophysique** : On a besoin du fichier `geophys.nc` — c'est une version déjà convertie et préparée du fichier géophysique NSRPS/SVS2 contenant le sol, la végétation, la topographie, etc. Ce fichier existe déjà chez ega001 (`/home/ega001/store8/machine_learning/prep_static_attrs/geophys.nc`) et n'a pas besoin d'être recréé à moins de changer complètement de domaine géographique.

**5.2 — Poids de grille géophysiques** : Avec `01_grid_weight_geophy.ipynb` + `derive_grid_weights_geophy.py`, on calcule quelle fraction de chaque pixel géophysique (~2.5km) appartient à chaque bassin — exactement le même principe qu'à l'étape 6, mais sur la grille géophysique. **Attention :** Cette sous-étape génère aussi un fichier `basin_areas.txt` avec la superficie drainée de chaque bassin. Ce fichier est utilisé à la sous-étape 5.4. Vérifier que toutes les superficies sont bien calculées (sinon les trouver manuellement sur les sites WSC ou USGS).

**5.3 — Attributs pondérés** : Avec `weight_geophys_master.ipynb` → `weight_geophys.ipynb`, on applique les poids calculés pour obtenir la valeur moyenne de chaque variable géophysique pour chaque bassin. Ce script soumet un job `ord_soumet` par bassin.

**5.4 — Combinaison finale** : Avec `03_comb_stat_attrs_and_add_others.py`, on rassemble tous les attributs calculés + les coordonnées géographiques + les superficies en un seul fichier `static_attributes.csv`. **Vérifier à la fin** que le script confirme "Latitudes were found for all stations" — sinon certains bassins manquent de coordonnées et ne seront pas utilisables.

**Fichiers clés :**
- Fichier source géophysique : `geophys.nc` (chez ega001, déjà prêt)
- Scripts : `01_grid_weight_geophy.ipynb`, `derive_grid_weights_geophy.py`
- Scripts : `weight_geophys_master.ipynb`, `weight_geophys.ipynb`
- Script final : `03_comb_stat_attrs_and_add_others.py`
- Sortie finale : `static_attributes.csv` (une ligne par bassin, une colonne par attribut)

**Ce que ça produit :** Le fichier `static_attributes.csv` — une table où chaque ligne est un bassin et chaque colonne est un attribut physique. C'est ce que le LSTM reçoit pour "connaître" le bassin.

**Pourquoi l'étape suivante en dépend :** Sans les attributs statiques, le LSTM n'a aucune connaissance des caractéristiques physiques des bassins — il ne pourrait pas généraliser d'un bassin à l'autre.

---

### ÉTAPE 6 — Calculer les poids de grille pour les forçages

**Ce qu'on fait :** Pour chaque bassin versant, on calcule quelle fraction de chaque pixel CaSR (10km × 10km) tombe à l'intérieur du bassin. Ce sont les **poids de grille** pour les forçages météo.

**Concrètement :** Le script `derive_grid_weights_new_WSC.py` prend en entrée :
- Un fichier `.nc` de CaSR (pour connaître la géométrie de la grille)
- Le shapefile du bassin (pour connaître sa forme)
Et produit un fichier de poids qui dit : "Pour le bassin X, le pixel à (lon, lat) contribue à 0.23, le pixel voisin à 0.15, etc."

**Fichier clé :**
- Script orchestrateur : `02_grid_weight.ipynb`
- Script de calcul : `derive_grid_weights_new_WSC.py`
- Argument `-c` : le nom de la colonne du shapefile qui contient l'identifiant de la station ("`StationNum`" pour WSC, "`GAGE_ID`" pour USGS)
- Sorties dans : `data/grid_weights/` (un fichier par bassin)

**Pourquoi c'est nécessaire :** C'est la clé pour extraire les forçages météo *par bassin*. Sans ces poids, on ne peut pas faire la moyenne pondérée des pixels pour obtenir la précipitation ou la température "sur" un bassin.

**Pourquoi l'étape suivante en dépend :** Les cartes de poids (étape 7) utilisent ces poids comme base pour créer les structures de calcul plus détaillées.

---

### ÉTAPE 7 — Générer les cartes de poids (weight maps)

**Ce qu'on fait :** C'est l'étape la plus coûteuse en calcul. On génère des "cartes de poids" — des structures de données qui encodent comment agréger spatialement les forçages horaires CaSR pour chaque bassin et chaque pas de temps.

**Pourquoi une étape de plus ?** L'étape 6 dit "quel pixel contribue à quel pourcentage". L'étape 7 fait la **vérification et la compilation** de ces informations sous une forme que NeuralHydrology peut consommer directement. C'est computationnellement intensif parce qu'on traite 24 ans × 365 jours × toutes les heures × tous les bassins.

**Fichiers clés :**
- Script maître : `03a_weight_map_master.ipynb` (soumet un job `ord_soumet` par bassin)
- Script par bassin : `weight_map_basin.ipynb`
- Pour les grands bassins (trop grands pour le walltime) : `03b1_weight_map_bigbasin_master.ipynb` puis `03b2_merge_bigbasin_chunks.ipynb`
- Les résultats vont dans : `geophys/grid_weights/`

**Attention :** Chaque bassin est soumis comme un job indépendant sur le cluster. Les bassins très grands (ex: 02KF005, 02KF009) peuvent dépasser le temps limite et doivent être traités en morceaux puis fusionnés.

**Pourquoi l'étape suivante en dépend :** L'étape 8 (forçages dynamiques) utilise ces cartes de poids pour extraire les valeurs journalières de chaque forçage pour chaque bassin à partir des fichiers NetCDF bruts.

---

### ÉTAPE 8 — Générer les forçages dynamiques

**Ce qu'on fait :** Pour chaque jour de 2000 à 2024, on ouvre le fichier `.nc` CaSR du jour, on applique les cartes de poids pour agréger les données sur chaque bassin, et on calcule une **moyenne journalière** de chaque variable météo (précipitation, températures min/max, composantes du vent, etc.).

Le résultat final : pour chaque bassin, une **série temporelle journalière** de tous les forçages, sur toute la période — c'est exactement ce que NeuralHydrology attend en entrée.

**En parallèle :** Cette étape calcule aussi les **indices climatiques** (`climate_indices.csv`) : des statistiques résumées sur la période de calibration (précipitation moyenne annuelle, fraction de jours secs, etc.). Ce sont des attributs "statiques" supplémentaires calculés à partir des forçages.

**Piège important sur les indices climatiques :** Ces indices sont calculés sur la période couverte par les fichiers NetCDF disponibles dans `time_series_csv/`. Si on veut une calibration propre, les indices climatiques **doivent être calculés uniquement sur la période de calibration** (pas sur toute la période). Cela signifie qu'il faudrait potentiellement générer les séries temporelles deux fois : une fois sur la période de calibration seulement (pour calculer les indices), et une fois sur toute la période (pour l'entraînement et la validation). En pratique, on garde un seul dossier de forçages sur toute la période, mais avec **plusieurs versions du fichier `climate_indices.csv`** dans un sous-dossier, et un lien symbolique `climate_indices.csv` qui pointe vers la version voulue.

**Fichier clé :**
- Script : `04_forcings.ipynb`
- Entrées : fichiers `.nc` dans `data/NETCDF_FOLDER/`, cartes de poids de l'étape 7
- Sorties : séries temporelles journalières dans `EXPERIMENT/Gl_Lstm_Exp1/time_series/` et `time_series_csv/`
- Sorties aussi : `climate_indices.csv` dans `EXPERIMENT/Gl_Lstm_Exp1/attributes/`

**Pourquoi c'est la dernière étape de prétraitement :** Après cette étape, on a tout ce dont NeuralHydrology a besoin :
- ✅ Forçages journaliers par bassin (`time_series/`)
- ✅ Attributs statiques par bassin (`attributes/static_attributes.csv`)
- ✅ Débits observés par bassin (`data/discharge_data/`)
- ✅ Liste des bassins (`EXPERIMENT/Gl_Lstm_Exp1/basins/`)

---

### ÉTAPE 9 — Entraîner le modèle LSTM

**Ce qu'on fait :** On lance l'entraînement de 10 modèles LSTM (10 membres d'ensemble) pendant 30 epochs chacun, sur GPU.

**Comment ça marche :**
- Chaque membre a un fichier de configuration `seed_X.yml` (X de 1 à 10) qui définit les hyperparamètres : période d'entraînement/validation, variables d'entrée, taille du modèle (`hidden_size=256`), taux d'apprentissage, etc.
- À chaque epoch, le LSTM voit toutes les données des 212 bassins sur la période d'entraînement. Il prédit les débits, compare avec les observations (via la métrique NSE ou MSE), et ajuste ses 300 000 paramètres pour réduire l'erreur.
- Les serveurs GPU ont un walltime de 6h (~15 epochs). On doit souvent reprendre l'entraînement en plusieurs sessions.

**Contrainte technique :** Jusqu'à 4 membres entraînés simultanément sur 4 GPUs. On utilise le scheduler (`nh_run_scheduler.py`) pour orchestrer ça automatiquement, y compris la reprise des membres incomplets.

**Fichiers clés :**
- Configs : `Apply_Gl_Lstm_Exp1/ens_configs/seed_1.yml` à `seed_10.yml`
- Script d'entraînement : `train_ensemble.ipynb` + `qsub_GPU_train_ensemble.pbs`
- Résultats dans : `Apply_Gl_Lstm_Exp1/runs/` (un dossier par membre)

**Ce que ça produit :** 10 modèles entraînés, chacun avec 30 fichiers de poids sauvegardés (`model_epoch001.pt` à `model_epoch030.pt`).

**Pourquoi l'étape suivante en dépend :** Pour évaluer, on charge le modèle de l'epoch final (epoch 30) de chaque membre.

---

### ÉTAPE 10 — Évaluer le modèle

**Ce qu'on fait :** On applique les 10 modèles entraînés sur la **période de validation** (données que le modèle n'a pas vues pendant l'entraînement) et on calcule les métriques de performance pour chaque bassin.

**Ce qu'on mesure :**
- **NSE** (Nash-Sutcliffe Efficiency) : 1 = parfait, 0 = pas mieux que la moyenne
- **KGE** (Kling-Gupta Efficiency) : similaire, plus équilibré
- On évalue sur les 141 stations de calibration ET les 71 stations de validation

**Fichiers clés :**
- Script : `qsub_GPU_eval_ensemble.pbs`
- Résultats dans : `Apply_Gl_Lstm_Exp1/results/`

**Ce que ça produit :** Des fichiers de résultats contenant les débits simulés vs observés, les métriques par bassin, et l'ensemble de prédictions (moyenne + incertitude des 10 membres).

---

## Résumé visuel du pipeline

```
DONNÉES SOURCES
     │
     ├─ CaSR 3.2 NetCDF (fichiers .nc, un par jour, 2000–2024)
     ├─ Shapefiles des bassins (WSC Canada + USGS USA)
     ├─ Données géophysiques (sol, végétation, topographie)
     └─ Débits observés (Hydat Canada + USGS USA)
     │
     ▼
[Étape 3] Sélection et harmonisation des shapefiles + listes de stations
     │ → data/shape_files/ + data/basins/*.txt
     ▼
[Étape 4] Extraction des débits observés (conversion 12-12 UTC)
     │ → data/discharge_data/
     ▼
[Étape 5] Attributs statiques (moyenne géophysique par bassin)
     │ → EXPERIMENT/attributes/static_attributes.csv
     ▼
[Étape 6] Poids de grille pour les forçages (% pixel par bassin)
     │ → data/grid_weights/
     ▼
[Étape 7] Cartes de poids (structures de calcul, la plus longue)
     │ → geophys/grid_weights/
     ▼
[Étape 8] Forçages dynamiques (séries temporelles journalières par bassin)
     │ → EXPERIMENT/time_series/ + attributes/climate_indices.csv
     ▼
     ┌────────────────────────────────────┐
     │ NEURALHYDROLOGY A MAINTENANT TOUT │
     │ Ce dont il a besoin pour apprendre │
     └────────────────────────────────────┘
     ▼
[Étape 9] Entraînement LSTM (10 membres × 30 epochs, GPU)
     │ → Apply_Gl_Lstm_Exp1/runs/
     ▼
[Étape 10] Évaluation (NSE, KGE sur période de validation)
     └─ → Apply_Gl_Lstm_Exp1/results/
```

---

## Pourquoi c'est difficile à faire sur un nouveau bassin ?

Quand tu veux ajouter un bassin ou changer l'ensemble de bassins, **tout le pipeline doit être refait ou mis à jour** :

1. Le shapefile du nouveau bassin doit être trouvé et ajouté (étape 3)
2. Les données de débit de ce bassin doivent être extraites (étape 4)
3. Ses attributs statiques doivent être calculés (étape 5)
4. Ses poids de grille doivent être calculés (étape 6)
5. Sa carte de poids doit être générée — la plus longue (étape 7)
6. Ses forçages journaliers doivent être extraits (étape 8)
7. Et enfin on peut ré-entraîner ou affiner le modèle (étape 9)

C'est pour ça que les étapes sont conçues pour être modulaires : si tu changes un bassin, tu ne refais pas tout — tu reprends depuis l'étape 3 pour ce bassin spécifique, et tu complètes.

---

## Les fichiers de code que tu vas utiliser

| Étape | Notebook/Script | Rôle |
|-------|----------------|------|
| 3 | `01a_merge_bassin.ipynb` | Fusionner listes de stations, copier shapefiles |
| 4 | `01b1_get_obs_hydat_EG_new.py` | Extraire débits Canada |
| 4 | `01b2_convert_daily_WSC_in_12-12_UTC.py` | Convertir en 12-12 UTC |
| 4 | `01c1_USGS_txtfiles_reading.py` + `01c2_*` | Idem pour USA |
| 5 | `01_grid_weight_geophy.ipynb` | Poids géophysiques |
| 5 | `weight_geophys_master.ipynb` | Calculer attributs pondérés |
| 5 | `03_comb_stat_attrs_and_add_others.py` | Combiner en `static_attributes.csv` |
| 6 | `02_grid_weight.ipynb` + `derive_grid_weights_new_WSC.py` | Poids de grille forçages |
| 7 | `03a_weight_map_master.ipynb` | Soumettre jobs de cartes de poids |
| 8 | `04_forcings.ipynb` | Générer séries temporelles journalières |
| 9 | `train_ensemble.ipynb` + `qsub_GPU_train_ensemble.pbs` | Entraîner les 10 membres |
| 9 | `nh_run_scheduler.py` | Orchestrer l'entraînement multi-GPU |
| 10 | `qsub_GPU_eval_ensemble.pbs` | Évaluer les modèles |

Les fichiers de configuration de NeuralHydrology (`seed_1.yml` à `seed_10.yml`) définissent tout ce que le modèle doit savoir : quelles variables utiliser en entrée, quelle période d'entraînement, quelle taille de modèle, combien d'epochs.

---

## Bugs connus et état du projet (Juin 2026)

Des diagnostics ont identifié plusieurs problèmes dans le code NeuralHydrology adapté pour ce projet. Ils ont **tous été documentés et ont des correctifs** :

| Problème | Gravité | Impact | État |
|----------|---------|--------|------|
| `torch.load()` sans `weights_only=False` → crash PyTorch 2.6+ | Bloquant | Reprise d'entraînement impossible dans le nouvel env | Correctif : ajouter `weights_only=False` dans `basetrainer.py` et `tester.py` |
| Glob non-récursif → mauvais epoch de départ à la 2e reprise | Bloquant | Réentraîne des epochs déjà faits, écrase le travail | Correctif : chercher aussi dans les sous-dossiers `continue_training_from_epoch*/` |
| Scheduler sans mode `continue_training` | Bloquant | Impossible de reprendre plusieurs modèles en parallèle | Correctif : ajouter ce mode à `nh_run_scheduler.py` |
| `epochs:` non recalculé automatiquement | Modéré | Risque de surentraîner si on oublie de modifier le fichier | Correctif : calculer `epochs_restants = cible - epoch_actuel` |
| GPU attend les données (pas de `pin_memory`) | Performance | Entraînement 10–30% plus lent que nécessaire | Correctif dans `basetrainer.py` |
| Pas de précision mixte (AMP) | Performance | Entraînement 30–100% plus lent sur GPU modernes | Correctif : activer `torch.cuda.amp` |

Ces correctifs sont décrits dans les fichiers [DIAGNOSTIC_CONTINUE_TRAINING.md](external_docs/DIAGNOSTIC_CONTINUE_TRAINING.md) et [DIAGNOSTIC_PERFORMANCE.md](external_docs/DIAGNOSTIC_PERFORMANCE.md), et ils ont été partiellement appliqués dans la branche `test` du dépôt (les fichiers `basetrainer.py`, `tester.py`, `inputlayer.py` ont des modifications).

---

## Ce que l'article GRIP-GL a prouvé

Pour contextualiser pourquoi tout ce travail vaut la peine : l'article scientifique (Mai et al., 2022, *Hydrology and Earth System Sciences*) a montré que sur le domaine des Grands Lacs avec 212 stations :

- Le LSTM **surpasse les 12 autres modèles physiques** dans toutes les configurations de test
- Même en **validation spatiale** (prédire sur des bassins non vus pendant l'entraînement), le LSTM reste compétitif
- Les modèles physiques calibrés localement perdent drastiquement en performance quand on les transfère ailleurs — le LSTM beaucoup moins

C'est ce résultat qui justifie l'adoption du LSTM dans le Système de Prévision Hydrologique (SPH/DHPS) d'ECCC.
