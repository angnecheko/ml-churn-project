# Churn Prediction API

Projet fil rouge d’industrialisation d’un modèle de machine learning pour illustrer la
prédiction du départ de clients d’un opérateur télécom. Le dépôt regroupe un jeu de données
synthétique, un entraînement scikit-learn, une API FastAPI, des tests, une image Docker et
une stack d’observabilité.

> **À garder en tête :** le jeu de données est pédagogique et synthétique. Les captures
> JMeter disponibles montrent un taux d’erreur de 100 % aux paliers observés ; elles
> témoignent d’un test exécuté, mais ne permettent pas de conclure à une capacité de charge
> validée.

## Fonctionnalités

- Génération et exploration de données synthétiques de churn.
- Comparaison de trois classifieurs et exposition d’un modèle via une API.
- Documentation interactive FastAPI et endpoint de métriques Prometheus.
- Tests d’API avec Pytest et contrôle du navigateur avec Selenium.
- Exécution de l’API, de Prometheus, Grafana et de la stack ELK avec Docker Compose.
- Workflow GitHub Actions pour lancer les tests sur `main` et `develop`.

## Architecture

```text
CSV → entraînement scikit-learn → models/model.pkl → FastAPI (/predict)
                                                    ├─ /metrics → Prometheus → Grafana
                                                    └─ logs → Logstash → Elasticsearch → Kibana
```

## Technologies

Python 3.12 · pandas · scikit-learn · FastAPI · Pydantic · Pytest · Selenium ·
Docker Compose · Prometheus · Grafana · Elasticsearch · Logstash · Kibana

## Structure du dépôt

| Chemin | Rôle |
|---|---|
| `api/` | Application FastAPI |
| `src/` | Génération des données et entraînement |
| `data/` | Dataset et sorties d’exploration |
| `models/` | Artefact du modèle |
| `notebooks/` | Exploration des données |
| `tests/` | Tests API et navigateur |
| `monitoring/` | Configuration Prometheus et Logstash |
| `docker-compose.yml`, `Dockerfile` | Conteneurisation et orchestration |
| `.github/workflows/` | Intégration continue |
| `docs/` | Documentation et livrables détaillés |

## Démarrage rapide avec Docker

Prérequis : Git, Docker Engine et Docker Compose v2.

```bash
git clone https://github.com/angnecheko/ml-churn-project.git
cd ml-churn-project
docker compose up -d --build
docker compose ps
curl --fail http://localhost:8001/health
```

La réponse attendue à `/health` est `{"status":"UP"}`.

| Service | Adresse locale |
|---|---|
| Documentation Swagger | <http://localhost:8001/docs> |
| Santé API | <http://localhost:8001/health> |
| Métriques API | <http://localhost:8001/metrics> |
| Prometheus | <http://localhost:9090> |
| Grafana | <http://localhost:3000> |
| Kibana | <http://localhost:5601> |

Elasticsearch est lié à `127.0.0.1:9200` dans la configuration Compose. Kibana est
également publié sur la boucle locale. Les autres ports ne sont pas tous restreints à
l’interface locale : sur un serveur partagé, configurer le pare-feu et l’authentification
avant toute exposition réseau.

## Exemple de prédiction

```bash
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35,
    "anciennete": 5,
    "revenu": 450000,
    "nb_reclamations": 2,
    "montant_mensuel": 25000,
    "retards_paiement": 2
  }'
```

La réponse fournit une classe (`CHURN` ou `NO_CHURN`), une probabilité, la version du
modèle et la durée de traitement. La valeur de probabilité dépend de l’artefact utilisé.

## Tests

```bash
python -m pip install -r requirements.txt
python -m pip install selenium
python -m pytest -v
```

Selenium est installé séparément dans le workflow CI et n’est pas actuellement inscrit
dans `requirements.txt`. Les quatre tests API ont été exécutés avec succès dans
l’environnement du projet. Le test Selenium requiert un navigateur Chrome compatible.

## Livrables

- [Documentation technique](docs/documentation-technique.docx)
- [Guide d’installation](docs/guide-installation.docx)
- [Rapport d’industrialisation](docs/rapport-industrialisation.docx)
- [Présentation finale modifiable](docs/presentation-finale.pptx)

Les documents distinguent les vérifications effectuées des limites observées. En
particulier, les rapports JMeter joints au rapport affichent 100 % d’erreurs ; leurs
temps de réponse ne doivent pas être interprétés comme des latences de requêtes réussies.

## Arrêter la stack

```bash
docker compose down
```

Cette commande conserve les volumes. `docker compose down -v` supprime aussi les données
persistantes de Grafana, Elasticsearch et Logstash.
