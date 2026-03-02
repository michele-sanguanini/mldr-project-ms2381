Feb 20
docker compose exec app uv sync

docker compose exec app dvc init -f

docker compose exec app dvc remote add -d storage s3://dvc/dvcstore

docker compose exec app dvc add WA_Fn-UseC_-Telco-Customer-Churn.csv  

docker compose exec app dvc push