train:
	python pipelines/train_flow.py

run:
	docker-compose up -d

test:
	pytest tests/ -v

monitor:
	python src/monitor.py

deploy:
	ssh -i mlops-capstone-house-price_key.pem azureuser@20.205.24.174 \
		"docker pull jarvis4130/mlops-app && docker stop \$$(docker ps -q) && docker run -d -p 8000:8000 jarvis4130/mlops-app"
