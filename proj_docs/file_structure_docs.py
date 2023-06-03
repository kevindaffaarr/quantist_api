"""
.
└── quantist_api
	├── __init__.py
	├── db.py # Database SQLAlchemy
	│	├── SQLAlchemy Database Connection
	│	├── get_dbs: dbs=SessionLocal()
	│	└── Database Model
	├── dp.py # Dependencies and Pydantics Schema
	│	├── Pydantics Database Schema
	│	└── Parameter Enum Class
	├── main.py
	│	├── Initiate Apps
	│	├── CORS Setting
	│	└── Include Router
	├── routers
	│	├── __init__.py
	│	├── param.py
	│	└── whaleanalysis.py
	└── proj_docs
		├── fastapi_docs.py
		└── file_structure_docs.py # This File
"""

"""
==========
Package Management
==========

pip install pipdeptree
pipdeptree

pip install --upgrade --upgrade-strategy eager -r requirements.txt
pip freeze > requirements.txt

Main Package (Prod):
python-dateutil
numpy
pandas
scikit-learn
SQLAlchemy
google-cloud-bigquery
sqlalchemy-bigquery
pydantic
gunicorn
fastapi
plotly
kaleido

python-dateutil numpy pandas scikit-learn SQLAlchemy google-cloud-bigquery sqlalchemy-bigquery pydantic gunicorn fastapi plotly kaleido

"""

