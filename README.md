# ML project

On Gitpod, run
pip install flask tensorflow pandas scikit-learn joblib

If you get an error installing tensowflow on gitpod, downgrade to python 3.11 (on gitpod python 3.12 is default, no compatible tensorflow is available): 
pyenv install 3.11
pyenv local 3.11


export FLASK_APP=app.py
flask run

