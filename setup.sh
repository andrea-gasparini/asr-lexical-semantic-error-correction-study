# TODO setup conda env
# TODO check submodules

# install python requirements
pip install -r requirements.txt

# install the libsndfile dependency of the soundfile package (https://pypi.org/project/soundfile/#installation)
sudo apt install libsndfile1

# install kenlm and its dependecies
cd src && bash setup_kenlm.sh