# I have not tested this script yet.
export PATH=$PATH:/opt/conda/bin
mv ./setup/requirement.txt /opt/pytorch
cd /opt/pytorch
pip install -U pip && pip install -r requirements.txt