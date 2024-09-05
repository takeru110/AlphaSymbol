# I have not tested this script yet.
export PATH=$PATH:/opt/conda/bin
mkdir /opt/pytorch/
cp ./setup/requirements.txt /opt/pytorch/
cd /opt/pytorch
pip install -U pip && pip install -r requirements.txt
cd ~