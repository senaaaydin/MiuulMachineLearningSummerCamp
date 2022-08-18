# Virtual Environment (Sanal Ortam) & Package Management (Paket Yönetimi)


'''

#sanal ortamların listelenmesi:
conda env list

#sanal ortam oluşturma:
conda create -n myenv

#sanal ortam aktif etme:
conda activate myenv

#Yüklü paketlerin listelenmesi:
conda list

#Aynı anda birden fazla paket yükleme
conda install numpy scipy

#paket silme
conda remove package_name

#belirli bir versiyona göre paket yükleme
conda install numpy=1.2

#paket yükseltme
conda upgrade conda

#tüm paketlerin yükseltilmesi
conda upgrade -- all




#pip : pypi (python package index) paket yönetim aracı

#paket yükleme:
pip install pandas

#paket yükleme versiyona göre:
pip install pandas==1.2.1

'''
