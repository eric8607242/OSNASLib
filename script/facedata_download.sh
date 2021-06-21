mkdir -p ./data/face/train/ 
mkdir -p ./data/face/test/

wget -O ./data/face/train/CASIA-WebFace.tar.gz https://www.dropbox.com/s/6x03igvbvfwx24w/CASIA-WebFace.tar.gz?dl=1
wget -O ./data/face/test/LFW.tar.gz https://www.dropbox.com/s/d1y5o66dn8vcpvv/LFW.tar.gz?dl=1

tar zxvf ./data/face/train/CASIA-WebFace.tar.gz
tar zxvf ./data/face/test/LFW.tar.gz

mv ./CASIA-WebFace ./data/face/train && rm ./data/face/train/CASIA-WebFace.tar.gz
mv ./LFW ./data/face/test && rm ./data/face/test/LFW.tar.gz
