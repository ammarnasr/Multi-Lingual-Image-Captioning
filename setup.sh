# Install Requirements
pip -q install --upgrade --no-cache-dir gdown
pip install -q -r  requirements.txt
echo "====================Requirements are installed===================="

# Clone The Repo
git clone https://github.com/ammarnasr/Multi-Lingual-Image-Captioning.git
echo "====================Repo is cloned===================="

#Dwonlad and Extract The Flickr8k Dataset Arabic-English then delete the zip file
gdown --id 18sV6KoFIjPEh-y8x20_LIm-vE_wyIeMv
unzip -qq ./data.zip -d ./Multi-Lingual-Image-Captioning
rm ./data.zip
echo "====================Dataset (Flickr8k Dataset Arabic-English) is downloaded===================="

#Change Directory to the Repo
cd ./Multi-Lingual-Image-Captioning
gdown --id 1_uX98Y_ykKO_s2d0l-lUPgdhUrlsrRyj
unzip -qq ./checkpoints.zip -d ./
rm ./checkpoints.zip
echo "====================Checkpoints are downloaded: ===================="
ls ./checkpoints

#Print The Current Directory
echo "====================Current Directory: ===================="
pwd


#Run The Training
output_dir=./output_dir_1
python train.py --output_dir=$output_dir 


