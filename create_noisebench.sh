echo "Cloning CleanCoNLL repository to get the full CoNLL-03 dataset with CleanCoNLL annotations" 
git clone https://github.com/flairNLP/CleanCoNLL.git
cd CleanCoNLL
chmod u+x create_cleanconll_from_conll03.sh
SCRIPT_ROOT=. bash create_cleanconll_from_conll03.sh
mkdir -p ../data/cleanconll
mv ./data/cleanconll/* ../data/cleanconll
cd ..
rm -rf CleanCoNLL/*
rm -rf CleanCoNLL
mv ./data/cleanconll/cleanconll.test ./data/noisebench/clean.test

echo "Creating NoiseBench dataset files and train/dev splits" 
python scripts/generate_data_files.py
echo "Done!" 
