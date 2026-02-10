# Step 0. Change this to your campus ID
CAMPUSID='9084426882'
mkdir -p $CAMPUSID

# Step 1. (Optional) Any preprocessing step, e.g., downloading pre-trained word embeddings
wget https://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip

# Step 2. Train models on two datasets.
##  2.1. Run experiments on SST
PREF='sst'
python main.py \
    --train "data/${PREF}-train.txt" \
    --dev "data/${PREF}-dev.txt" \
    --test "data/${PREF}-test.txt" \
    --dev_output "${CAMPUSID}/${PREF}-dev-output.txt" \
    --test_output "${CAMPUSID}/${PREF}-test-output.txt" \
    --model "${CAMPUSID}/${PREF}-model.pt" \
    --emb_file ./wordEmbedding/crawl-300d-2M.vec \
    --word_drop 0.2 \
    --emb_drop 0.4 \
    --max_train_epoch 10 \
    --lrate_decay 0.0001 \
    --grad_clip 1.0

##  2.2 Run experiments on CF-IMDB
PREF='cfimdb'
python main.py \
    --train "data/${PREF}-train.txt" \
    --dev "data/${PREF}-dev.txt" \
    --test "data/${PREF}-test.txt" \
    --dev_output "${CAMPUSID}/${PREF}-dev-output.txt" \
    --test_output "${CAMPUSID}/${PREF}-test-output.txt" \
    --model "${CAMPUSID}/${PREF}-model.pt" \
    --max_train_epoch 10 \
    --hid_size 600 \
    --hid_layer 4 \
    --grad_clip 10.0


# Step 3. Prepare submission:
##  3.1. Copy your code to the $CAMPUSID folder
for file in 'main.py' 'model.py' 'vocab.py' 'setup.py'; do
	cp $file ${CAMPUSID}/
done
##  3.2. Compress the $CAMPUSID folder to $CAMPUSID.zip (containing only .py/.txt/.pdf/.sh files)
python prepare_submit.py ${CAMPUSID} ${CAMPUSID}
##  3.3. Submit the zip file to Canvas (https://canvas.wisc.edu/courses/292771/assignments)! Congrats!
