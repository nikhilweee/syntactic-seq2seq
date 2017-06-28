paste -d '$\n\n\n\n\n\n\n\n$\n\n#' \
 data/split-unique/src-test.txt.unique.split \
 data/split-unique/tgt-test.txt.unique.split* \
 data/unk-500/pred-test.txt.unique.split \
 data/unk-500-exclusive/pred-test.txt.unique.split \
 data/unk-600-lemma/pred-test.txt.unique.split \
 /dev/null \
 | sed 's/#/\n=====================/' |sed 's/\$/\n---------------------\n/' | less
