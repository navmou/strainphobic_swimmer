betas=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
mkdir beta{0..10}
for i in {0..10}; do mkdir beta$i/{1..50}; done
for i in {0..10}; do cp main.cpp beta$i ;done
for j in {0..10}; do sed -i 's/beta = 0.0/beta = '"${betas[j]}"'/' beta$j/main.cpp; done
for i in {0..10}; do for k in {1..50}; do cp train beta$i/$k/; done; done
for i in {0..10}; do for k in {1..50}; do cp field-read.h beta$i/$k/; done ; done
for i in {0..10}; do for k in {1..50}; do cp beta$i/main.cpp beta$i/$k/; done ; done
for i in {0..10}; do for k in {1..50}; do cd beta$i/$k/; sbatch train; cd .. ; cd .. ; sleep 0.01; done; done
