rm -r {0..10}
policies=("{4,4,4,4,1,4,4,4,4,0}" "{0,4,4,1,4,3,4,4,4,4}" "{3,4,4,2,4,4,4,4,4,4}" "{1,0,2,0,2,2,0,2,2,0}" "{0,4,2,1,2,0,0,1,2,3}" "{2,2,4,2,3,2,0,2,3,2}" "{2,2,0,4,4,1,3,3,4,2}" "{3,3,0,4,3,0,1,0,2,1}" "{1,4,1,1,4,4,0,3,3,1}" "{3,4,1,2,4,0,3,2,1,0}" "{2,4,3,4,2,4,3,3,2,2}" )

mkdir {0..10}
for i in {0..10}; do cp main.cpp $i/;done
for i in {0..10}; do cp field-read.h $i/;done
for j in {0..10}; do sed -i 's/int policy\[\] =;/int policy\[\] = '"${policies[j]}"';/' $j/main.cpp; done
for j in {0..10}; do sed -i 's/file("beta0/file("beta'"$j"'/' $j/main.cpp; done
for i in {0..10}; do cp ../perf $i/; done
for i in {0..10}; do cd $i/; sbatch perf; cd .. ; sleep 0.01; done
