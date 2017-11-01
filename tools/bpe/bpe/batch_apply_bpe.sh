if [ $# != 5 ];then
    echo $0 infile outfile codes lang line-per-file
    exit
fi
infile=$1
outfile=$2
codes=$3
lang=$4
line=$5
bin=`dirname $0`

outdir=`mktemp -d BPE.XXX`
[ ! -s $outdir ] && mkdir $outdir
split -l $line $infile $outdir/parts.

for f in $outdir/parts.??; do
    echo "BPE on $f "
    echo "$bin/apply_bpe.py -c $codes -l $lang < $f > $f.seg &"
    cat $f | $bin/rmrs.py | $bin/apply_bpe.py -c $codes -l $lang > $f.seg &
done

wait
cat $outdir/*.seg > $outfile
echo "BPE segmentation done"

