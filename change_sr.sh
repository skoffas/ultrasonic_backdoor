for d in $(find . -type d | grep \/)
do
	cd $d
	for f in $(ls | grep .wav)
	do
		sox $f -r 44100 tmp.wav
		mv tmp.wav $f
	done
	cd -
done
