#!/bin/bash

for i in `seq 0 9`; do
    echo $i
    convert +append morph.$i.{0..9}.png /tmp/$i.png
done
convert -append /tmp/{0..9}.png out.png
